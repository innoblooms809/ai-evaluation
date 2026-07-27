"""
api.py — Qwen Evaluation API (Multi-Question Evaluation Format)

POST /evaluation
  Receives: student_id, exam_id, question, expected_answer, student_answer, max_marks
  Uses local Qwen2.5-3B-Instruct to evaluate via prompting
  Returns structured JSON evaluation matching the exact user specification

Run:
    py -m uvicorn api:app --host 127.0.0.1 --port 8002
"""

import sys
import json
import re
import logging
from pathlib import Path

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from fastapi import FastAPI, HTTPException, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any
from transformers import AutoModelForCausalLM, AutoTokenizer

from configs.model_config import MODEL_PATH_OR_ID, MODEL_KWARGS, TOKENIZER_KWARGS

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── FastAPI App ──────────────────────────────────
app = FastAPI(title="Qwen Evaluation API", version="1.0.0")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Model Loading (singleton) ───────────────────
_model = None
_tokenizer = None


def get_model():
    """Load Qwen model once and reuse."""
    global _model, _tokenizer
    if _model is None:
        logger.info(f"Loading Qwen model from: {MODEL_PATH_OR_ID}")
        _tokenizer = AutoTokenizer.from_pretrained(MODEL_PATH_OR_ID, **TOKENIZER_KWARGS)
        if _tokenizer.pad_token is None:
            _tokenizer.pad_token = _tokenizer.eos_token
        _model = AutoModelForCausalLM.from_pretrained(MODEL_PATH_OR_ID, **MODEL_KWARGS)
        _model.eval()
        logger.info("Qwen model loaded successfully!")
    return _model, _tokenizer


# ── Load model at startup ────────────────────────
@app.on_event("startup")
async def startup_event():
    logger.info("Loading Qwen model on startup...")
    get_model()
    logger.info("Model ready!")


# ── Health Check ─────────────────────────────────
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Qwen Evaluation API",
        "model": MODEL_PATH_OR_ID,
        "device": "CUDA" if torch.cuda.is_available() else "CPU",
    }


# ── Response Models ──────────────────────────────
class QuestionEvaluation(BaseModel):
    question_id: str
    score: float
    feedback: str
    strengths: list[str]
    missing_concepts: list[str]


class EvaluationResponse(BaseModel):
    student_id: str
    exam_id: str
    total_score: float
    feedback: str
    evaluations: list[QuestionEvaluation]


# ── Evaluation Prompt Builder ────────────────────
def build_evaluation_prompt(
    student_id: str,
    exam_id: str,
    question: str,
    expected_answer: str,
    student_answer: str,
    max_marks: int = 5,
) -> str:
    """Build the evaluation prompt for Qwen."""
    return f"""You are a professional, expert exam evaluator. You will be given:
1. A question paper (possibly containing multiple questions)
2. The standard/model answer key (teacher's reference answers)
3. The student's answer sheet (possibly containing answers to multiple questions)

Your job is to:
- Identify each individual question from the question paper.
- For each question, evaluate the student's corresponding answer against the standard/model answer.
- Provide a per-question score, detailed feedback, list of strengths, and list of missing concepts.
- Provide a comprehensive overall feedback summary.

Input Information:
- Student ID: {student_id}
- Exam ID: {exam_id}

- Question Paper (OCR/Text):
{question}

- Expected Answer / Model Answer Key (OCR/Text):
{expected_answer}

- Student Answer (OCR/Text):
{student_answer}

Evaluation Rules:
1. Ignore minor typos, formatting issues, or spelling errors. Focus on the core concepts, correctness, and semantic meaning.
2. Be objective and precise. Deduct marks proportionally based on errors or missing information.
3. CRITICAL - DYNAMIC MARKS PER QUESTION: Each question in the question paper may explicitly state its allocated maximum marks in brackets, such as `[Marks: 2]`, `[Marks: 5]`, or `(2 marks)`. You MUST parse and use the exact maximum marks specified for each question to determine its score limit (e.g., if a question has `[Marks: 2]`, the awarded score must be between 0 and 2). If a question does not specify marks, use {max_marks} as the default maximum marks.
4. For each question's feedback, explain what the student did well, what was incorrect/missing, and why the specific score was awarded out of the question's total allocated marks.
5. List key points/concepts the student answered correctly under strengths.
6. List concepts/keywords that were omitted under missing concepts.

Output Format:
You must output the evaluation in the exact format shown below. Use the literal section headers (e.g. `[Overall Feedback]`, `[Question ID]`, `[Score]`, `[Strengths]`, `[Missing Concepts]`, `[Feedback]`) and separate questions with `---`. Do not include JSON, XML, or any markdown code blocks (such as ```).

Format Example:
[Overall Feedback]
<Overall feedback text summary of 2-3 paragraphs>

---
[Question ID] Q.1
[Score] 4
[Strengths]
- Explained the primary concept.
- Stated the correct formula.
[Missing Concepts]
- Failed to state the unit.
[Feedback]
Good answer with minor omissions in the units.

---
[Question ID] Q.2
[Score] 5
[Strengths]
- Correctly listed all factors.
[Missing Concepts]
[Feedback]
Excellent answer.
"""


# ── Generate with Qwen ───────────────────────────
def generate_evaluation(prompt: str, max_new_tokens: int = 1024) -> str:
    """Send prompt to Qwen and get the response."""
    model, tokenizer = get_model()

    system_content = (
        "You are an automated exam evaluation system.\n\n"
        "Rules:\n"
        "- Evaluate strictly according to the provided answer key.\n"
        "- Do not guess or infer information beyond the provided answers.\n"
        "- The same input must always produce the same output.\n"
        "- Return ONLY structured text matching the requested format.\n"
        "- Do not output JSON, XML, markdown code fences, or explanations outside the requested format."
    )
    messages = [
        {"role": "system", "content": system_content},
        {"role": "user", "content": prompt},
    ]

    text = tokenizer.apply_chat_template(
        messages,
        tokenize=False,
        add_generation_prompt=True,
    )

    inputs = tokenizer(
        text,
        return_tensors="pt",
        truncation=True,
        max_length=2048,
    ).to(model.device)

    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            do_sample=False,
            num_beams=1,
            pad_token_id=tokenizer.pad_token_id,
            eos_token_id=tokenizer.eos_token_id,
            repetition_penalty=1.1,
        )

    # Decode only new tokens
    generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
    response = tokenizer.decode(generated_ids, skip_special_tokens=True)
    return response.strip()


# ── Parse JSON from model output ─────────────────
def parse_json_response(raw: str) -> dict:
    """Extract and parse JSON from Qwen's output."""
    errors = []
    # Try direct parse
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        errors.append(f"Direct parse failed: {e}")

    # Try extracting from markdown code block
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if match:
        content = match.group(1).strip()
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            errors.append(f"Markdown parse failed: {e}")

    # Try finding JSON object in raw text
    start = raw.find("{")
    end = raw.rfind("}")
    if start != -1 and end != -1:
        content = raw[start:end + 1]
        try:
            return json.loads(content)
        except json.JSONDecodeError as e:
            errors.append(f"Extract parse failed: {e}")

    raise ValueError(f"Could not parse JSON. Errors: {'; '.join(errors)}. Raw output sample: {raw[:300]}")


# ── Parse Structured Text from model output ──────
def parse_structured_text(raw: str) -> dict:
    """
    Parses Qwen's structured text output.
    Format is expected to be:
    [Overall Feedback]
    <feedback text>

    ---
    [Question ID] Q.1
    [Score] 4
    [Strengths]
    - Defined concept.
    [Missing Concepts]
    - Missed unit.
    [Feedback]
    Good.
    """
    result = {
        "total_score": 0.0,
        "feedback": "",
        "evaluations": []
    }

    # 1. Extract Overall Feedback
    overall_match = re.search(r"\[Overall Feedback\]\s*(.*?)(?=\s*---|\s*\[Question ID\]|$)", raw, re.DOTALL | re.IGNORECASE)
    if overall_match:
        result["feedback"] = overall_match.group(1).strip()

    # 2. Split into blocks by question separator "---" or by "[Question ID]"
    blocks = []
    if "---" in raw:
        blocks = raw.split("---")
    else:
        # Fallback: split by "[Question ID]" and reconstruct blocks
        parts = re.split(r"(\[Question ID\])", raw, flags=re.IGNORECASE)
        current_block = ""
        for part in parts:
            if part.lower() == "[question id]":
                if current_block:
                    blocks.append(current_block)
                current_block = "[Question ID]"
            else:
                current_block += part
        if current_block:
            blocks.append(current_block)

    for block in blocks:
        if "[Question ID]" not in block:
            continue

        # Parse Question ID
        qid_match = re.search(r"\[Question ID\]\s*(.*?)\n", block, re.IGNORECASE)
        qid = qid_match.group(1).strip() if qid_match else "Unknown"

        # Parse Score
        score_match = re.search(r"\[Score\]\s*([\d\.]+)", block, re.IGNORECASE)
        try:
            score = float(score_match.group(1).strip()) if score_match else 0.0
        except ValueError:
            score = 0.0

        # Parse Strengths
        strengths = []
        strengths_match = re.search(r"\[Strengths\]\s*(.*?)(?=\s*\[|$)", block, re.DOTALL | re.IGNORECASE)
        if strengths_match:
            for line in strengths_match.group(1).strip().split("\n"):
                cleaned = line.strip().lstrip("-* ").strip()
                if cleaned:
                    strengths.append(cleaned)

        # Parse Missing Concepts
        missing_concepts = []
        missing_match = re.search(r"\[Missing Concepts\]\s*(.*?)(?=\s*\[|$)", block, re.DOTALL | re.IGNORECASE)
        if missing_match:
            for line in missing_match.group(1).strip().split("\n"):
                cleaned = line.strip().lstrip("-* ").strip()
                if cleaned:
                    missing_concepts.append(cleaned)

        # Parse Feedback
        feedback_match = re.search(r"\[Feedback\]\s*(.*?)(?=\s*\[|$)", block, re.DOTALL | re.IGNORECASE)
        feedback = feedback_match.group(1).strip() if feedback_match else ""

        result["evaluations"].append({
            "question_id": qid,
            "score": score,
            "feedback": feedback,
            "strengths": strengths,
            "missing_concepts": missing_concepts
        })

    # Sum up the total score
    result["total_score"] = sum(ev["score"] for ev in result["evaluations"])
    return result


# ── POST /evaluation ─────────────────────────────
@app.post("/evaluation", response_model=EvaluationResponse)
async def evaluate(
    student_id: str = Form("1"),
    exam_id: str = Form("1"),
    question: str = Form(...),
    expected_answer: str = Form(...),
    student_answer: str = Form(...),
    max_marks: int = Form(5),
):
    """
    Evaluate student's multi-question answer sheet against the model answers using Qwen.

    Required fields (form-data):
      - question:        The exam question paper text
      - expected_answer: The correct/teacher answer key text
      - student_answer:  The student's answer text (from OCR)

    Optional:
      - student_id:      Student ID string (default: "1")
      - exam_id:         Exam ID string (default: "1")
      - max_marks:       Default max marks per question (default: 5)
    """
    logger.info(f"[/evaluation] student_id={student_id}, exam_id={exam_id}, max_marks={max_marks}")

    # Build prompt
    prompt = build_evaluation_prompt(
        student_id=student_id,
        exam_id=exam_id,
        question=question,
        expected_answer=expected_answer,
        student_answer=student_answer,
        max_marks=max_marks,
    )

    # Generate with Qwen
    try:
        raw_output = generate_evaluation(prompt)
        logger.info(f"[/evaluation] Qwen raw output: {raw_output[:300]}")
    except Exception as e:
        logger.error(f"[/evaluation] Generation failed: {e}")
        raise HTTPException(500, f"Model generation failed: {str(e)}")

    # Parse structured text
    try:
        evaluation = parse_structured_text(raw_output)
        evaluation["student_id"] = student_id
        evaluation["exam_id"] = exam_id
        logger.info(f"[/evaluation] Evaluation complete — total_score: {evaluation.get('total_score', '?')}")
        return EvaluationResponse(**evaluation)
    except Exception as e:
        logger.error(f"[/evaluation] Parsing/validation failed: {e}")
        # Save raw output to a debug file to inspect the complete response
        debug_file = PROJECT_ROOT / "qwen_raw_error.json"
        try:
            debug_file.write_text(raw_output, encoding="utf-8")
            logger.info(f"Saved full raw output to {debug_file} for inspection.")
        except Exception as save_err:
            logger.error(f"Failed to save raw output: {save_err}")
        raise HTTPException(500, f"Could not parse structured text from model: {str(e)}. Raw output saved to debug file.")


# ── POST /evaluation/json (JSON body version) ────
class EvaluationRequest(BaseModel):
    student_id: str = "1"
    exam_id: str = "1"
    question: str
    expected_answer: str
    student_answer: str
    max_marks: int = 5


@app.post("/evaluation/json", response_model=EvaluationResponse)
async def evaluate_json(req: EvaluationRequest):
    """
    Same as /evaluation but accepts JSON body instead of form-data.
    """
    logger.info(f"[/evaluation/json] student_id={req.student_id}, exam_id={req.exam_id}, max_marks={req.max_marks}")

    prompt = build_evaluation_prompt(
        student_id=req.student_id,
        exam_id=req.exam_id,
        question=req.question,
        expected_answer=req.expected_answer,
        student_answer=req.student_answer,
        max_marks=req.max_marks,
    )

    try:
        raw_output = generate_evaluation(prompt)
        logger.info(f"[/evaluation/json] Qwen raw output: {raw_output[:300]}")
    except Exception as e:
        logger.error(f"[/evaluation/json] Generation failed: {e}")
        raise HTTPException(500, f"Model generation failed: {str(e)}")

    try:
        evaluation = parse_structured_text(raw_output)
        evaluation["student_id"] = req.student_id
        evaluation["exam_id"] = req.exam_id
        logger.info(f"[/evaluation/json] Evaluation complete — total_score: {evaluation.get('total_score', '?')}")
        return EvaluationResponse(**evaluation)
    except Exception as e:
        logger.error(f"[/evaluation/json] Parsing/validation failed: {e}")
        raise HTTPException(500, f"Could not parse structured text from model: {str(e)}. Raw output: {raw_output}")
