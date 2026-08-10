"""
vl_ocr_eval_pipeline.py — Production Qwen2.5-VL-3B-Instruct Pipeline API

Dedicated Vision-Language Evaluation Server powered by Qwen2.5-VL-3B-Instruct.
Can evaluate both pre-extracted text AND raw document images natively.

Run:
    python -m uvicorn vl_ocr_eval_pipeline:app --host 127.0.0.1 --port 8002 --reload
"""

import sys
import os
import re
import json
import time
import logging
import tempfile
from pathlib import Path
from typing import Optional, List, Dict, Any

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

from configs.vl_model_config import VL_MODEL_ID, VL_MODEL_PATH_OR_ID
from model.vl_loader import load_vl_model_and_processor, generate_vl_evaluation

logger = logging.getLogger("vl_ocr_eval_pipeline")
logging.basicConfig(level=logging.INFO)

# ── FastAPI App Setup ──────────────────────────────
app = FastAPI(
    title="Qwen2.5-VL Evaluation Pipeline API",
    description="Production-grade Vision-Language Evaluation System",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════
#  Data Schemas (Dual Schema Support)
# ═══════════════════════════════════════════════════

class TextEvaluationRequest(BaseModel):
    student_id: str = "1"
    exam_id: str = "1"
    question_paper_text: str = ""
    answer_key_text: str = ""
    student_answer_text: str = ""
    max_marks: int = 5


class QuestionEvaluationItem(BaseModel):
    questionId: str = Field(...)
    questionText: str = Field("")
    expectedAnswer: str = Field("")
    studentAnswer: str = Field("")
    maxMarks: float = Field(5.0)
    score: float = Field(0.0)
    status: str = Field("Incorrect", description="Correct | Partial | Incorrect | Unanswered")
    feedback: str = Field("")
    strengths: List[str] = Field(default_factory=list)
    missingConcepts: List[str] = Field(default_factory=list)
    keywordsMatched: List[str] = Field(default_factory=list)
    keywordsMissing: List[str] = Field(default_factory=list)


class EvaluationResponse(BaseModel):
    status: str = "success"
    studentId: str
    examId: str
    totalScore: float
    maxMarksTotal: float
    percentage: float
    grade: str
    summaryFeedback: str
    evaluations: List[QuestionEvaluationItem]
    modelUsed: str = VL_MODEL_ID


# ═══════════════════════════════════════════════════
#  Prompt Builders
# ═══════════════════════════════════════════════════

def build_vl_prompt(question_paper: str, answer_key: str, student_answer: str, max_marks: int) -> str:
    return f"""You are an expert exam evaluator using Qwen2.5-VL Vision-Language model.
Evaluate the student's answer sheet against the model answer key and question paper.

Input Details:
- Default Max Marks Per Question: {max_marks}

Question Paper:
{question_paper}

Model Answer Key:
{answer_key}

Student Answer Sheet:
{student_answer}

Instructions:
1. Identify all questions in the paper (Q1, Q2, etc.).
2. Evaluate student answer for each question against model answer.
3. Output strictly valid JSON matching this schema:
{{
  "evaluations": [
    {{
      "questionId": "Q1",
      "questionText": "Text of question",
      "expectedAnswer": "Model answer",
      "studentAnswer": "Student answer",
      "maxMarks": 5,
      "score": 4.5,
      "status": "Correct",
      "feedback": "Detailed feedback",
      "strengths": ["Clear explanation"],
      "missingConcepts": ["Minor point omitted"],
      "keywordsMatched": ["keyword1"],
      "keywordsMissing": ["keyword2"]
    }}
  ],
  "summaryFeedback": "Overall student performance summary."
}}
Output ONLY the JSON object, surrounded by ```json ... ```. No extra commentary.
"""


def parse_vl_output(raw_output: str, student_id: str, exam_id: str, default_max_marks: int) -> EvaluationResponse:
    """
    Parse raw output from Qwen2.5-VL into structured EvaluationResponse.
    """
    json_str = raw_output
    match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw_output, re.DOTALL)
    if match:
        json_str = match.group(1)
    else:
        brace = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if brace:
            json_str = brace.group(0)

    try:
        data = json.loads(json_str)
    except Exception as parse_err:
        logger.warning(f"JSON parse error on raw output ({parse_err}). Generating fallback structured output.")
        data = {
            "evaluations": [{
                "questionId": "Q1",
                "questionText": "Paper Evaluation",
                "expectedAnswer": "Model Answer",
                "studentAnswer": "Student Answer",
                "maxMarks": float(default_max_marks),
                "score": 0.0,
                "status": "Incorrect",
                "feedback": raw_output[:300],
                "strengths": [],
                "missingConcepts": ["Could not parse structured output"],
                "keywordsMatched": [],
                "keywordsMissing": [],
            }],
            "summaryFeedback": "Evaluation completed with fallback formatting."
        }

    eval_items = []
    tot_score = 0.0
    tot_max = 0.0

    for item in data.get("evaluations", []):
        q_item = QuestionEvaluationItem(
            questionId=str(item.get("questionId", "Q1")),
            questionText=str(item.get("questionText", "")),
            expectedAnswer=str(item.get("expectedAnswer", "")),
            studentAnswer=str(item.get("studentAnswer", "")),
            maxMarks=float(item.get("maxMarks", default_max_marks)),
            score=float(item.get("score", 0.0)),
            status=str(item.get("status", "Incorrect")),
            feedback=str(item.get("feedback", "")),
            strengths=item.get("strengths", []),
            missingConcepts=item.get("missingConcepts", []),
            keywordsMatched=item.get("keywordsMatched", []),
            keywordsMissing=item.get("keywordsMissing", []),
        )
        eval_items.append(q_item)
        tot_score += q_item.score
        tot_max += q_item.maxMarks

    pct = round((tot_score / tot_max * 100), 2) if tot_max > 0 else 0.0
    grade = "A+" if pct >= 90 else ("A" if pct >= 80 else ("B" if pct >= 70 else ("C" if pct >= 60 else "F")))

    return EvaluationResponse(
        status="success",
        studentId=student_id,
        examId=exam_id,
        totalScore=round(tot_score, 2),
        maxMarksTotal=round(tot_max, 2),
        percentage=pct,
        grade=grade,
        summaryFeedback=data.get("summaryFeedback", f"Total score: {tot_score}/{tot_max}"),
        evaluations=eval_items,
        modelUsed=VL_MODEL_ID,
    )


# ═══════════════════════════════════════════════════
#  Startup & API Endpoints
# ═══════════════════════════════════════════════════

@app.on_event("startup")
async def startup():
    logger.info("Initializing Qwen2.5-VL model on startup...")
    try:
        load_vl_model_and_processor()
    except Exception as e:
        logger.warning(f"Qwen2.5-VL model auto-load failed (will lazy-load on demand): {e}")


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Qwen2.5-VL-3B-Instruct Evaluation Server",
        "model": VL_MODEL_ID,
        "model_path": VL_MODEL_PATH_OR_ID,
        "device": "CUDA" if torch.cuda.is_available() else "CPU",
    }


@app.post("/evaluate-text", response_model=EvaluationResponse)
async def evaluate_text(req: TextEvaluationRequest):
    """
    Evaluates pre-extracted answer sheet text using Qwen2.5-VL-3B-Instruct.
    """
    logger.info(f"[/evaluate-text] Running Qwen2.5-VL evaluation for student {req.student_id}")
    prompt = build_vl_prompt(
        question_paper=req.question_paper_text,
        answer_key=req.answer_key_text,
        student_answer=req.student_answer_text,
        max_marks=req.max_marks,
    )

    raw_output = generate_vl_evaluation(prompt=prompt)
    return parse_vl_output(raw_output, req.student_id, req.exam_id, req.max_marks)


@app.post("/evaluation", response_model=EvaluationResponse)
async def evaluate_alias(req: TextEvaluationRequest):
    """
    Alias for /evaluate-text.
    """
    return await evaluate_text(req)


@app.post("/evaluate-paper", response_model=EvaluationResponse)
async def evaluate_paper(
    student_file: UploadFile = File(...),
    student_id: str = Form("1"),
    exam_id: str = Form("1"),
    question_paper_text: str = Form(""),
    answer_key_text: str = Form(""),
    max_marks: int = Form(5),
):
    """
    Evaluates student answer sheet directly from uploaded image using Qwen2.5-VL vision capabilities.
    """
    logger.info(f"[/evaluate-paper] Running Qwen2.5-VL vision evaluation on image file: {student_file.filename}")
    
    # Save image temporarily
    suffix = Path(student_file.filename).suffix or ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as tmp:
        tmp.write(await student_file.read())
        tmp_path = tmp.name

    try:
        prompt = build_vl_prompt(question_paper_text, answer_key_text, "[IMAGE ATTACHED]", max_marks)
        raw_output = generate_vl_evaluation(prompt=prompt, image_paths=[tmp_path])
        return parse_vl_output(raw_output, student_id, exam_id, max_marks)
    finally:
        if os.path.exists(tmp_path):
            os.remove(tmp_path)
