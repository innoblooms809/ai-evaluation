"""
workernode.py — Evaluation Pipeline

Flow:
  1. If file (image/PDF) is provided → Mistral OCR → extract student answer text
  2. Send question + expected_answer + student_answer to Gemini → get evaluation
  3. Return structured JSON result
"""

import os
import re
import json
import base64
import tempfile
import logging
import time
from typing import Any, Dict

from dotenv import load_dotenv
from tools import OCRTool
from google import genai

# ── Load environment ──
load_dotenv()

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ── Mistral OCR client ──
ocr_tool = OCRTool()   # uses MISTRAL_API_KEY from .env

# ── Gemini client ──
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
if not GEMINI_API_KEY:
    raise RuntimeError("GEMINI_API_KEY is not set in .env")

gemini_client = genai.Client(api_key=GEMINI_API_KEY)
GEMINI_MODEL = "gemini-2.5-flash"


# ──────────────────────────────────────────────────────────────────────
# STEP 1: Extract text from file using Mistral OCR
# ──────────────────────────────────────────────────────────────────────
def extract_text_from_file(file_bytes: bytes, filename: str) -> str:
    """
    Use Mistral OCR to extract text from an uploaded image or PDF.

    Supports: .pdf, .png, .jpg, .jpeg
    Returns: extracted text as a string
    """
    ext = os.path.splitext(filename)[1].lower()
    extracted_text = ""

    if ext == ".pdf" or file_bytes[:4] == b"%PDF":
        # ── PDF: upload to Mistral, get signed URL, run OCR ──
        logger.info("📄 Running Mistral OCR on PDF...")
        with tempfile.NamedTemporaryFile(suffix=".pdf", delete=False) as tmp:
            tmp.write(file_bytes)
            pdf_path = tmp.name

        md_path = pdf_path + ".md"
        try:
            file_id = ocr_tool.upload_pdf(pdf_path)
            ocr_tool.ocr_from_uploaded_pdf(file_id, md_path)
            with open(md_path, encoding="utf-8") as f:
                extracted_text = f.read()
        finally:
            if os.path.exists(pdf_path):
                os.remove(pdf_path)
            if os.path.exists(md_path):
                os.remove(md_path)

    elif ext in {".png", ".jpg", ".jpeg"}:
        # ── Image: convert to base64 data URL, run OCR ──
        logger.info("🖼️  Running Mistral OCR on image...")
        b64 = base64.b64encode(file_bytes).decode("ascii")
        mime = "image/png" if ext == ".png" else "image/jpeg"
        data_url = f"data:{mime};base64,{b64}"

        md_path = tempfile.mktemp(suffix=".md")
        try:
            ocr_tool.ocr_from_image_url(data_url, md_path, include_images=False)
            with open(md_path, encoding="utf-8") as f:
                extracted_text = f.read()
        finally:
            if os.path.exists(md_path):
                os.remove(md_path)

    else:
        # ── Unknown format: try to decode as UTF-8 text ──
        logger.info("📝 Unknown file type, reading as plain text...")
        extracted_text = file_bytes.decode("utf-8", errors="replace")

    return extracted_text.strip()


# ──────────────────────────────────────────────────────────────────────
# STEP 2: Evaluate using Gemini
# ──────────────────────────────────────────────────────────────────────
def evaluate_with_gemini(
    question: str,
    expected_answer: str,
    student_answer: str,
    max_marks: int = 10,
) -> dict:
    """
    Send question + expected answer + student OCR answer to Gemini.
    Gemini evaluates semantically and returns structured JSON.
    """
    prompt = f"""You are a professional, expert exam evaluator. Your job is to semantically analyze the student's answer, compare it to the expected/model answer for the given question, and provide a fair score along with clear, actionable, and detailed feedback explaining why the score was given.

Input Information:
- Question:
{question}

- Expected Answer (Model Answer Key):
{expected_answer}

- Student OCR Answer:
{student_answer}

Evaluation Rules:
1. Ignore OCR-induced typos, formatting errors, or minor spelling mistakes. Focus entirely on semantic meaning, scientific/mathematical correctness, core concepts, and formulas.
2. Be objective, fair, and precise. Deduct marks proportionally for missing details or errors.
3. In the feedback, explicitly address:
   - What key points the student answered correctly.
   - What parts were incorrect, incomplete, or completely missing.
   - A clear explanation of why the specific marks were awarded or deducted based on the question and expected answer.

Return ONLY a valid JSON object matching this structure (no markdown fences, no trailing commas):
{{
  "similarity_percentage": <integer from 0 to 100 representing how closely the semantic concepts of the student's answer align with the expected answer>,
  "marks_awarded": <number from 0 to {max_marks} based on correctness>,
  "max_marks": {max_marks},
  "missing_points": ["list of specific concepts, keywords, or steps from the expected answer that the student omitted"],
  "feedback": "A comprehensive evaluation explanation. Explain what was correct, what was incorrect, and why the marks were awarded/deducted. Keep it professional and constructive."
}}"""

    logger.info(f"🤖 Sending to Gemini ({GEMINI_MODEL}) for evaluation...")
    
    def call_gemini_with_retry(prompt_text, max_retries=3, base_delay=45):
        for attempt in range(max_retries):
            try:
                return gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt_text,
                )
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < max_retries - 1:
                        logger.warning(f"⚠️ Rate limit hit (429). Retrying in {base_delay}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(base_delay)
                        base_delay *= 1.5
                    else:
                        logger.error("❌ Max retries reached for Gemini API.")
                        raise
                else:
                    raise

    response = call_gemini_with_retry(prompt)

    raw = response.text.strip()
    logger.info(f"Gemini raw response: {raw[:500]}")

    # Extract JSON block or find curly braces
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if match:
        raw = match.group(1).strip()
    else:
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1:
            raw = raw[start:end+1]
    
    raw = raw.strip()

    return json.loads(raw)



# ──────────────────────────────────────────────────────────────────────
# STEP 2b: Multi-Question Evaluation using Gemini
# ──────────────────────────────────────────────────────────────────────
def evaluate_multi_question_with_gemini(
    question_text: str,
    standard_answer_text: str,
    student_answer_text: str,
    max_marks: int = 5,
) -> dict:
    """
    Send the full OCR'd question paper, standard answer key, and student answer
    to Gemini. Gemini identifies each question, evaluates the student's answer
    per-question, and returns a structured JSON with per-question evaluations,
    overall total_score, and overall feedback.
    """
    prompt = f"""You are a professional, expert exam evaluator. You will be given:
1. A question paper (possibly containing multiple questions)
2. The standard/model answer key
3. The student's answer sheet

Your job is to:
- Identify each individual question from the question paper.
- For each question, evaluate the student's corresponding answer against the standard answer.
- Provide a per-question score, detailed feedback, list of strengths, and list of missing concepts.
- Provide an overall total score and a comprehensive overall feedback summary.

Input Information:

- Question Paper (OCR Text):
{question_text}

- Standard/Model Answer Key (OCR Text):
{standard_answer_text}

- Student's Answer Sheet (OCR Text):
{student_answer_text}

Evaluation Rules:
1. Ignore OCR-induced typos, formatting errors, or minor spelling mistakes. Focus entirely on semantic meaning, scientific/mathematical correctness, core concepts, and formulas.
2. Be objective, fair, and precise. Deduct marks proportionally for missing details or errors.
3. Each question should be scored out of the maximum marks indicated for that specific question in the Question Paper text (e.g., [Marks: 2]). If no marks are explicitly indicated for a question, assume a default maximum of {max_marks} marks.
4. For each question's feedback, explicitly address what key points the student answered correctly and what parts were incorrect, incomplete, or missing.
5. The "strengths" list should contain specific things the student got right.
6. The "missing_concepts" list should contain specific concepts, keywords, or steps from the expected answer that the student omitted. If the answer is perfect, return an empty list.
7. The overall "feedback" should be a comprehensive summary (2-3 paragraphs) of the student's overall performance, highlighting patterns of strength and areas for improvement.
8. The "total_score" should be the sum of all individual question scores.

Return ONLY a valid JSON object matching this exact structure (no markdown fences, no trailing commas):
{{
  "total_score": <number — sum of all individual question scores>,
  "feedback": "<comprehensive overall feedback summary — 2-3 paragraphs>",
  "evaluations": [
    {{
      "question_id": "<question identifier, e.g. Q.1, Q.2, etc.>",
      "score": <number representing the score awarded based on the question's maximum marks>,
      "feedback": "<detailed per-question evaluation feedback>",
      "strengths": ["list of specific things the student got right"],
      "missing_concepts": ["list of specific concepts/keywords the student missed"]
    }}
  ]
}}"""

    logger.info(f"🤖 Sending multi-question evaluation to Gemini ({GEMINI_MODEL})...")
    
    def call_gemini_with_retry(prompt_text, max_retries=3, base_delay=45):
        for attempt in range(max_retries):
            try:
                return gemini_client.models.generate_content(
                    model=GEMINI_MODEL,
                    contents=prompt_text,
                )
            except Exception as e:
                error_str = str(e)
                if "429" in error_str or "RESOURCE_EXHAUSTED" in error_str:
                    if attempt < max_retries - 1:
                        logger.warning(f"⚠️ Rate limit hit (429). Retrying in {base_delay}s... (Attempt {attempt+1}/{max_retries})")
                        time.sleep(base_delay)
                        base_delay *= 1.5
                    else:
                        logger.error("❌ Max retries reached for Gemini API.")
                        raise
                else:
                    raise

    response = call_gemini_with_retry(prompt)

    raw = response.text.strip()
    logger.info(f"Gemini raw response (multi-Q): {raw[:500]}")

    # Extract JSON block or find curly braces
    match = re.search(r"```(?:json)?\s*(.*?)\s*```", raw, flags=re.DOTALL | re.IGNORECASE)
    if match:
        raw = match.group(1).strip()
    else:
        start = raw.find('{')
        end = raw.rfind('}')
        if start != -1 and end != -1:
            raw = raw[start:end+1]
            
    raw = raw.strip()

    return json.loads(raw)


# ──────────────────────────────────────────────────────────────────────
# MAIN: Full pipeline — OCR + Evaluate
# ──────────────────────────────────────────────────────────────────────
def evaluate_answer_agent(input_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Full evaluation pipeline:
      1. If file → Mistral OCR → extract student answer text
      2. Send to Gemini for evaluation
      3. Return structured result

    input_data keys:
      - question        (str, required)
      - expected_answer (str, required)
      - file            (bytes, optional) — image or PDF of handwritten answer
      - filename        (str, optional)   — original filename
      - text            (str, optional)   — plain text student answer
      - max_marks       (int, optional)   — defaults to 10
    """
    question = input_data.get("question", "")
    expected_answer = input_data.get("expected_answer", "")
    max_marks = input_data.get("max_marks", 10)

    file_bytes = input_data.get("file")
    filename = input_data.get("filename", "").lower()
    student_answer = input_data.get("text", "")

    # ── STEP 1: OCR if file is provided ──
    if file_bytes:
        try:
            student_answer = extract_text_from_file(file_bytes, filename)
            logger.info(f"✅ OCR extracted text ({len(student_answer)} chars): {student_answer[:200]}...")
        except Exception as e:
            logger.error(f"❌ OCR failed: {e}")
            return {"error": f"OCR failed: {str(e)}"}

    if not student_answer:
        return {"error": "No student answer text provided or extracted."}

    # ── STEP 2: Evaluate with Gemini ──
    try:
        evaluation = evaluate_with_gemini(
            question=question,
            expected_answer=expected_answer,
            student_answer=student_answer,
            max_marks=max_marks,
        )

        # Add the OCR text to the result so the caller can see what was extracted
        evaluation["ocr_text"] = student_answer if file_bytes else ""

        return {"evaluations": [evaluation]}

    except json.JSONDecodeError as e:
        logger.error(f"❌ Gemini returned invalid JSON: {e}")
        return {"error": f"Gemini returned invalid JSON: {str(e)}"}
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        return {"error": f"Evaluation failed: {str(e)}"}


def evaluate_ocr_docs_pipeline(
    student_bytes: bytes,
    student_filename: str,
    standard_bytes: bytes,
    standard_filename: str,
    question_bytes: bytes,
    question_filename: str,
    max_marks: int = 10,
) -> Dict[str, Any]:
    """
    OCR student_ans, standard_ans, and question files using Mistral OCR.
    Save each extracted text to a local file inside extracted_files/ directory.
    Evaluate the student's answer against the standard answer using Gemini.
    """
    logger.info("🎬 Starting evaluate_ocr_docs_pipeline...")
    os.makedirs("extracted_files", exist_ok=True)

    # 1. OCR Student Answer
    try:
        student_text = extract_text_from_file(student_bytes, student_filename)
        with open("extracted_files/student_ans.txt", "w", encoding="utf-8") as f:
            f.write(student_text)
        logger.info("💾 Saved extracted student_ans text to extracted_files/student_ans.txt")
    except Exception as e:
        logger.error(f"❌ Failed to OCR student_ans: {e}")
        return {"error": f"Failed to OCR student_ans: {str(e)}"}

    # 2. OCR Standard Answer
    try:
        standard_text = extract_text_from_file(standard_bytes, standard_filename)
        with open("extracted_files/standard_ans.txt", "w", encoding="utf-8") as f:
            f.write(standard_text)
        logger.info("💾 Saved extracted standard_ans text to extracted_files/standard_ans.txt")
    except Exception as e:
        logger.error(f"❌ Failed to OCR standard_ans: {e}")
        return {"error": f"Failed to OCR standard_ans: {str(e)}"}

    # 3. OCR Question
    try:
        question_text = extract_text_from_file(question_bytes, question_filename)
        with open("extracted_files/question.txt", "w", encoding="utf-8") as f:
            f.write(question_text)
        logger.info("💾 Saved extracted question text to extracted_files/question.txt")
    except Exception as e:
        logger.error(f"❌ Failed to OCR question: {e}")
        return {"error": f"Failed to OCR question: {str(e)}"}

    # 4. Multi-question evaluation with Gemini
    try:
        result = evaluate_multi_question_with_gemini(
            question_text=question_text,
            standard_answer_text=standard_text,
            student_answer_text=student_text,
            max_marks=max_marks,
        )

        return {
            "student_ans_ocr": student_text,
            "standard_ans_ocr": standard_text,
            "question_ocr": question_text,
            "total_score": result.get("total_score", 0),
            "feedback": result.get("feedback", ""),
            "evaluations": result.get("evaluations", []),
        }
    except json.JSONDecodeError as e:
        logger.error(f"❌ Gemini returned invalid JSON: {e}")
        return {"error": f"Gemini returned invalid JSON: {str(e)}"}
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        return {"error": f"Evaluation failed: {str(e)}"}


def evaluate_ocr_text_ref_pipeline(
    student_bytes: bytes,
    student_filename: str,
    question_text: str,
    standard_text: str,
    max_marks: int = 10,
) -> Dict[str, Any]:
    """
    OCR the student's answer sheet, then perform multi-question evaluation
    against standard answers and questions passed as plain text strings.
    """
    logger.info("🎬 Starting evaluate_ocr_text_ref_pipeline...")

    # 1. OCR Student Answer
    try:
        student_text = extract_text_from_file(student_bytes, student_filename)
        logger.info(f"✅ Student answer OCR extracted ({len(student_text)} chars)")
    except Exception as e:
        logger.error(f"❌ Failed to OCR student_ans: {e}")
        return {"error": f"Failed to OCR student answers: {str(e)}"}

    # 2. Multi-question evaluation with Gemini
    try:
        result = evaluate_multi_question_with_gemini(
            question_text=question_text,
            standard_answer_text=standard_text,
            student_answer_text=student_text,
            max_marks=max_marks,
        )

        return {
            "student_ans_ocr": student_text,
            "total_score": result.get("total_score", 0),
            "feedback": result.get("feedback", ""),
            "evaluations": result.get("evaluations", []),
        }
    except json.JSONDecodeError as e:
        logger.error(f"❌ Gemini returned invalid JSON: {e}")
        return {"error": f"Gemini returned invalid JSON: {str(e)}"}
    except Exception as e:
        logger.error(f"❌ Evaluation failed: {e}")
        return {"error": f"Evaluation failed: {str(e)}"}


