"""
main.py — Evalgen API

POST /evaluation
  - question        (required, form-data)
  - expected_answer (required, form-data)
  - file            (optional, file upload — image or PDF of handwritten answer)
  - text            (optional, form-data  — plain text student answer)

  Exactly ONE of file or text must be provided.

Flow:
  Image/PDF → Mistral OCR → extracted text → Gemini AI → evaluation JSON
"""

import os
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional, Any

# Import evaluation pipeline
try:
    from workernode import evaluate_answer_agent, evaluate_ocr_docs_pipeline, evaluate_ocr_text_ref_pipeline
    WORKERNODE_AVAILABLE = True
except ImportError as e:
    print(f"⚠️  Workernode import failed ({e}). Running in fallback mode.")
    WORKERNODE_AVAILABLE = False

    def evaluate_answer_agent(input_data: dict, **kwargs):
        """Fallback when dependencies are missing."""
        return {
            "evaluations": [{
                "ocr_text": "",
                "similarity_percentage": 0,
                "marks_awarded": 0,
                "max_marks": 10,
                "missing_points": ["Fallback mode — dependencies not installed"],
                "feedback": "Service running in fallback mode."
            }]
        }

    def evaluate_ocr_docs_pipeline(*args, **kwargs):
        """Fallback when dependencies are missing."""
        return {
            "error": "Workernode not available — fallback mode."
        }



app = FastAPI(title="Evalgen Evaluation API")

# CORS — allow all origins for development
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Health Check ──
@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "Evalgen Evaluation API",
        "workernode_available": WORKERNODE_AVAILABLE,
    }


# ── Evaluation Endpoint ──
class EvaluationResponse(BaseModel):
    evaluations: Any


@app.post("/evaluation", response_model=EvaluationResponse)
async def evaluate(
    question: str               = Form(...),
    expected_answer: str        = Form(...),
    file: Optional[UploadFile]  = File(None),
    text: Optional[str]         = Form(None),
):
    """
    Evaluate a student's answer against an expected answer.

    Required fields (form-data):
      - question:        The exam question
      - expected_answer: The correct / model answer

    Provide exactly ONE of:
      - file: Handwritten answer image (JPG/PNG) or PDF
      - text: Plain-text student answer

    Response:
    {
      "evaluations": [{
        "ocr_text":               "...",
        "similarity_percentage":  85,
        "marks_awarded":          8,
        "max_marks":              10,
        "missing_points":         ["..."],
        "feedback":               "..."
      }]
    }
    """

    # ── Validate: exactly one source ──
    has_file = file is not None and getattr(file, "filename", "") != ""
    has_text = text is not None and text.strip() != ""

    print(f"[/evaluation] question='{question[:80]}', has_file={has_file}, has_text={has_text}")

    if has_file and has_text:
        raise HTTPException(400, "Provide exactly one of: file or text — not both.")
    if not has_file and not has_text:
        raise HTTPException(400, "Provide exactly one of: file or text.")

    # ── Build input for workernode ──
    input_data = {
        "question": question,
        "expected_answer": expected_answer,
        "max_marks": 10,
    }

    if has_file:
        input_data["file"] = await file.read()
        input_data["filename"] = file.filename.lower()
        print(f"[/evaluation] File received: {file.filename} ({len(input_data['file'])} bytes)")
    else:
        input_data["text"] = text.strip()
        print(f"[/evaluation] Text received: {text.strip()[:100]}...")

    # ── Run evaluation pipeline ──
    result = evaluate_answer_agent(input_data)

    if isinstance(result, dict) and "error" in result:
        print(f"[/evaluation] ❌ Error: {result['error']}")
        raise HTTPException(500, result["error"])

    print(f"[/evaluation] ✅ Evaluation complete")
    return EvaluationResponse(evaluations=result.get("evaluations", []))


# ── OCR-based Multi-Question Evaluation Endpoint ──
class QuestionEvaluation(BaseModel):
    question_id: str
    score: float
    feedback: str
    strengths: list[str]
    missing_concepts: list[str]


class OCREvaluationResponse(BaseModel):
    student_id: str
    exam_id: str
    total_score: float
    feedback: str
    evaluations: list[QuestionEvaluation]
    student_ans_ocr: str
    standard_ans_ocr: str
    question_ocr: str


@app.post("/ocr-evaluation", response_model=OCREvaluationResponse)
async def ocr_evaluation(
    student_ans: UploadFile = File(...),
    standard_ans: UploadFile = File(...),
    question: UploadFile = File(...),
    max_marks: int = Form(10),
    student_id: str = Form(...),
    exam_id: str = Form(...),
):
    """
    Perform Mistral OCR on:
      - student_ans (PDF/Image)
      - standard_ans (PDF/Image)
      - question (PDF/Image)
    Save each extracted text to a local file in extracted_files/,
    then evaluate each question in the student's answer against the
    standard/model answer using Gemini.

    Returns per-question evaluations with scores, feedback, strengths,
    and missing_concepts, plus overall total_score and feedback.
    """
    print(f"[/ocr-evaluation] Received files: student_ans='{student_ans.filename}', standard_ans='{standard_ans.filename}', question='{question.filename}'")
    print(f"[/ocr-evaluation] student_id='{student_id}', exam_id='{exam_id}', max_marks={max_marks}")

    student_bytes = await student_ans.read()
    standard_bytes = await standard_ans.read()
    question_bytes = await question.read()

    result = evaluate_ocr_docs_pipeline(
        student_bytes=student_bytes,
        student_filename=student_ans.filename,
        standard_bytes=standard_bytes,
        standard_filename=standard_ans.filename,
        question_bytes=question_bytes,
        question_filename=question.filename,
        max_marks=max_marks,
    )

    if isinstance(result, dict) and "error" in result:
        print(f"[/ocr-evaluation] ❌ Error: {result['error']}")
        raise HTTPException(500, result["error"])

    print(f"[/ocr-evaluation] ✅ OCR & Evaluation complete")
    return OCREvaluationResponse(
        student_id=student_id,
        exam_id=exam_id,
        total_score=result.get("total_score", 0),
        feedback=result.get("feedback", ""),
        evaluations=result.get("evaluations", []),
        student_ans_ocr=result.get("student_ans_ocr", ""),
        standard_ans_ocr=result.get("standard_ans_ocr", ""),
        question_ocr=result.get("question_ocr", ""),
    )


# ── OCR-based Multi-Question Evaluation with Text Reference Endpoint ──
class OCREvaluationTextRefResponse(BaseModel):
    student_id: str
    exam_id: str
    total_score: float
    feedback: str
    evaluations: list[QuestionEvaluation]
    student_ans_ocr: str


@app.post("/ocr-evaluation-text-ref", response_model=OCREvaluationTextRefResponse)
async def ocr_evaluation_text_ref(
    student_ans: UploadFile = File(...),
    question_text: str = Form(...),
    standard_ans_text: str = Form(...),
    max_marks: int = Form(10),
    student_id: str = Form(...),
    exam_id: str = Form(...),
):
    """
    Perform Mistral OCR on the student's answer sheet (student_ans), then evaluate
    each question against the provided question_text and standard_ans_text strings
    using Gemini.
    """
    print(f"[/ocr-evaluation-text-ref] Received file: student_ans='{student_ans.filename}'")
    print(f"[/ocr-evaluation-text-ref] student_id='{student_id}', exam_id='{exam_id}', max_marks={max_marks}")

    student_bytes = await student_ans.read()

    result = evaluate_ocr_text_ref_pipeline(
        student_bytes=student_bytes,
        student_filename=student_ans.filename,
        question_text=question_text,
        standard_text=standard_ans_text,
        max_marks=max_marks,
    )

    if isinstance(result, dict) and "error" in result:
        print(f"[/ocr-evaluation-text-ref] ❌ Error: {result['error']}")
        raise HTTPException(500, result["error"])

    print(f"[/ocr-evaluation-text-ref] ✅ OCR & Text Reference Evaluation complete")
    return OCREvaluationTextRefResponse(
        student_id=student_id,
        exam_id=exam_id,
        total_score=result.get("total_score", 0),
        feedback=result.get("feedback", ""),
        evaluations=result.get("evaluations", []),
        student_ans_ocr=result.get("student_ans_ocr", ""),
    )




# ──────────────────────────────────────────────────────────────────────
# OLD /evaluation endpoint (commented out — kept for reference)
# ──────────────────────────────────────────────────────────────────────
# @app.post("/evaluation")
# async def solve_questions(
#     file: Optional[UploadFile] = File(None),
#     text: Optional[str]        = Form(None),
#     grade: str                 = Form(...),
#     curriculum: str            = Form(...),
#     mcq_options: Optional[str] = Form("{}")
# ):
#     ...old pipeline with langchain + vertex AI + YouTube/WebSearch...
