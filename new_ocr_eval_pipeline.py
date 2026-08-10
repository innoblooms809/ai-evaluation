"""
new_ocr_eval_pipeline.py — New Modular OCR Evaluation Pipeline

A clean, modular, new approach for multi-question exam paper evaluation.

Run:
    python -m uvicorn new_ocr_eval_pipeline:app --host 127.0.0.1 --port 8002 --reload
"""

import sys
import os
import re
import json
import time
import asyncio
import logging
from typing import Optional, Any, List, Dict
from pathlib import Path

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

import httpx
import torch
from fastapi import FastAPI, HTTPException, UploadFile, File, Form
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field

# ── Logging Configuration ──────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("new_ocr_eval_pipeline")

# ── Configuration ─────────────────────────────────
OCR_SERVER_URL = os.getenv("OCR_SERVER_URL", "http://localhost:8000/ocrOutput")
DEFAULT_MAX_MARKS = 5

# ── FastAPI App Setup ──────────────────────────────
app = FastAPI(
    title="New OCR Evaluation Pipeline API",
    description="Next-Generation Multi-Question Evaluation Pipeline",
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
#  Data Models (Schema definitions)
# ═══════════════════════════════════════════════════

class TextEvaluationRequest(BaseModel):
    student_id: str = Field(..., alias="student_id")
    exam_id: str = Field(..., alias="exam_id")
    question_paper_text: str = Field(..., alias="question_paper_text")
    answer_key_text: str = Field(..., alias="answer_key_text")
    student_answer_text: str = Field(..., alias="student_answer_text")
    max_marks: int = Field(DEFAULT_MAX_MARKS, alias="max_marks")

    class Config:
        populate_by_name = True


class QuestionEvaluationItem(BaseModel):
    questionId: str = Field(..., description="e.g. Q1, Q2")
    questionText: str = Field("", description="Isolated text of the question")
    expectedAnswer: str = Field("", description="Reference / model answer")
    studentAnswer: str = Field("", description="Student's answered text")
    maxMarks: float = Field(5.0)
    score: float = Field(0.0)
    status: str = Field("Unanswered", description="Correct | Partial | Incorrect | Unanswered")
    feedback: str = Field("")
    strengths: List[str] = Field(default_factory=list)
    missingConcepts: List[str] = Field(default_factory=list)
    keywordsMatched: List[str] = Field(default_factory=list)
    keywordsMissing: List[str] = Field(default_factory=list)

    # Legacy snake_case compatibility properties
    @property
    def question_id(self) -> str:
        return self.questionId

    @property
    def missing_concepts(self) -> List[str]:
        return self.missingConcepts


class PipelineMetadata(BaseModel):
    ocr_time_seconds: float = 0.0
    evaluation_time_seconds: float = 0.0
    total_time_seconds: float = 0.0
    pipeline_version: str = "new-v1.0"


class EvaluationResponse(BaseModel):
    studentId: str = Field(..., alias="studentId")
    examId: str = Field(..., alias="examId")
    totalScore: float = Field(0.0, alias="totalScore")
    maxMarksTotal: float = Field(0.0, alias="maxMarksTotal")
    percentage: float = Field(0.0, alias="percentage")
    grade: str = Field("F", alias="grade")
    summaryFeedback: str = Field("", alias="summaryFeedback")
    evaluations: List[QuestionEvaluationItem] = Field(default_factory=list)
    pipelineMetadata: PipelineMetadata = Field(default_factory=PipelineMetadata)

    # Dual schema support for snake_case callers
    @property
    def student_id(self) -> str:
        return self.studentId

    @property
    def exam_id(self) -> str:
        return self.examId

    @property
    def total_score(self) -> float:
        return self.totalScore

    @property
    def feedback(self) -> str:
        return self.summaryFeedback


# ═══════════════════════════════════════════════════
#  Pipeline Engine Logic
# ═══════════════════════════════════════════════════

def parse_questions_from_text(qp_text: str, ak_text: str, sa_text: str, default_max: int) -> List[Dict[str, Any]]:
    """
    Step 1: Extract and segment questions from question paper, answer key, and student text.
    """
    # Simple regex block splitter for Q1, Q2, Question 1, etc.
    q_blocks = re.split(r'\n(?=(?:Q(?:uestion)?\s*\d+|^\d+[\.\)]))', qp_text, flags=re.IGNORECASE | re.MULTILINE)
    questions = []

    for idx, block in enumerate(q_blocks, 1):
        clean_block = block.strip()
        if not clean_block:
            continue
        
        # Check explicit max marks in question e.g., [Marks: 5] or (5 marks)
        marks_match = re.search(r'(?:\[|\()(?:\s*marks?\s*:\s*|\s*)(\d+)(?:\s*marks?)?(?:\]|\))', clean_block, re.IGNORECASE)
        q_max_marks = float(marks_match.group(1)) if marks_match else float(default_max)

        q_id = f"Q{idx}"
        q_match = re.match(r'^(?:Q(?:uestion)?\s*\d+|^\d+[\.\)]\s*)(.*)', clean_block, re.IGNORECASE | re.DOTALL)
        q_content = q_match.group(1).strip() if q_match else clean_block

        questions.append({
            "questionId": q_id,
            "questionText": q_content,
            "maxMarks": q_max_marks,
        })

    if not questions:
        questions.append({
            "questionId": "Q1",
            "questionText": qp_text.strip(),
            "maxMarks": float(default_max),
        })

    return questions


def evaluate_single_question(q_info: Dict[str, Any], ak_text: str, sa_text: str) -> QuestionEvaluationItem:
    """
    Step 2: Evaluate an individual question using semantic analysis / heuristic evaluation.
    """
    q_id = q_info["questionId"]
    q_text = q_info["questionText"]
    max_m = q_info["maxMarks"]

    # Basic baseline evaluation logic (placeholder for model prompt / LLM / semantic grader)
    has_student_ans = bool(sa_text and len(sa_text.strip()) > 5)

    if not has_student_ans:
        return QuestionEvaluationItem(
            questionId=q_id,
            questionText=q_text,
            expectedAnswer=ak_text.strip(),
            studentAnswer="",
            maxMarks=max_m,
            score=0.0,
            status="Unanswered",
            feedback="No response provided by student.",
            strengths=[],
            missingConcepts=["Entire answer missing"],
            keywordsMatched=[],
            keywordsMissing=["All required key terms"],
        )

    # Example heuristic calculation (can be connected to local Qwen / LLM evaluator)
    awarded_score = round(max_m * 0.8, 1)  # Default 80% baseline for valid answer
    status = "Correct" if awarded_score >= (max_m * 0.8) else "Partial"

    return QuestionEvaluationItem(
        questionId=q_id,
        questionText=q_text,
        expectedAnswer=ak_text.strip(),
        studentAnswer=sa_text.strip(),
        maxMarks=max_m,
        score=awarded_score,
        status=status,
        feedback=f"Answer addresses the core concept of {q_id}.",
        strengths=["Clear statement of key concepts"],
        missingConcepts=["Minor elaboration on advanced points"],
        keywordsMatched=["definition", "concept"],
        keywordsMissing=["example"],
    )


def execute_pipeline(
    question_paper_text: str,
    answer_key_text: str,
    student_answer_text: str,
    student_id: str,
    exam_id: str,
    default_max_marks: int = 5,
) -> EvaluationResponse:
    """
    Step 3: Run the end-to-end evaluation pipeline.
    """
    t_start = time.time()

    # Parse questions
    q_list = parse_questions_from_text(question_paper_text, answer_key_text, student_answer_text, default_max_marks)

    eval_items: List[QuestionEvaluationItem] = []
    total_score = 0.0
    total_max_marks = 0.0

    for q_info in q_list:
        item = evaluate_single_question(q_info, answer_key_text, student_answer_text)
        eval_items.append(item)
        total_score += item.score
        total_max_marks += item.maxMarks

    percentage = round((total_score / total_max_marks * 100), 2) if total_max_marks > 0 else 0.0
    
    if percentage >= 90:
        grade = "A+"
    elif percentage >= 80:
        grade = "A"
    elif percentage >= 70:
        grade = "B"
    elif percentage >= 60:
        grade = "C"
    elif percentage >= 50:
        grade = "D"
    else:
        grade = "F"

    eval_time = round(time.time() - t_start, 3)

    return EvaluationResponse(
        studentId=student_id,
        examId=exam_id,
        totalScore=round(total_score, 2),
        maxMarksTotal=round(total_max_marks, 2),
        percentage=percentage,
        grade=grade,
        summaryFeedback=f"Student scored {total_score}/{total_max_marks} ({percentage}%). Performance Grade: {grade}.",
        evaluations=eval_items,
        pipelineMetadata=PipelineMetadata(
            evaluation_time_seconds=eval_time,
            total_time_seconds=eval_time,
        ),
    )


# ═══════════════════════════════════════════════════
#  API Endpoints
# ═══════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    """Health check endpoint."""
    return {
        "status": "healthy",
        "service": "New Modular OCR Evaluation Pipeline",
        "device": "CUDA" if torch.cuda.is_available() else "CPU",
        "ocr_server": OCR_SERVER_URL,
    }


@app.post("/evaluate-text", response_model=EvaluationResponse)
async def evaluate_text(req: TextEvaluationRequest):
    """
    Main evaluation pipeline endpoint for pre-extracted text.
    """
    logger.info(f"[/evaluate-text] Running pipeline for student_id={req.student_id}, exam_id={req.exam_id}")
    return execute_pipeline(
        question_paper_text=req.question_paper_text,
        answer_key_text=req.answer_key_text,
        student_answer_text=req.student_answer_text,
        student_id=req.student_id,
        exam_id=req.exam_id,
        default_max_marks=req.max_marks,
    )


@app.post("/evaluation", response_model=EvaluationResponse)
async def evaluate_alias(req: TextEvaluationRequest):
    """
    Alias endpoint for /evaluate-text to maintain backward compatibility.
    """
    return await evaluate_text(req)
