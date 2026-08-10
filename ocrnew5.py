"""
ocrnew5.py — AI Evaluation Pipeline v5.5 (Multi-Agent StateGraph Architecture)

Architecture Flow (LangGraph Multi-Agent StateGraph):
              ┌───────────────────────────┐
              │  START EVALUATION STREAM  │
              └─────────────┬─────────────┘
                            ▼
              ┌───────────────────────────┐
              │ Document Analyzer Agent   │
              └─────────────┬─────────────┘
                            ▼
              ┌───────────────────────────┐
              │ Question Extraction Agent │
              └─────────────┬─────────────┘
                            ▼
              ┌───────────────────────────┐
              │ VL Vision Structuring     │
              └─────────────┬─────────────┘
                            ▼
              ┌───────────────────────────┐
              │ Rubric Extractor Agent    │
              └─────────────┬─────────────┘
                            ▼
              ┌───────────────────────────┐
              │ Answer Mapping Agent      │
              └─────────────┬─────────────┘
                            ▼
              ┌───────────────────────────┐
              │ Validation Agent          │
              └─────────────┬─────────────┘
                            ▼
              ┌───────────────────────────┐
              │ Evidence Assembly Agent   │
              └─────────────┬─────────────┘
                            ▼
              ┌───────────────────────────┐
              │ Dual Evaluator Agent      │
              │ (Qwen3B Text + QwenVL Vis)│
              └─────────────┬─────────────┘
                            ▼
              ┌───────────────────────────┐
              │ Mark Aggregator Agent     │
              └─────────────┬─────────────┘
                            ▼
              ┌───────────────────────────┐
              │ END / EVALUATION RESPONSE │
              └───────────────────────────┘

Port: 8006
Endpoints:
  GET  /health          — Health check and multi-agent graph capabilities
  POST /evaluate-text   — Primary JSON evaluation endpoint (invoked by Node.js backend)
  POST /evaluate-paper  — Multipart file upload evaluation endpoint (PDF/Image)
  POST /evaluate-graph  — LangGraph execution endpoint returning full trace & output

Run:
    python -m uvicorn ocrnew5:app --host 127.0.0.1 --port 8006 --reload
"""

import sys
import os
import re
import io
import json
import math
import time
import asyncio
import logging
import hashlib
import tempfile
import unicodedata
from enum import Enum
from pathlib import Path
from typing import Optional, Any, List, Dict, Tuple, Set
from dataclasses import dataclass, field as dc_field
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime

from typing_extensions import TypedDict
from langchain_core.language_models.llms import LLM
from langchain_core.callbacks.manager import CallbackManagerForLLMRun
from langgraph.graph import StateGraph, START, END

import threading
import functools
import httpx
import torch
from pydantic import BaseModel, Field
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Fix Windows console encoding
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(PROJECT_ROOT))

# ── Dual Intel Xeon / Multi-Core Optimizations ───
CPU_COUNT = os.cpu_count() or 56
OPTIMAL_TORCH_THREADS = min(max(CPU_COUNT // 4, 8), 16)

os.environ["MKL_NUM_THREADS"] = str(OPTIMAL_TORCH_THREADS)
os.environ["OMP_NUM_THREADS"] = str(OPTIMAL_TORCH_THREADS)
os.environ["KMP_AFFINITY"] = "granularity=fine,compact,1,0"
os.environ["KMP_BLOCKTIME"] = "1"

# Thread locks for model generation thread-safety
_text_model_lock = threading.Lock()
_vl_model_lock = threading.Lock()

# Shared singleton HTTP client pool for OCR communication
_ocr_client: Optional[httpx.AsyncClient] = None

def get_ocr_client() -> httpx.AsyncClient:
    global _ocr_client
    if _ocr_client is None or _ocr_client.is_closed:
        _ocr_client = httpx.AsyncClient(
            limits=httpx.Limits(max_keepalive_connections=20, max_connections=100),
            timeout=180.0,
        )
    return _ocr_client

# Optimize PyTorch CPU Thread Engine
torch.set_num_threads(OPTIMAL_TORCH_THREADS)

# Import OCR pipeline v5 core utilities
from ocr_eval_pipeline5 import (
    PIPELINE_VERSION, PROMPT_VERSION, DEFAULT_MAX_MARKS, OCR_SERVER_URL,
    HIGH_CONFIDENCE_THRESHOLD, LOW_CONFIDENCE_THRESHOLD, MAX_PARALLEL_WORKERS,
    MAX_LLM_RETRIES, HAS_PYMUPDF, HAS_PIL, MULTI_PAGE_MAX, MAX_QUESTIONS_SUPPORTED,
    MODEL_PATH_OR_ID, TEXT_MODEL_VERSION, VL_MODEL_VERSION,
    QuestionType, AlignmentPath, RubricAllocation, EvidencePacket,
    TextEvalResult, VisualEvalResult, AggregatedResult, OCRBlock,
    AlignmentScore, ExtractedQuestion, ConceptScore, QuestionEvaluation,
    PipelineMetadata, MarksDetail, ConfidenceDetail, AnswerComparison,
    ReasoningDetail, KeywordsDetail, QuestionEvalDetail, QuestionItem,
    SummaryDetail, FeedbackDetail, MetadataDetail, EvaluationResponse,
    TextEvaluationRequest, detect_and_parse_input, normalize_ocr_text,
    compute_ocr_confidence, detect_script, build_question_tree, parse_model_answers,
    vl_structure_answer_key, vl_structure_student_answers, extract_rubric_allocations,
    map_answers_vl_first, validate_mappings, assemble_evidence, evaluate_text,
    evaluate_visual, aggregate_marks, extract_page_images_from_pdf,
    extract_page_images_from_image, cleanup_session_images, call_ocr_server,
    generate_with_qwen, generate_with_vl, get_text_model, get_vl_model
)

from contextlib import asynccontextmanager

# ── Logging ──
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(name)s | %(message)s",
)
logger = logging.getLogger("ocrnew5")

# ── Lifespan Preloading ──
@asynccontextmanager
async def lifespan(app: FastAPI):
    logger.info("🚀 Pre-loading Qwen 3B Text & Qwen-VL Vision weights at startup...")
    try:
        get_text_model()
        get_vl_model()
        logger.info("✅ All AI Model weights pre-loaded successfully! Server is ready.")
    except Exception as e:
        logger.warning(f"⚠️ Model pre-loading warning (will lazy-load on demand): {e}")
    yield
    logger.info("👋 Shutting down ocrnew5 pipeline server.")

# ── FastAPI App Setup ──
app = FastAPI(
    title="AI Evaluation Pipeline v5.5 (Multi-Agent Engine)",
    description="Next-generation multi-agent AI exam evaluation system powered by LangGraph & LangChain.",
    version="5.5.0",
    lifespan=lifespan,
)


app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ═══════════════════════════════════════════════════
#  LangChain & LangGraph Multi-Agent Architecture
# ═══════════════════════════════════════════════════

class Qwen3BLangChainLLM(LLM):
    """LangChain LLM wrapper for local Qwen 3B Text Evaluation Model."""
    max_new_tokens: int = 300

    @property
    def _llm_type(self) -> str:
        return "qwen_3b_evaluator"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        return generate_with_qwen(prompt, max_new_tokens=tokens)


class QwenVLLangChainLLM(LLM):
    """LangChain LLM wrapper for local Qwen2.5-VL Vision-Language Model."""
    max_new_tokens: int = 500

    @property
    def _llm_type(self) -> str:
        return "qwen_vl_evaluator"

    def _call(
        self,
        prompt: str,
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> str:
        tokens = kwargs.get("max_new_tokens", self.max_new_tokens)
        image_paths = kwargs.get("image_paths")
        return generate_with_vl(prompt, image_paths=image_paths, max_new_tokens=tokens)


# Instantiate global LangChain LLM components
qwen_text_agent = Qwen3BLangChainLLM()
qwen_vl_agent = QwenVLLangChainLLM()


class PipelineGraphStateOCRNew5(TypedDict, total=False):
    """Multi-Agent Pipeline Graph State for ocrnew5."""
    # Input parameters
    question_paper_text: str
    answer_key_text: str
    student_answer_text: str
    student_id: str
    exam_id: str
    default_max_marks: float
    student_page_images: List[str]
    answer_key_page_images: List[str]
    session_id: str
    pipeline_start: float

    # Metadata tracking
    metadata: PipelineMetadata

    # Agent 1: Document Analyzer outputs
    doc_format: str
    clean_questions: str
    clean_answers: str
    clean_student: str
    student_ocr_conf: float
    student_script: str
    non_latin_ratio: float
    doc_word_count: int

    # Agent 2: Question Extraction outputs
    questions: List[ExtractedQuestion]
    model_answers: Dict[str, str]

    # Agent 3: VL Structuring outputs
    vl_answer_key_struct: List[Dict[str, Any]]
    vl_student_struct: List[Dict[str, Any]]

    # Agent 4: Rubric Extraction outputs
    rubric_allocations: Dict[str, RubricAllocation]

    # Agent 5: Answer Mapping outputs
    aligned_texts: Dict[str, str]
    confidence_map: Dict[str, float]
    signal_map: Dict[str, Dict[str, float]]
    page_map: Dict[str, int]

    # Agent 6: Validation outputs
    validation_warnings: List[str]

    # Agent 7: Evidence Assembly outputs
    evidence_packets: List[EvidencePacket]

    # Agent 8: Dual Evaluation outputs
    aggregated_results: List[AggregatedResult]

    # Agent 9: Mark Aggregation & Response outputs
    final_response: EvaluationResponse
    error: Optional[str]


# ── LangGraph Multi-Agent Nodes ──

def document_analyzer_agent_node(state: PipelineGraphStateOCRNew5) -> Dict[str, Any]:
    """
    Agent 1: Document Analyzer Agent
    Inspects incoming document format, normalizes OCR text, measures script & OCR confidence,
    and analyzes overall document structure.
    """
    metadata = state.get("metadata", PipelineMetadata())

    q_raw = state.get("question_paper_text", "")
    a_raw = state.get("answer_key_text", "")
    s_raw = state.get("student_answer_text", "")

    q_text = detect_and_parse_input(q_raw)
    a_text = detect_and_parse_input(a_raw)
    s_text = s_raw

    t0 = time.time()
    clean_questions = q_text
    clean_answers = normalize_ocr_text(a_text)
    clean_student = normalize_ocr_text(s_text)
    metadata.cleaning_time_seconds = round(time.time() - t0, 3)

    student_ocr_conf = compute_ocr_confidence(clean_student)
    student_script, non_latin_ratio = detect_script(clean_student)
    word_count = len(clean_student.split())

    doc_format = "JSON" if (q_raw.strip().startswith("{") or q_raw.strip().startswith("[")) else "Text"

    logger.info(f"[Document Analyzer Agent] Format: {doc_format} | Words: {word_count} | OCR Conf: {student_ocr_conf:.2f} | Script: {student_script}")

    return {
        "doc_format": doc_format,
        "clean_questions": clean_questions,
        "clean_answers": clean_answers,
        "clean_student": clean_student,
        "student_ocr_conf": student_ocr_conf,
        "student_script": student_script,
        "non_latin_ratio": non_latin_ratio,
        "doc_word_count": word_count,
        "metadata": metadata,
    }


def question_extraction_agent_node(state: PipelineGraphStateOCRNew5) -> Dict[str, Any]:
    """
    Agent 2: Question Extraction Agent
    Parses question paper hierarchy, sub-questions, MCQ options, max marks, and parses model answer key.
    """
    clean_questions = state.get("clean_questions", "")
    clean_answers = state.get("clean_answers", "")
    default_max_marks = float(state.get("default_max_marks", DEFAULT_MAX_MARKS))
    metadata = state.get("metadata", PipelineMetadata())

    t0 = time.time()
    questions = build_question_tree(clean_questions, default_max_marks)

    if not questions:
        questions.append(ExtractedQuestion(
            question_id="Q.1",
            question_text=clean_questions[:300] if clean_questions else "Evaluate student answer sheet.",
            question_type=QuestionType.SUBJECTIVE,
            max_marks=float(default_max_marks),
            extraction_confidence=0.70,
        ))

    metadata.extraction_time_seconds = round(time.time() - t0, 3)
    metadata.questions_detected = len(questions)
    metadata.question_types_found = list(set(q.question_type.value for q in questions))

    model_answers = parse_model_answers(clean_answers, questions)

    logger.info(f"[Question Extraction Agent] Extracted {len(questions)} questions, {len(model_answers)} model answers")

    return {
        "questions": questions,
        "model_answers": model_answers,
        "metadata": metadata,
    }


def vl_structuring_agent_node(state: PipelineGraphStateOCRNew5) -> Dict[str, Any]:
    """Agent 3: Qwen-VL Vision Structuring Agent."""
    clean_answers = state.get("clean_answers", "")
    clean_student = state.get("clean_student", "")
    default_max_marks = int(state.get("default_max_marks", DEFAULT_MAX_MARKS))
    metadata = state.get("metadata", PipelineMetadata())

    t0 = time.time()
    vl_answer_key_struct = vl_structure_answer_key(clean_answers, default_max_marks)
    vl_student_struct = vl_structure_student_answers(clean_student)
    metadata.vl_structuring_time_seconds = round(time.time() - t0, 3)

    logger.info(f"[VL Structuring Agent] Extracted {len(vl_answer_key_struct)} key questions, {len(vl_student_struct)} student blocks")

    return {
        "vl_answer_key_struct": vl_answer_key_struct,
        "vl_student_struct": vl_student_struct,
        "metadata": metadata,
    }


def rubric_extractor_agent_node(state: PipelineGraphStateOCRNew5) -> Dict[str, Any]:
    """Agent 4: Dynamic Rubric Extractor Agent (text vs. visual mark allocation)."""
    questions = state.get("questions", [])
    vl_answer_key_struct = state.get("vl_answer_key_struct", [])
    default_max_marks = int(state.get("default_max_marks", DEFAULT_MAX_MARKS))
    metadata = state.get("metadata", PipelineMetadata())

    t0 = time.time()
    rubric_allocations = extract_rubric_allocations(vl_answer_key_struct, questions, default_max_marks)
    metadata.rubric_extraction_time_seconds = round(time.time() - t0, 3)

    diagram_count = sum(1 for r in rubric_allocations.values() if r.has_diagram)
    metadata.diagram_questions_count = diagram_count

    logger.info(f"[Rubric Extractor Agent] Rubric extracted: {len(rubric_allocations)} allocations ({diagram_count} diagram questions)")

    return {
        "rubric_allocations": rubric_allocations,
        "metadata": metadata,
    }


def answer_mapping_agent_node(state: PipelineGraphStateOCRNew5) -> Dict[str, Any]:
    """Agent 5: Answer Alignment & Mapping Agent (VL-First + 7-Signal DP Fallback)."""
    questions = state.get("questions", [])
    vl_student_struct = state.get("vl_student_struct", [])
    clean_student = state.get("clean_student", "")
    model_answers = state.get("model_answers", {})
    student_script = state.get("student_script", "Latin")
    metadata = state.get("metadata", PipelineMetadata())

    t0 = time.time()
    aligned_texts, confidence_map, signal_map, page_map = map_answers_vl_first(
        questions, vl_student_struct, clean_student, model_answers, student_script
    )
    metadata.alignment_time_seconds = round(time.time() - t0, 3)

    high_conf_ids = [q.question_id for q in questions if confidence_map.get(q.question_id, 0) >= HIGH_CONFIDENCE_THRESHOLD]
    low_conf_ids = [q.question_id for q in questions if confidence_map.get(q.question_id, 0) < HIGH_CONFIDENCE_THRESHOLD]
    metadata.high_confidence_count = len(high_conf_ids)
    metadata.low_confidence_count = len(low_conf_ids)

    logger.info(f"[Answer Mapping Agent] Mapped {len(high_conf_ids)} high-conf, {len(low_conf_ids)} low-conf")

    return {
        "aligned_texts": aligned_texts,
        "confidence_map": confidence_map,
        "signal_map": signal_map,
        "page_map": page_map,
        "metadata": metadata,
    }


def validation_agent_node(state: PipelineGraphStateOCRNew5) -> Dict[str, Any]:
    """Agent 6: Validation Agent (cross-validation and sanity checks)."""
    questions = state.get("questions", [])
    aligned_texts = state.get("aligned_texts", {})
    confidence_map = state.get("confidence_map", {})
    rubric_allocations = state.get("rubric_allocations", {})

    validation_warnings = validate_mappings(questions, aligned_texts, confidence_map, rubric_allocations)

    logger.info(f"[Validation Agent] {len(validation_warnings)} validation warnings")

    return {
        "validation_warnings": validation_warnings,
    }


def evidence_assembly_agent_node(state: PipelineGraphStateOCRNew5) -> Dict[str, Any]:
    """Agent 7: Evidence Assembly Agent."""
    questions = state.get("questions", [])
    aligned_texts = state.get("aligned_texts", {})
    confidence_map = state.get("confidence_map", {})
    signal_map = state.get("signal_map", {})
    page_map = state.get("page_map", {})
    rubric_allocations = state.get("rubric_allocations", {})
    model_answers = state.get("model_answers", {})
    student_page_images = state.get("student_page_images", [])
    answer_key_page_images = state.get("answer_key_page_images", [])
    student_ocr_conf = state.get("student_ocr_conf", 1.0)
    metadata = state.get("metadata", PipelineMetadata())

    evidence_packets = assemble_evidence(
        questions=questions,
        aligned_texts=aligned_texts,
        confidence_map=confidence_map,
        signal_map=signal_map,
        page_map=page_map,
        rubric_allocations=rubric_allocations,
        model_answers=model_answers,
        student_page_images=student_page_images,
        answer_key_page_images=answer_key_page_images,
        ocr_confidence=student_ocr_conf,
    )

    if student_page_images:
        metadata.page_images_extracted = len(student_page_images)

    logger.info(f"[Evidence Assembly Agent] Assembled {len(evidence_packets)} evidence packets")

    return {
        "evidence_packets": evidence_packets,
        "metadata": metadata,
    }


def dual_evaluation_agent_node(state: PipelineGraphStateOCRNew5) -> Dict[str, Any]:
    """Agent 8: Parallel Dual Evaluator Agent (Qwen 3B Text + Qwen-VL Visual)."""
    evidence_packets = state.get("evidence_packets", [])
    metadata = state.get("metadata", PipelineMetadata())

    t0_eval = time.time()

    def process_question(ev: EvidencePacket) -> AggregatedResult:
        text_res = evaluate_text(ev)
        visual_res = evaluate_visual(ev)
        return aggregate_marks(ev, text_res, visual_res)

    with ThreadPoolExecutor(max_workers=MAX_PARALLEL_WORKERS) as executor:
        aggregated_results = list(executor.map(process_question, evidence_packets))

    metadata.text_evaluation_time_seconds = round(time.time() - t0_eval, 3)
    metadata.visual_evaluations_run = sum(1 for r in aggregated_results if not r.visual_result.skipped)

    logger.info(f"[Dual Evaluator Agent] Evaluated {len(aggregated_results)} questions in {metadata.text_evaluation_time_seconds}s")

    return {
        "aggregated_results": aggregated_results,
        "metadata": metadata,
    }


def mark_aggregator_agent_node(state: PipelineGraphStateOCRNew5) -> Dict[str, Any]:
    """Agent 9: Mark Aggregator & Response Builder Agent."""
    aggregated_results = state.get("aggregated_results", [])
    evidence_packets = state.get("evidence_packets", [])
    confidence_map = state.get("confidence_map", {})
    student_id = state.get("student_id", "1")
    exam_id = state.get("exam_id", "1")
    pipeline_start = state.get("pipeline_start", time.time())
    student_ocr_conf = state.get("student_ocr_conf", 1.0)
    validation_warnings = state.get("validation_warnings", [])
    metadata = state.get("metadata", PipelineMetadata())

    total_score = sum(r.final_score for r in aggregated_results)
    max_possible = sum(r.final_max_marks for r in aggregated_results)
    pct = (total_score / max_possible * 100.0) if max_possible > 0 else 0.0

    metadata.human_review_count = sum(1 for r in aggregated_results if r.needs_human_review)
    metadata.total_time_seconds = round(time.time() - pipeline_start, 3)

    correct_cnt = sum(1 for r in aggregated_results if r.status == "Correct")
    partial_cnt = sum(1 for r in aggregated_results if r.status == "Partial")
    incorrect_cnt = sum(1 for r in aggregated_results if r.status == "Incorrect")
    unanswered_cnt = sum(1 for r in aggregated_results if r.status == "Unanswered")

    grade = (
        "A+" if pct >= 90 else
        "A" if pct >= 80 else
        "B" if pct >= 70 else
        "C" if pct >= 50 else
        "D" if pct >= 35 else "F"
    )

    all_strengths = list(dict.fromkeys(
        s for r in aggregated_results for s in r.text_result.strengths
    ))[:3] or ["Demonstrates fundamental understanding."]

    all_improvements = list(dict.fromkeys(
        m for r in aggregated_results for m in r.text_result.missing_concepts
    ))[:3] or ["Provide complete answers with key terminologies."]

    for r in aggregated_results:
        if not r.visual_result.skipped and r.visual_result.labels_missing:
            all_improvements.append(f"Missing diagram labels: {', '.join(r.visual_result.labels_missing[:3])}")
            break

    summary_obj = SummaryDetail(
        obtainedMarks=round(total_score, 1),
        totalMarks=round(max_possible, 1),
        percentage=round(pct, 1),
        grade=grade,
        questionsEvaluated=len(aggregated_results),
        correct=correct_cnt, partial=partial_cnt,
        incorrect=incorrect_cnt, unanswered=unanswered_cnt,
    )

    feedback_obj = FeedbackDetail(
        overall=f"Student scored {round(total_score, 1)}/{round(max_possible, 1)} ({pct:.1f}%, Grade: {grade}).",
        strengths=all_strengths,
        improvements=all_improvements,
    )

    question_items: List[QuestionItem] = []
    for idx, agg in enumerate(aggregated_results):
        ev = evidence_packets[idx]
        tr = agg.text_result
        vr = agg.visual_result

        combined_feedback = tr.feedback or tr.reasoning or ""
        if not vr.skipped and vr.diagram_feedback:
            combined_feedback += f"\n[Diagram]: {vr.diagram_feedback}"

        combined_strengths = (tr.strengths or ["Attempted solution."])
        if not vr.skipped and vr.labels_correct:
            combined_strengths.append(f"Diagram labels correct: {', '.join(vr.labels_correct[:3])}")

        combined_missing = (tr.missing_concepts or [])
        if not vr.skipped and vr.labels_missing:
            combined_missing.append(f"Diagram labels missing: {', '.join(vr.labels_missing[:3])}")

        conf_reason = (
            "Exact match." if agg.status == "Correct" else
            "High semantic similarity." if agg.status == "Partial" else
            "Low semantic match."
        )

        q_eval = QuestionEvalDetail(
            confidence=ConfidenceDetail(score=round(ev.alignment_confidence, 2), reason=conf_reason),
            reasoning=ReasoningDetail(
                analysis=combined_feedback or f"Scored {agg.final_score}/{agg.final_max_marks}.",
                comparison=AnswerComparison(student=ev.student_answer[:200], expected=ev.expected_answer[:200]),
                conceptsIdentified=combined_strengths,
                missingConcepts=combined_missing,
                markJustification=(
                    f"Text: {tr.score}/{agg.rubric.text_marks} + "
                    f"Visual: {vr.score}/{agg.rubric.visual_marks} = "
                    f"{agg.final_score}/{agg.final_max_marks}"
                ),
            ),
            feedback=combined_feedback,
            strengths=combined_strengths,
            improvements=combined_missing if agg.status != "Correct" else [],
            keywords=KeywordsDetail(matched=tr.keywords_matched, missing=tr.keywords_missing),
            alignment_path=ev.alignment_path,
        )

        question_items.append(QuestionItem(
            questionId=f"Q{idx + 1}",
            questionText=ev.question_text,
            studentAnswer=ev.student_answer[:300],
            expectedAnswer=ev.expected_answer[:300],
            marks=MarksDetail(
                obtained=round(agg.final_score, 1),
                maximum=round(agg.final_max_marks, 1),
                text_obtained=round(tr.score, 1),
                text_maximum=round(agg.rubric.text_marks, 1),
                visual_obtained=round(vr.score if not vr.skipped else 0.0, 1),
                visual_maximum=round(agg.rubric.visual_marks, 1),
            ),
            status=agg.status,
            evaluation=q_eval,
        ))

    eval_id = f"IB{hashlib.md5((student_id + exam_id + str(pipeline_start)).encode()).hexdigest()[:6].upper()}"

    avg_conf = sum(confidence_map.values()) / max(len(confidence_map), 1)
    metadata_obj = MetadataDetail(
        model=f"{TEXT_MODEL_VERSION} + {VL_MODEL_VERSION}",
        evaluationVersion="v5.5-ocrnew5-langgraph",
        processingTime=round(metadata.total_time_seconds, 2),
        timestamp=datetime.utcnow().strftime("%Y-%m-%dT%H:%M:%SZ"),
        alignmentStats={
            "avg_alignment_confidence": round(avg_conf, 3),
            "high_confidence_count": metadata.high_confidence_count,
            "low_confidence_count": metadata.low_confidence_count,
            "vl_structuring_time_s": metadata.vl_structuring_time_seconds,
            "diagram_questions": metadata.diagram_questions_count,
            "visual_evaluations_run": metadata.visual_evaluations_run,
            "page_images_extracted": metadata.page_images_extracted,
            "alignment_time_s": metadata.alignment_time_seconds,
            "validation_warnings": len(validation_warnings),
        },
    )

    pipeline_log = {
        "pipeline_name": "ocrnew5",
        "pipeline_version": "5.5.0",
        "architecture": "LangGraph 9-Agent StateGraph Architecture",
        "student_id": student_id,
        "exam_id": exam_id,
        "questions_detected": len(evidence_packets),
        "diagram_questions": metadata.diagram_questions_count,
        "visual_evals_run": metadata.visual_evaluations_run,
        "ocr_confidence": student_ocr_conf,
        "avg_alignment_confidence": round(avg_conf, 3),
        "total_score": round(total_score, 1),
        "max_possible": round(max_possible, 1),
        "grade": grade,
        "total_time_s": metadata.total_time_seconds,
    }
    logger.info(f"[Mark Aggregator Agent] Evaluation complete: {json.dumps(pipeline_log)}")

    response = EvaluationResponse(
        status="success",
        evaluationId=eval_id,
        studentId=student_id,
        examId=exam_id,
        summary=summary_obj,
        feedback=feedback_obj,
        questions=question_items,
        metadata=metadata_obj,
    )

    return {
        "metadata": metadata,
        "final_response": response,
    }


# ── StateGraph Construction ──

def create_eval_pipeline_graph_ocrnew5() -> Any:
    """Build and compile the 9-Agent StateGraph for ocrnew5."""
    workflow = StateGraph(PipelineGraphStateOCRNew5)

    # Register Nodes
    workflow.add_node("document_analyzer", document_analyzer_agent_node)
    workflow.add_node("question_extraction", question_extraction_agent_node)
    workflow.add_node("vl_structuring", vl_structuring_agent_node)
    workflow.add_node("rubric_extractor", rubric_extractor_agent_node)
    workflow.add_node("answer_mapping", answer_mapping_agent_node)
    workflow.add_node("validation", validation_agent_node)
    workflow.add_node("evidence_assembly", evidence_assembly_agent_node)
    workflow.add_node("dual_evaluation", dual_evaluation_agent_node)
    workflow.add_node("mark_aggregator", mark_aggregator_agent_node)

    # Register Flow Edges
    workflow.add_edge(START, "document_analyzer")
    workflow.add_edge("document_analyzer", "question_extraction")
    workflow.add_edge("question_extraction", "vl_structuring")
    workflow.add_edge("vl_structuring", "rubric_extractor")
    workflow.add_edge("rubric_extractor", "answer_mapping")
    workflow.add_edge("answer_mapping", "validation")
    workflow.add_edge("validation", "evidence_assembly")
    workflow.add_edge("evidence_assembly", "dual_evaluation")
    workflow.add_edge("dual_evaluation", "mark_aggregator")
    workflow.add_edge("mark_aggregator", END)

    return workflow.compile()


# Compile global application graph for ocrnew5
eval_pipeline_ocrnew5_app = create_eval_pipeline_graph_ocrnew5()


def run_full_pipeline_ocrnew5(
    question_paper_text: str,
    answer_key_text: str,
    student_answer_text: str,
    student_id: str = "1",
    exam_id: str = "1",
    default_max_marks: int = DEFAULT_MAX_MARKS,
    student_page_images: Optional[List[str]] = None,
    answer_key_page_images: Optional[List[str]] = None,
) -> EvaluationResponse:
    """
    Main Orchestrator for ocrnew5 pipeline.
    Runs inputs through the 9-Agent StateGraph.
    """
    pipeline_start = time.time()
    session_id = hashlib.md5(f"{student_id}{exam_id}{pipeline_start}".encode()).hexdigest()[:12]

    initial_state: PipelineGraphStateOCRNew5 = {
        "question_paper_text": question_paper_text,
        "answer_key_text": answer_key_text,
        "student_answer_text": student_answer_text,
        "student_id": student_id,
        "exam_id": exam_id,
        "default_max_marks": float(default_max_marks),
        "student_page_images": student_page_images or [],
        "answer_key_page_images": answer_key_page_images or [],
        "session_id": session_id,
        "metadata": PipelineMetadata(),
        "pipeline_start": pipeline_start,
    }

    final_state = eval_pipeline_ocrnew5_app.invoke(initial_state)

    if "final_response" in final_state and final_state["final_response"]:
        return final_state["final_response"]
    else:
        raise RuntimeError(f"ocrnew5 multi-agent execution failed: {final_state.get('error', 'Unknown state error')}")


# ═══════════════════════════════════════════════════
#  FastAPI API Endpoints (Port 8006)
# ═══════════════════════════════════════════════════

@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "service": "AI Evaluation Pipeline v5.5 (ocrnew5 Multi-Agent Engine)",
        "pipeline_version": "5.5.0",
        "port": 8006,
        "architecture": "LangGraph 9-Agent StateGraph Architecture",
        "agents": [
            "Agent 1: Document Analyzer Agent",
            "Agent 2: Question Extraction Agent",
            "Agent 3: Qwen-VL Vision Structuring Agent",
            "Agent 4: Dynamic Rubric Extractor Agent",
            "Agent 5: Answer Alignment & Mapping Agent",
            "Agent 6: Validation Agent",
            "Agent 7: Question Evidence Assembly Agent",
            "Agent 8: Parallel Dual Evaluator Agent (Text + Visual)",
            "Agent 9: Mark Aggregator & Response Builder Agent",
        ],
        "ocr_server": OCR_SERVER_URL,
        "capabilities": {
            "max_pages": MULTI_PAGE_MAX,
            "max_questions": MAX_QUESTIONS_SUPPORTED,
            "diagram_evaluation": True,
            "rubric_driven_scoring": True,
            "multi_agent_graph": True,
        },
    }


@app.post("/evaluate-text", response_model=EvaluationResponse)
async def evaluate_text_endpoint(req: TextEvaluationRequest):
    """Primary JSON text API endpoint for Node.js backend integration (Port 8006)."""
    logger.info(f"[/evaluate-text ocrnew5] student_id={req.student_id}, exam_id={req.exam_id}")
    return run_full_pipeline_ocrnew5(
        question_paper_text=req.question_paper_text,
        answer_key_text=req.answer_key_text,
        student_answer_text=req.student_answer_text,
        student_id=req.student_id,
        exam_id=req.exam_id,
        default_max_marks=req.max_marks,
    )


@app.post("/evaluate-paper", response_model=EvaluationResponse)
async def evaluate_paper_endpoint(
    question_file: UploadFile = File(..., description="Question paper (JSON/Text/PDF)"),
    answer_key_file: UploadFile = File(..., description="Model answer key (JSON/Text/PDF)"),
    student_file: UploadFile = File(..., description="Student answer sheet (image/PDF)"),
    student_id: str = Form("1"),
    exam_id: str = Form("1"),
    max_marks: int = Form(DEFAULT_MAX_MARKS),
):
    """File upload evaluation endpoint (Port 8006)."""
    t0 = time.time()
    session_id = hashlib.md5(f"{student_id}{exam_id}{t0}".encode()).hexdigest()[:12]

    q_bytes = await question_file.read()
    a_bytes = await answer_key_file.read()
    s_bytes = await student_file.read()

    student_page_images: List[str] = []
    answer_key_page_images: List[str] = []

    s_fname = (student_file.filename or "student.pdf").lower()
    a_fname = (answer_key_file.filename or "answer_key.json").lower()

    if s_fname.endswith(".pdf"):
        student_page_images = extract_page_images_from_pdf(s_bytes, session_id, "student")
    elif any(s_fname.endswith(ext) for ext in [".png", ".jpg", ".jpeg", ".webp"]):
        student_page_images = extract_page_images_from_image(s_bytes, session_id, s_fname)

    if a_fname.endswith(".pdf"):
        answer_key_page_images = extract_page_images_from_pdf(a_bytes, session_id, "anskey")

    question_text = await call_ocr_server(q_bytes, question_file.filename or "question.json")
    answer_key_text = await call_ocr_server(a_bytes, answer_key_file.filename or "answer_key.json")
    student_text = await call_ocr_server(s_bytes, student_file.filename or "student.pdf")

    try:
        result = run_full_pipeline_ocrnew5(
            question_paper_text=question_text,
            answer_key_text=answer_key_text,
            student_answer_text=student_text,
            student_id=student_id,
            exam_id=exam_id,
            default_max_marks=max_marks,
            student_page_images=student_page_images,
            answer_key_page_images=answer_key_page_images,
        )
    finally:
        cleanup_session_images(session_id)

    return result


if __name__ == "__main__":
    import uvicorn
    logger.info("🚀 Launching ocrnew5 AI Evaluation Server on http://127.0.0.1:8006")
    uvicorn.run("ocrnew5:app", host="127.0.0.1", port=8006, reload=True)
