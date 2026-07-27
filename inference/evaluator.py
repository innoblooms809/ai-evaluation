"""
Inference evaluator: loads the base model in 4-bit, applies the trained
LoRA adapter, and generates structured evaluations.

Production usage:
    from inference.evaluator import AnswerSheetEvaluator

    evaluator = AnswerSheetEvaluator("outputs/checkpoints/final")
    result = evaluator.evaluate(
        subject="Physics",
        question="What is gravity?",
        maximum_marks=5,
        rubric_criteria=[
            {"name": "Core concept", "marks": 2, "description": "States the correct definition."},
            {"name": "Key points", "marks": 2, "description": "Includes supporting details."},
            {"name": "Clarity", "marks": 1, "description": "Clear explanation."},
        ],
        reference_answer="Gravity is the force of attraction between masses.",
        student_answer="Gravity pulls things down.",
    )
    print(result)
"""

from __future__ import annotations

import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import torch
from transformers import PreTrainedTokenizerBase

from data.preprocessor import SYSTEM_PROMPT, _build_user_prompt
from model.loader import load_for_inference

logger = logging.getLogger(__name__)


@dataclass
class EvaluationResult:
    """Parsed evaluation result with metadata."""

    marks_awarded: int
    criterion_scores: list[dict[str, Any]]
    overall_feedback: str
    strengths: list[str]
    improvements: list[str]
    raw_output: str = ""
    parse_success: bool = True


@dataclass
class AnswerSheetEvaluator:
    """Production inference engine for answer sheet evaluation.

    Loads the base Qwen2.5-3B-Instruct model in 4-bit quantization
    and applies the trained LoRA adapter on top.

    Parameters
    ----------
    adapter_path : str | Path
        Path to the saved LoRA adapter directory.
    model_id : str | None
        Override the base model ID (defaults to config).
    max_new_tokens : int
        Maximum tokens to generate for the evaluation.
    temperature : float
        Sampling temperature (lower = more deterministic).
    top_p : float
        Nucleus sampling threshold.
    """

    adapter_path: str | Path
    model_id: str | None = None
    max_new_tokens: int = 512
    temperature: float = 0.1
    top_p: float = 0.9
    _model: Any = field(default=None, init=False, repr=False)
    _tokenizer: PreTrainedTokenizerBase | None = field(default=None, init=False, repr=False)

    def _ensure_loaded(self):
        """Lazy-load model and tokenizer on first use."""
        if self._model is None:
            kwargs = {"adapter_path": self.adapter_path}
            if self.model_id:
                kwargs["model_id"] = self.model_id
            self._model, self._tokenizer = load_for_inference(**kwargs)

    def _build_messages(
        self,
        subject: str,
        question: str,
        maximum_marks: int,
        rubric_criteria: list[dict],
        reference_answer: str,
        student_answer: str,
    ) -> list[dict[str, str]]:
        """Build the chat messages for a single evaluation request."""
        sample = {
            "subject": subject,
            "question": question,
            "maximum_marks": maximum_marks,
            "rubric": {"criteria": rubric_criteria},
            "reference_answer": reference_answer,
            "student_answer": student_answer,
        }
        return [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": _build_user_prompt(sample)},
        ]

    def _generate(self, messages: list[dict[str, str]]) -> str:
        """Generate a response from the model."""
        self._ensure_loaded()

        text = self._tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,     # Add the assistant prefix for generation
        )

        inputs = self._tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=1024,
        ).to(self._model.device)

        with torch.no_grad():
            outputs = self._model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.temperature > 0,
                pad_token_id=self._tokenizer.pad_token_id,
                eos_token_id=self._tokenizer.eos_token_id,
                repetition_penalty=1.1,
            )

        # Decode only the newly generated tokens
        generated_ids = outputs[0][inputs["input_ids"].shape[1]:]
        response = self._tokenizer.decode(generated_ids, skip_special_tokens=True)
        return response.strip()

    @staticmethod
    def _parse_json_response(raw_output: str) -> dict | None:
        """Extract and parse JSON from model output, handling common issues."""
        # Try direct parse first
        try:
            return json.loads(raw_output)
        except json.JSONDecodeError:
            pass

        # Try to extract JSON from markdown code blocks
        json_match = re.search(r"```(?:json)?\s*\n?(.*?)\n?```", raw_output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                pass

        # Try to find a JSON object in the output
        brace_match = re.search(r"\{.*\}", raw_output, re.DOTALL)
        if brace_match:
            try:
                return json.loads(brace_match.group(0))
            except json.JSONDecodeError:
                pass

        return None

    def evaluate(
        self,
        subject: str,
        question: str,
        maximum_marks: int,
        rubric_criteria: list[dict],
        reference_answer: str,
        student_answer: str,
        retries: int = 2,
    ) -> EvaluationResult:
        """Evaluate a single student answer.

        Parameters
        ----------
        subject : str
            Subject name.
        question : str
            The question text.
        maximum_marks : int
            Maximum marks for the question.
        rubric_criteria : list[dict]
            List of {"name": str, "marks": int, "description": str}.
        reference_answer : str
            The model/reference answer.
        student_answer : str
            The student's answer to evaluate.
        retries : int
            Number of retries if JSON parsing fails.

        Returns
        -------
        EvaluationResult
            Parsed evaluation with marks, feedback, and strengths/improvements.
        """
        messages = self._build_messages(
            subject, question, maximum_marks,
            rubric_criteria, reference_answer, student_answer,
        )

        last_raw = ""
        for attempt in range(1 + retries):
            if attempt > 0:
                logger.warning("Retry %d/%d — previous output was not valid JSON", attempt, retries)

            raw_output = self._generate(messages)
            last_raw = raw_output
            parsed = self._parse_json_response(raw_output)

            if parsed is not None:
                return EvaluationResult(
                    marks_awarded=parsed.get("marks_awarded", 0),
                    criterion_scores=parsed.get("criterion_scores", []),
                    overall_feedback=parsed.get("overall_feedback", ""),
                    strengths=parsed.get("strengths", []),
                    improvements=parsed.get("improvements", []),
                    raw_output=raw_output,
                    parse_success=True,
                )

        # All retries exhausted — return raw output
        logger.error("Failed to parse JSON after %d attempts", 1 + retries)
        return EvaluationResult(
            marks_awarded=0,
            criterion_scores=[],
            overall_feedback="ERROR: Could not parse model output as JSON.",
            strengths=[],
            improvements=[],
            raw_output=last_raw,
            parse_success=False,
        )

    def evaluate_batch(
        self,
        samples: list[dict],
    ) -> list[EvaluationResult]:
        """Evaluate multiple samples sequentially.

        Parameters
        ----------
        samples : list[dict]
            Each dict must have: subject, question, maximum_marks,
            rubric_criteria, reference_answer, student_answer.

        Returns
        -------
        list[EvaluationResult]
        """
        results = []
        for i, sample in enumerate(samples):
            logger.info("Evaluating sample %d / %d", i + 1, len(samples))
            result = self.evaluate(**sample)
            results.append(result)
        return results
