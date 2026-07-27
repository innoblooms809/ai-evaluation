"""
Pydantic schemas for strict validation of raw dataset samples.

Every JSON sample is validated through these models *before* it enters
the training pipeline, catching malformed data early.
"""

from __future__ import annotations

from pydantic import BaseModel, Field, model_validator


# ──────────────────────────────────────────────
# Rubric
# ──────────────────────────────────────────────
class RubricCriterion(BaseModel):
    """A single grading criterion inside a rubric."""

    name: str = Field(..., min_length=1, description="Criterion name, e.g. 'Core concept'")
    marks: int = Field(..., ge=0, description="Maximum marks for this criterion")
    description: str = Field(..., min_length=1, description="What this criterion measures")


class Rubric(BaseModel):
    """The full rubric for a question."""

    criteria: list[RubricCriterion] = Field(..., min_length=1)

    @property
    def total_marks(self) -> int:
        return sum(c.marks for c in self.criteria)

    @property
    def criterion_names(self) -> set[str]:
        return {c.name for c in self.criteria}

    def max_marks_for(self, criterion_name: str) -> int:
        """Return the max marks for a named criterion."""
        for c in self.criteria:
            if c.name == criterion_name:
                return c.marks
        raise ValueError(f"Unknown criterion: {criterion_name!r}")


# ──────────────────────────────────────────────
# Evaluation (ground-truth label)
# ──────────────────────────────────────────────
class CriterionScore(BaseModel):
    """Score assigned to a single rubric criterion."""

    criterion: str = Field(..., min_length=1)
    score: int = Field(..., ge=0)
    reason: str = Field(..., min_length=1)


class Evaluation(BaseModel):
    """The structured evaluation output the model should learn to produce."""

    marks_awarded: int = Field(..., ge=0)
    criterion_scores: list[CriterionScore] = Field(..., min_length=1)
    overall_feedback: str = Field(..., min_length=1)
    strengths: list[str] = Field(default_factory=list)
    improvements: list[str] = Field(default_factory=list)


# ──────────────────────────────────────────────
# Full sample
# ──────────────────────────────────────────────
class AnswerSheetSample(BaseModel):
    """One complete sample from the dataset."""

    subject: str = Field(..., min_length=1)
    question: str = Field(..., min_length=1)
    maximum_marks: int = Field(..., ge=1)
    rubric: Rubric
    reference_answer: str = Field(..., min_length=1)
    student_answer: str = Field(..., min_length=1)
    evaluation: Evaluation

    # ── Cross-field validation ──────────────────

    @model_validator(mode="after")
    def validate_marks_consistency(self) -> "AnswerSheetSample":
        """Ensure awarded marks ≤ maximum_marks and per-criterion totals match."""
        errors: list[str] = []

        # Total awarded must not exceed maximum
        if self.evaluation.marks_awarded > self.maximum_marks:
            errors.append(
                f"marks_awarded ({self.evaluation.marks_awarded}) exceeds "
                f"maximum_marks ({self.maximum_marks})"
            )

        # Sum of criterion scores should equal marks_awarded
        criterion_sum = sum(cs.score for cs in self.evaluation.criterion_scores)
        if criterion_sum != self.evaluation.marks_awarded:
            errors.append(
                f"Sum of criterion scores ({criterion_sum}) != "
                f"marks_awarded ({self.evaluation.marks_awarded})"
            )

        # Each criterion score must not exceed its rubric maximum
        rubric_names = self.rubric.criterion_names
        for cs in self.evaluation.criterion_scores:
            if cs.criterion not in rubric_names:
                errors.append(f"Criterion '{cs.criterion}' not found in rubric")
                continue
            max_for_criterion = self.rubric.max_marks_for(cs.criterion)
            if cs.score > max_for_criterion:
                errors.append(
                    f"Criterion '{cs.criterion}' score ({cs.score}) exceeds "
                    f"rubric max ({max_for_criterion})"
                )

        if errors:
            raise ValueError(
                "Marks consistency validation failed:\n  • " + "\n  • ".join(errors)
            )
        return self
