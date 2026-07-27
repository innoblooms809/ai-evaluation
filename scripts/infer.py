"""
Inference entrypoint: evaluate student answers using the fine-tuned model.

Modes:
  1. Interactive — type questions in the terminal
  2. Single — evaluate one answer via CLI arguments
  3. Batch — evaluate all samples from a JSON file

Usage:
    python scripts/infer.py --adapter-path outputs/checkpoints/final --interactive
    python scripts/infer.py --adapter-path outputs/checkpoints/final --input-file data/raw/test.json
    python scripts/infer.py --adapter-path outputs/checkpoints/final \\
        --subject "Physics" \\
        --question "What is gravity?" \\
        --max-marks 5 \\
        --reference "Gravity is the force of attraction between masses." \\
        --student "Gravity pulls things down."
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

# Ensure stdout/stderr supports UTF-8 on Windows consoles to prevent encoding errors
if hasattr(sys.stdout, "reconfigure"):
    try:
        sys.stdout.reconfigure(encoding="utf-8")
    except Exception:
        pass
if hasattr(sys.stderr, "reconfigure"):
    try:
        sys.stderr.reconfigure(encoding="utf-8")
    except Exception:
        pass

# Add project root to path
PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from inference.evaluator import AnswerSheetEvaluator

logger = logging.getLogger(__name__)


def print_result(result, sample_info: str = ""):
    """Pretty-print an evaluation result."""
    if sample_info:
        print(f"\n{'-' * 50}")
        print(f"  {sample_info}")
        print(f"{'-' * 50}")

    if result.parse_success:
        output = {
            "marks_awarded": result.marks_awarded,
            "criterion_scores": result.criterion_scores,
            "overall_feedback": result.overall_feedback,
            "strengths": result.strengths,
            "improvements": result.improvements,
        }
        print(json.dumps(output, indent=2, ensure_ascii=False))
    else:
        print(f"[WARNING] Failed to parse JSON output. Raw model response:")
        print(result.raw_output)


def interactive_mode(evaluator: AnswerSheetEvaluator):
    """Interactive evaluation mode -- enter answers in the terminal."""
    print("\n+--------------------------------------------------+")
    print("|    Answer Sheet Evaluator -- Interactive Mode    |")
    print("|    Type 'quit' or 'exit' to stop                 |")
    print("+--------------------------------------------------+\n")

    # Default rubric for convenience
    default_rubric = [
        {"name": "Core concept", "marks": 2, "description": "States the correct definition or principle."},
        {"name": "Key points", "marks": 2, "description": "Includes important supporting details."},
        {"name": "Clarity", "marks": 1, "description": "Clear and concise explanation."},
    ]

    while True:
        try:
            print("\n" + "=" * 50)
            subject = input("Subject: ").strip()
            if subject.lower() in ("quit", "exit"):
                break

            question = input("Question: ").strip()
            if question.lower() in ("quit", "exit"):
                break

            max_marks_str = input("Maximum marks (default: 5): ").strip()
            max_marks = int(max_marks_str) if max_marks_str else 5

            reference = input("Reference answer: ").strip()
            student = input("Student answer: ").strip()

            use_default = input("Use default rubric? (Y/n): ").strip().lower()
            if use_default in ("n", "no"):
                print("Enter rubric criteria (JSON list):")
                rubric_json = input().strip()
                rubric = json.loads(rubric_json)
            else:
                rubric = default_rubric

            print("\nEvaluating…")
            result = evaluator.evaluate(
                subject=subject,
                question=question,
                maximum_marks=max_marks,
                rubric_criteria=rubric,
                reference_answer=reference,
                student_answer=student,
            )
            print_result(result)

        except KeyboardInterrupt:
            print("\n\nExiting…")
            break
        except json.JSONDecodeError as e:
            print(f"Invalid JSON: {e}")
        except Exception as e:
            print(f"Error: {e}")


def batch_mode(evaluator: AnswerSheetEvaluator, input_file: str, output_file: str | None):
    """Evaluate all samples from a JSON file."""
    logger.info("Loading samples from %s", input_file)
    with open(input_file, "r", encoding="utf-8") as f:
        raw_samples = json.load(f)

    results = []
    for i, sample in enumerate(raw_samples):
        print(f"\rEvaluating {i + 1} / {len(raw_samples)}...", end="", flush=True)

        rubric_criteria = sample.get("rubric", {}).get("criteria", [])
        result = evaluator.evaluate(
            subject=sample["subject"],
            question=sample["question"],
            maximum_marks=sample["maximum_marks"],
            rubric_criteria=rubric_criteria,
            reference_answer=sample["reference_answer"],
            student_answer=sample["student_answer"],
        )

        result_dict = {
            "question": sample["question"],
            "student_answer": sample["student_answer"],
            "predicted_evaluation": {
                "marks_awarded": result.marks_awarded,
                "criterion_scores": result.criterion_scores,
                "overall_feedback": result.overall_feedback,
                "strengths": result.strengths,
                "improvements": result.improvements,
            },
            "parse_success": result.parse_success,
        }

        # Add ground truth if available
        if "evaluation" in sample:
            result_dict["ground_truth_marks"] = sample["evaluation"]["marks_awarded"]

        results.append(result_dict)
        print_result(result, f"Q: {sample['question']}")

    print(f"\n\nProcessed {len(results)} samples")

    if output_file:
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        print(f"Results saved to {output_file}")

    # Print summary statistics
    successes = sum(1 for r in results if r["parse_success"])
    print(f"\nParse success rate: {successes}/{len(results)} ({100 * successes / len(results):.0f}%)")

    if any("ground_truth_marks" in r for r in results):
        correct = sum(
            1 for r in results
            if r.get("ground_truth_marks") == r["predicted_evaluation"]["marks_awarded"]
        )
        print(f"Exact marks match: {correct}/{len(results)} ({100 * correct / len(results):.0f}%)")


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate student answers using fine-tuned Qwen2.5-3B-Instruct"
    )
    parser.add_argument(
        "--adapter-path", type=str, required=True,
        help="Path to the saved LoRA adapter directory",
    )
    parser.add_argument(
        "--model-id", type=str, default=None,
        help="Override base model ID",
    )
    parser.add_argument(
        "--interactive", action="store_true",
        help="Run in interactive mode",
    )
    parser.add_argument(
        "--input-file", type=str, default=None,
        help="JSON file with samples to evaluate (batch mode)",
    )
    parser.add_argument(
        "--output-file", type=str, default=None,
        help="Save batch results to this file",
    )
    # Single evaluation arguments
    parser.add_argument("--subject", type=str, help="Subject name")
    parser.add_argument("--question", type=str, help="Question text")
    parser.add_argument("--max-marks", type=int, default=5, help="Maximum marks")
    parser.add_argument("--reference", type=str, help="Reference answer")
    parser.add_argument("--student", type=str, help="Student answer to evaluate")
    parser.add_argument(
        "--temperature", type=float, default=0.1,
        help="Generation temperature (default: 0.1)",
    )
    parser.add_argument(
        "--max-new-tokens", type=int, default=512,
        help="Maximum new tokens to generate (default: 512)",
    )

    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)-8s | %(message)s",
        datefmt="%Y-%m-%d %H:%M:%S",
    )

    evaluator = AnswerSheetEvaluator(
        adapter_path=args.adapter_path,
        model_id=args.model_id,
        temperature=args.temperature,
        max_new_tokens=args.max_new_tokens,
    )

    if args.interactive:
        interactive_mode(evaluator)
    elif args.input_file:
        batch_mode(evaluator, args.input_file, args.output_file)
    elif args.question and args.reference and args.student:
        default_rubric = [
            {"name": "Core concept", "marks": 2, "description": "States the correct definition."},
            {"name": "Key points", "marks": 2, "description": "Includes supporting details."},
            {"name": "Clarity", "marks": 1, "description": "Clear explanation."},
        ]
        result = evaluator.evaluate(
            subject=args.subject or "General",
            question=args.question,
            maximum_marks=args.max_marks,
            rubric_criteria=default_rubric,
            reference_answer=args.reference,
            student_answer=args.student,
        )
        print_result(result)
    else:
        parser.print_help()
        print("\nError: Provide --interactive, --input-file, or --question/--reference/--student")
        sys.exit(1)


if __name__ == "__main__":
    main()
