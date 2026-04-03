"""Precision, recall, F1 and accuracy metrics for gap detection evaluation.

Compares model-predicted assessments against ground truth annotations.

Usage:
    from evaluation.metrics import compute_metrics
    metrics = compute_metrics(predicted_assessments, ground_truth_assessments)
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

# Match degree ordering for partial credit
_DEGREE_ORDER = {"FULLY_MET": 2, "PARTIALLY_MET": 1, "NOT_MET": 0}


@dataclass
class EvaluationMetrics:
    """Metrics for one test case."""

    test_id: str
    exact_accuracy: float        # % of requirements with exact match
    soft_accuracy: float         # % with exact or adjacent match (PARTIAL ↔ MET/NOT_MET)
    gap_precision: float         # TP / (TP + FP) for NOT_MET detection
    gap_recall: float            # TP / (TP + FN) for NOT_MET detection
    gap_f1: float                # Harmonic mean of precision and recall
    verdict_correct: bool        # Whether overall verdict matches
    requirement_details: list[dict]  # Per-requirement breakdown

    def __str__(self) -> str:
        return (
            f"Test {self.test_id}: "
            f"Exact={self.exact_accuracy:.1%} Soft={self.soft_accuracy:.1%} "
            f"P={self.gap_precision:.2f} R={self.gap_recall:.2f} F1={self.gap_f1:.2f} "
            f"Verdict={'✓' if self.verdict_correct else '✗'}"
        )


def compute_metrics(
    predicted: dict[str, str],
    ground_truth: dict[str, str],
    predicted_verdict: str,
    expected_verdict: str,
    test_id: str = "unknown",
) -> EvaluationMetrics:
    """Compute evaluation metrics for one test case.

    Args:
        predicted: Dict mapping requirement ID → predicted match degree.
        ground_truth: Dict mapping requirement ID → ground truth match degree.
        predicted_verdict: Predicted overall verdict string.
        expected_verdict: Ground truth overall verdict string.
        test_id: Identifier for this test case.

    Returns:
        EvaluationMetrics dataclass with all computed metrics.
    """
    all_req_ids = sorted(set(predicted.keys()) | set(ground_truth.keys()))
    details: list[dict] = []

    exact_matches = 0
    soft_matches = 0
    true_positives = 0   # Correctly predicted NOT_MET
    false_positives = 0  # Predicted NOT_MET but was MET/PARTIAL
    false_negatives = 0  # Predicted MET/PARTIAL but was NOT_MET

    for req_id in all_req_ids:
        pred = predicted.get(req_id, "NOT_MET")
        truth = ground_truth.get(req_id, "NOT_MET")

        exact = pred == truth
        adjacent = abs(_DEGREE_ORDER.get(pred, 0) - _DEGREE_ORDER.get(truth, 0)) <= 1
        soft = exact or adjacent

        if exact:
            exact_matches += 1
        if soft:
            soft_matches += 1

        # NOT_MET detection (binary classification)
        pred_not_met = pred == "NOT_MET"
        truth_not_met = truth == "NOT_MET"
        if pred_not_met and truth_not_met:
            true_positives += 1
        elif pred_not_met and not truth_not_met:
            false_positives += 1
        elif not pred_not_met and truth_not_met:
            false_negatives += 1

        details.append({
            "requirement_id": req_id,
            "predicted": pred,
            "ground_truth": truth,
            "exact_match": exact,
            "soft_match": soft,
        })

    n = len(all_req_ids)
    exact_accuracy = exact_matches / n if n > 0 else 0.0
    soft_accuracy = soft_matches / n if n > 0 else 0.0

    precision_denom = true_positives + false_positives
    recall_denom = true_positives + false_negatives
    precision = true_positives / precision_denom if precision_denom > 0 else 0.0
    recall = true_positives / recall_denom if recall_denom > 0 else 0.0
    f1 = (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

    return EvaluationMetrics(
        test_id=test_id,
        exact_accuracy=exact_accuracy,
        soft_accuracy=soft_accuracy,
        gap_precision=precision,
        gap_recall=recall,
        gap_f1=f1,
        verdict_correct=(predicted_verdict == expected_verdict),
        requirement_details=details,
    )


def aggregate_metrics(all_metrics: list[EvaluationMetrics]) -> dict:
    """Aggregate metrics across all test cases.

    Args:
        all_metrics: List of EvaluationMetrics from individual test cases.

    Returns:
        Dict with mean values for all numeric metrics.
    """
    if not all_metrics:
        return {}

    n = len(all_metrics)
    return {
        "n_test_cases": n,
        "mean_exact_accuracy": sum(m.exact_accuracy for m in all_metrics) / n,
        "mean_soft_accuracy": sum(m.soft_accuracy for m in all_metrics) / n,
        "mean_gap_precision": sum(m.gap_precision for m in all_metrics) / n,
        "mean_gap_recall": sum(m.gap_recall for m in all_metrics) / n,
        "mean_gap_f1": sum(m.gap_f1 for m in all_metrics) / n,
        "verdict_accuracy": sum(1 for m in all_metrics if m.verdict_correct) / n,
    }
