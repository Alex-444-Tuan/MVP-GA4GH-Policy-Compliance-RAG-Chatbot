"""Run evaluation suite against all test letters and print metrics.

Usage:
    python scripts/run_evaluation.py

Requires the full stack to be running:
    - Neo4j + PostgreSQL (seeded)
    - FastAPI server on localhost:8000
"""

from __future__ import annotations

import asyncio
import json
import logging
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).parent.parent))

from evaluation.metrics import EvaluationMetrics, aggregate_metrics, compute_metrics
from src.config import configure_logging

configure_logging()
logger = logging.getLogger(__name__)

DATA_DIR = Path(__file__).parent.parent / "data"
EVAL_DIR = Path(__file__).parent.parent / "evaluation"
API_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 300


def load_ground_truth() -> dict:
    """Load the annotated ground truth from evaluation/ground_truth.json."""
    path = EVAL_DIR / "ground_truth.json"
    return json.loads(path.read_text(encoding="utf-8"))


def analyze_letter(letter_text: str) -> dict | None:
    """Submit a letter to the API and return the gap report."""
    try:
        with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
            response = client.post(
                f"{API_BASE_URL}/analyze/text",
                json={"letter_text": letter_text, "rrf_lexical_weight": 0.4},
            )
        if response.status_code == 200:
            return response.json()
        else:
            logger.error("API returned %d: %s", response.status_code, response.text[:200])
            return None
    except httpx.ConnectError:
        logger.error("Cannot connect to API at %s. Is the server running?", API_BASE_URL)
        return None


def run_test_case(test_case: dict) -> EvaluationMetrics | None:
    """Run one test case and return its metrics."""
    letter_file = Path(__file__).parent.parent / test_case["letter_file"]
    if not letter_file.exists():
        logger.error("Test letter not found: %s", letter_file)
        return None

    letter_text = letter_file.read_text(encoding="utf-8")
    logger.info("Running test case %s: %s", test_case["id"], test_case["description"])

    result = analyze_letter(letter_text)
    if result is None:
        logger.error("Analysis failed for test case %s", test_case["id"])
        return None

    gap_report = result.get("gap_report", {})
    assessments = gap_report.get("assessments", [])
    predicted = {a["requirement_id"]: a["match_degree"] for a in assessments}
    predicted_verdict = gap_report.get("verdict", "UNKNOWN")

    ground_truth = test_case["expected_assessments"]
    expected_verdict = test_case["expected_verdict"]

    metrics = compute_metrics(
        predicted=predicted,
        ground_truth=ground_truth,
        predicted_verdict=predicted_verdict,
        expected_verdict=expected_verdict,
        test_id=test_case["id"],
    )
    logger.info("%s", metrics)
    return metrics


def main() -> None:
    """Entry point for the evaluation runner."""
    logger.info("=== GA4GH Compliance Evaluation ===")

    # Verify API is reachable
    try:
        with httpx.Client(timeout=10) as client:
            health = client.get(f"{API_BASE_URL}/health")
        if health.status_code != 200:
            logger.error("API health check failed. Ensure the server is running.")
            sys.exit(1)
        logger.info("API is healthy.")
    except httpx.ConnectError:
        logger.error("Cannot connect to API at %s.", API_BASE_URL)
        sys.exit(1)

    ground_truth_data = load_ground_truth()
    test_cases = ground_truth_data["test_cases"]
    logger.info("Running %d test cases...", len(test_cases))

    all_metrics: list[EvaluationMetrics] = []
    for test_case in test_cases:
        metrics = run_test_case(test_case)
        if metrics:
            all_metrics.append(metrics)

    if not all_metrics:
        logger.error("No test cases completed successfully.")
        sys.exit(1)

    # Print aggregated results
    agg = aggregate_metrics(all_metrics)
    print("\n" + "=" * 60)
    print("EVALUATION RESULTS")
    print("=" * 60)
    for m in all_metrics:
        print(f"\n{m}")
        for detail in m.requirement_details:
            status = "✓" if detail["exact_match"] else ("~" if detail["soft_match"] else "✗")
            print(f"  {status} {detail['requirement_id']}: "
                  f"pred={detail['predicted']} truth={detail['ground_truth']}")

    print("\n" + "-" * 60)
    print("AGGREGATE METRICS")
    print("-" * 60)
    for key, value in agg.items():
        if isinstance(value, float):
            print(f"  {key}: {value:.3f}")
        else:
            print(f"  {key}: {value}")
    print("=" * 60)


if __name__ == "__main__":
    main()
