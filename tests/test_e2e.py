"""End-to-end tests: letter in → gap report out.

These tests require the full stack (Neo4j + PostgreSQL seeded, FastAPI running).
Mark with @pytest.mark.integration and run with:
    pytest tests/test_e2e.py -m integration
"""

from __future__ import annotations

import json
from pathlib import Path

import httpx
import pytest

DATA_DIR = Path(__file__).parent.parent / "data" / "test_letters"
API_BASE_URL = "http://localhost:8000"
REQUEST_TIMEOUT = 300


# ── Health check ──────────────────────────────────────────────────────────────


@pytest.mark.integration
def test_api_health():
    """API should respond to health check."""
    with httpx.Client(timeout=10) as client:
        resp = client.get(f"{API_BASE_URL}/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


# ── Analysis endpoint tests ───────────────────────────────────────────────────


@pytest.mark.integration
def test_analyze_sample_letter_returns_200():
    """POST /analyze/text should return 200 for the sample letter."""
    letter_text = (DATA_DIR / "sample_letter_with_gaps.txt").read_text(encoding="utf-8")
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        resp = client.post(
            f"{API_BASE_URL}/analyze/text",
            json={"letter_text": letter_text},
        )
    assert resp.status_code == 200
    data = resp.json()
    assert "session_id" in data
    assert "gap_report" in data


@pytest.mark.integration
def test_analyze_sample_letter_verdict():
    """Sample letter should be INVALID_MAJOR_REVISION (has critical gaps)."""
    letter_text = (DATA_DIR / "sample_letter_with_gaps.txt").read_text(encoding="utf-8")
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        resp = client.post(
            f"{API_BASE_URL}/analyze/text",
            json={"letter_text": letter_text},
        )
    data = resp.json()
    verdict = data["gap_report"]["verdict"]
    assert verdict == "INVALID_MAJOR_REVISION"


@pytest.mark.integration
def test_analyze_returns_15_assessments():
    """Gap report should always contain exactly 15 requirement assessments."""
    letter_text = (DATA_DIR / "sample_letter_with_gaps.txt").read_text(encoding="utf-8")
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        resp = client.post(
            f"{API_BASE_URL}/analyze/text",
            json={"letter_text": letter_text},
        )
    assessments = resp.json()["gap_report"]["assessments"]
    assert len(assessments) == 15


@pytest.mark.integration
def test_analyze_req01_and_req02_met_for_sample():
    """REQ-01 (purpose stated) and REQ-02 (PI identified) should be FULLY_MET."""
    letter_text = (DATA_DIR / "sample_letter_with_gaps.txt").read_text(encoding="utf-8")
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        resp = client.post(
            f"{API_BASE_URL}/analyze/text",
            json={"letter_text": letter_text},
        )
    assessments = {
        a["requirement_id"]: a["match_degree"]
        for a in resp.json()["gap_report"]["assessments"]
    }
    assert assessments.get("REQ-01") == "FULLY_MET"
    assert assessments.get("REQ-02") == "FULLY_MET"


@pytest.mark.integration
def test_analyze_req06_not_met_for_sample():
    """REQ-06 (breach notification) should be NOT_MET for the sample letter."""
    letter_text = (DATA_DIR / "sample_letter_with_gaps.txt").read_text(encoding="utf-8")
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        resp = client.post(
            f"{API_BASE_URL}/analyze/text",
            json={"letter_text": letter_text},
        )
    assessments = {
        a["requirement_id"]: a["match_degree"]
        for a in resp.json()["gap_report"]["assessments"]
    }
    assert assessments.get("REQ-06") == "NOT_MET"


@pytest.mark.integration
def test_session_retrieval():
    """GET /session/{id} should return the stored analysis."""
    letter_text = (DATA_DIR / "minimal_sloppy_letter.txt").read_text(encoding="utf-8")
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        post_resp = client.post(
            f"{API_BASE_URL}/analyze/text",
            json={"letter_text": letter_text},
        )
        session_id = post_resp.json()["session_id"]

        get_resp = client.get(f"{API_BASE_URL}/session/{session_id}")

    assert get_resp.status_code == 200
    assert get_resp.json()["session_id"] == session_id


@pytest.mark.integration
def test_session_not_found():
    """GET /session/{unknown} should return 404."""
    with httpx.Client(timeout=10) as client:
        resp = client.get(f"{API_BASE_URL}/session/nonexistent-uuid-0000")
    assert resp.status_code == 404


@pytest.mark.integration
def test_analyze_empty_letter_rejected():
    """POST /analyze/text should reject empty letter text."""
    with httpx.Client(timeout=10) as client:
        resp = client.post(
            f"{API_BASE_URL}/analyze/text",
            json={"letter_text": "hi"},
        )
    assert resp.status_code == 422  # Pydantic validation (min_length=50)


@pytest.mark.integration
def test_followup_requires_valid_session():
    """POST /followup should return 404 for unknown session."""
    with httpx.Client(timeout=10) as client:
        resp = client.post(
            f"{API_BASE_URL}/followup",
            json={"session_id": "nonexistent", "message": "What is REQ-06?"},
        )
    assert resp.status_code == 404


@pytest.mark.integration
def test_fully_compliant_letter_verdict():
    """Fully compliant letter should ideally produce VALID verdict."""
    letter_text = (DATA_DIR / "fully_compliant_letter.txt").read_text(encoding="utf-8")
    with httpx.Client(timeout=REQUEST_TIMEOUT) as client:
        resp = client.post(
            f"{API_BASE_URL}/analyze/text",
            json={"letter_text": letter_text},
        )
    data = resp.json()
    verdict = data["gap_report"]["verdict"]
    # Should be VALID or at most INVALID_FIXABLE (no critical gaps)
    assert verdict in ("VALID", "INVALID_FIXABLE")
