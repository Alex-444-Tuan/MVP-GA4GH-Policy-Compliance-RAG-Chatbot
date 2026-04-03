"""Unit tests for the gap detection and verdict logic."""

from __future__ import annotations

import pytest

from src.analysis.gap_detector import aggregate_assessments, compute_verdict
from src.models.schemas import (
    LetterChunk,
    MatchDegree,
    RequirementAssessment,
    Severity,
    Verdict,
)


def _make_assessment(
    req_id: str,
    match_degree: MatchDegree,
    severity: Severity = Severity.MAJOR,
) -> RequirementAssessment:
    """Helper to create a minimal RequirementAssessment."""
    return RequirementAssessment(
        requirement_id=req_id,
        description=f"Description for {req_id}",
        match_degree=match_degree,
        evidence_from_letter="Some evidence",
        evidence_from_policy="Policy text",
        reasoning="Test reasoning",
        severity=severity,
    )


# ── Verdict computation tests ─────────────────────────────────────────────────


def test_verdict_valid_all_met():
    """All FULLY_MET → VALID."""
    assessments = [
        _make_assessment(f"REQ-{i:02d}", MatchDegree.FULLY_MET)
        for i in range(1, 16)
    ]
    assert compute_verdict(assessments) == Verdict.VALID


def test_verdict_major_revision_critical_not_met():
    """Any CRITICAL + NOT_MET → INVALID_MAJOR_REVISION."""
    assessments = [
        _make_assessment("REQ-01", MatchDegree.NOT_MET, Severity.CRITICAL),
        _make_assessment("REQ-02", MatchDegree.FULLY_MET, Severity.MAJOR),
    ]
    assert compute_verdict(assessments) == Verdict.INVALID_MAJOR_REVISION


def test_verdict_fixable_only_minor_not_met():
    """NOT_MET but all are MINOR → INVALID_FIXABLE."""
    assessments = [
        _make_assessment("REQ-13", MatchDegree.NOT_MET, Severity.MINOR),
        _make_assessment("REQ-14", MatchDegree.NOT_MET, Severity.MINOR),
        _make_assessment("REQ-01", MatchDegree.FULLY_MET, Severity.CRITICAL),
    ]
    assert compute_verdict(assessments) == Verdict.INVALID_FIXABLE


def test_verdict_fixable_major_not_met():
    """NOT_MET with MAJOR severity → INVALID_FIXABLE (not MAJOR_REVISION)."""
    assessments = [
        _make_assessment("REQ-09", MatchDegree.NOT_MET, Severity.MAJOR),
        _make_assessment("REQ-01", MatchDegree.FULLY_MET, Severity.CRITICAL),
        _make_assessment("REQ-05", MatchDegree.FULLY_MET, Severity.CRITICAL),
    ]
    assert compute_verdict(assessments) == Verdict.INVALID_FIXABLE


def test_verdict_partially_met_does_not_trigger_major_revision():
    """PARTIALLY_MET CRITICAL does not trigger INVALID_MAJOR_REVISION (only NOT_MET does)."""
    assessments = [
        _make_assessment("REQ-05", MatchDegree.PARTIALLY_MET, Severity.CRITICAL),
    ]
    # PARTIALLY_MET is not NOT_MET — should be INVALID_FIXABLE or VALID
    verdict = compute_verdict(assessments)
    assert verdict in (Verdict.VALID, Verdict.INVALID_FIXABLE)


# ── Aggregation tests ─────────────────────────────────────────────────────────


def test_aggregate_takes_best_match():
    """Aggregation should take the best match_degree across chunks."""
    req_meta = {
        "REQ-01": ("Research purpose", Severity.CRITICAL),
    }
    chunk1 = [_make_assessment("REQ-01", MatchDegree.NOT_MET)]
    chunk2 = [_make_assessment("REQ-01", MatchDegree.PARTIALLY_MET)]
    chunk3 = [_make_assessment("REQ-01", MatchDegree.FULLY_MET)]

    result = aggregate_assessments([chunk1, chunk2, chunk3], req_meta)
    assert len(result) == 1
    assert result[0].match_degree == MatchDegree.FULLY_MET


def test_aggregate_defaults_to_not_met_when_unassessed():
    """Requirements not seen in any chunk should default to NOT_MET."""
    req_meta = {
        "REQ-01": ("Research purpose", Severity.CRITICAL),
        "REQ-15": ("Governing law", Severity.MINOR),
    }
    # Only REQ-01 assessed
    chunk1 = [_make_assessment("REQ-01", MatchDegree.FULLY_MET)]

    result = aggregate_assessments([chunk1], req_meta)
    assert len(result) == 2

    req_15_result = next(r for r in result if r.requirement_id == "REQ-15")
    assert req_15_result.match_degree == MatchDegree.NOT_MET


def test_aggregate_output_sorted_by_req_id():
    """Aggregated results should be sorted by requirement ID."""
    req_meta = {
        "REQ-03": ("Institutional affiliation", Severity.MAJOR),
        "REQ-01": ("Research purpose", Severity.CRITICAL),
        "REQ-02": ("PI identification", Severity.CRITICAL),
    }
    chunks = [[
        _make_assessment("REQ-01", MatchDegree.FULLY_MET),
        _make_assessment("REQ-02", MatchDegree.FULLY_MET),
        _make_assessment("REQ-03", MatchDegree.NOT_MET),
    ]]
    result = aggregate_assessments(chunks, req_meta)
    ids = [r.requirement_id for r in result]
    assert ids == sorted(ids)


def test_aggregate_severity_from_canonical_source():
    """Aggregation should use severity from req_meta, not from LLM assessment."""
    req_meta = {
        "REQ-01": ("Research purpose", Severity.CRITICAL),
    }
    # LLM might return wrong severity — should be overridden
    assessment = _make_assessment("REQ-01", MatchDegree.NOT_MET, Severity.MINOR)
    result = aggregate_assessments([[assessment]], req_meta)
    assert result[0].severity == Severity.CRITICAL
