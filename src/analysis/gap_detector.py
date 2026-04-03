"""Per-requirement gap analysis and overall verdict computation.

Processes all retrieval results, calls the LLM for each chunk,
aggregates assessments across chunks (taking best match per requirement),
and computes the final verdict.

Verdict logic:
- VALID: zero NOT_MET across all requirements
- INVALID_FIXABLE: has NOT_MET but none are CRITICAL
- INVALID_MAJOR_REVISION: any CRITICAL requirement is NOT_MET
"""

from __future__ import annotations

import json
import logging
import time
from collections import defaultdict

import anthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.analysis.prompts import GAP_ANALYSIS_SYSTEM_PROMPT, GAP_ANALYSIS_USER_PROMPT
from src.config import get_settings
from src.models.schemas import (
    GapReport,
    LetterChunk,
    MatchDegree,
    RequirementAssessment,
    RetrievalResult,
    Severity,
    Verdict,
)

logger = logging.getLogger(__name__)

# Match degree priority for aggregation (higher = better)
_MATCH_PRIORITY: dict[MatchDegree, int] = {
    MatchDegree.FULLY_MET: 2,
    MatchDegree.PARTIALLY_MET: 1,
    MatchDegree.NOT_MET: 0,
}


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((anthropic.APIError, anthropic.APIConnectionError)),
)
def _call_gap_analysis_llm(
    letter_chunk: str,
    requirements_json: str,
    policy_context: str,
    client: anthropic.Anthropic,
) -> dict:
    """Call Claude Sonnet for gap analysis. Returns parsed JSON response dict.

    Args:
        letter_chunk: Text of the letter chunk being analyzed.
        requirements_json: JSON string of requirements to check.
        policy_context: Combined text of retrieved policy chunks.
        client: Anthropic API client.

    Returns:
        Parsed JSON dict with 'assessments' list.

    Raises:
        ValueError: If the LLM response cannot be parsed as valid JSON.
    """
    user_message = GAP_ANALYSIS_USER_PROMPT.format(
        letter_chunk=letter_chunk[:3000],
        requirements_json=requirements_json[:2000],
        policy_context=policy_context[:3000],
    )

    cfg = get_settings()
    t0 = time.perf_counter()
    message = client.messages.create(
        model=cfg.analysis_model,
        max_tokens=2048,
        timeout=cfg.llm_timeout_seconds,
        system=GAP_ANALYSIS_SYSTEM_PROMPT,
        messages=[{"role": "user", "content": user_message}],
    )
    elapsed = time.perf_counter() - t0
    logger.debug("Gap analysis LLM call: %.2fs", elapsed)

    raw = message.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse gap analysis JSON. Raw response: %r", raw[:500])
        raise ValueError(f"LLM returned invalid JSON: {e}") from e


def _build_requirements_json(retrieval_result: RetrievalResult) -> str:
    """Serialize requirements from a retrieval result to compact JSON."""
    reqs = [
        {
            "id": r.id,
            "description": r.description,
            "severity": r.severity.value,
            "daa_clause": r.daa_clause_category,
            "keywords": r.keywords[:5],
        }
        for r in retrieval_result.requirements
    ]
    return json.dumps(reqs, indent=None)


def _build_policy_context(retrieval_result: RetrievalResult) -> str:
    """Concatenate retrieved policy chunk texts for context."""
    parts = []
    for chunk in retrieval_result.policy_chunks[:3]:  # Top-3 to stay within token budget
        title = chunk.section_title or "Policy Section"
        parts.append(f"[{title}]\n{chunk.text[:800]}")
    return "\n\n---\n\n".join(parts)


def analyze_chunk(
    retrieval_result: RetrievalResult,
    client: anthropic.Anthropic,
) -> list[RequirementAssessment]:
    """Run gap analysis for one letter chunk against its retrieved requirements.

    Args:
        retrieval_result: Retrieval context for this chunk.
        client: Anthropic API client.

    Returns:
        List of RequirementAssessment for each requirement found in retrieval.
    """
    if not retrieval_result.requirements:
        logger.debug("No requirements retrieved for chunk %d — skipping LLM call.",
                     retrieval_result.letter_chunk.chunk_index)
        return []

    requirements_json = _build_requirements_json(retrieval_result)
    policy_context = _build_policy_context(retrieval_result)

    try:
        response = _call_gap_analysis_llm(
            letter_chunk=retrieval_result.letter_chunk.text,
            requirements_json=requirements_json,
            policy_context=policy_context,
            client=client,
        )
    except (ValueError, anthropic.APIError) as e:
        logger.error("Gap analysis failed for chunk %d: %s",
                     retrieval_result.letter_chunk.chunk_index, e)
        return []

    assessments: list[RequirementAssessment] = []
    req_severity_map = {r.id: r.severity for r in retrieval_result.requirements}

    for raw_assessment in response.get("assessments", []):
        try:
            req_id = raw_assessment.get("requirement_id", "")
            match_degree = MatchDegree(raw_assessment.get("match_degree", "NOT_MET"))
            severity = req_severity_map.get(req_id, Severity.MAJOR)

            assessments.append(RequirementAssessment(
                requirement_id=req_id,
                description=next(
                    (r.description for r in retrieval_result.requirements if r.id == req_id), ""
                ),
                match_degree=match_degree,
                evidence_from_letter=raw_assessment.get("evidence_from_letter", "No relevant content found"),
                evidence_from_policy=raw_assessment.get("evidence_from_policy", ""),
                reasoning=raw_assessment.get("reasoning", ""),
                severity=severity,
            ))
        except (ValueError, KeyError) as e:
            logger.warning("Skipping malformed assessment: %s — %r", e, raw_assessment)
            continue

    logger.debug("Chunk %d: %d assessments (%d FULLY_MET, %d PARTIALLY, %d NOT_MET)",
                 retrieval_result.letter_chunk.chunk_index,
                 len(assessments),
                 sum(1 for a in assessments if a.match_degree == MatchDegree.FULLY_MET),
                 sum(1 for a in assessments if a.match_degree == MatchDegree.PARTIALLY_MET),
                 sum(1 for a in assessments if a.match_degree == MatchDegree.NOT_MET))
    return assessments


def aggregate_assessments(
    all_assessments: list[list[RequirementAssessment]],
    all_requirements_meta: dict[str, tuple[str, Severity]],
) -> list[RequirementAssessment]:
    """Aggregate per-chunk assessments into one final assessment per requirement.

    For each requirement, take the best (highest-priority) match_degree
    seen across all chunks, along with its evidence and reasoning.

    Args:
        all_assessments: List of per-chunk assessment lists.
        all_requirements_meta: Dict mapping req_id → (description, severity)
                                for all 15 requirements.

    Returns:
        One RequirementAssessment per requirement (15 total).
    """
    # Group by requirement ID
    by_req: dict[str, list[RequirementAssessment]] = defaultdict(list)
    for chunk_assessments in all_assessments:
        for assessment in chunk_assessments:
            by_req[assessment.requirement_id].append(assessment)

    final: list[RequirementAssessment] = []

    # Ensure all 15 requirements are represented
    for req_id, (description, severity) in all_requirements_meta.items():
        assessments_for_req = by_req.get(req_id, [])

        if not assessments_for_req:
            # Not assessed — default to NOT_MET
            final.append(RequirementAssessment(
                requirement_id=req_id,
                description=description,
                match_degree=MatchDegree.NOT_MET,
                evidence_from_letter="No relevant content found in any section of the letter.",
                evidence_from_policy="",
                reasoning="This requirement was not addressed in any section of the submitted letter.",
                severity=severity,
            ))
            continue

        # Pick the assessment with the highest match priority
        best = max(assessments_for_req, key=lambda a: _MATCH_PRIORITY[a.match_degree])
        # Ensure description and severity are set from canonical source
        best.description = description
        best.severity = severity
        final.append(best)

    # Sort by requirement ID for consistent output
    final.sort(key=lambda a: a.requirement_id)
    return final


def compute_verdict(assessments: list[RequirementAssessment]) -> Verdict:
    """Compute the overall compliance verdict from aggregated assessments.

    Rules:
    - VALID: no NOT_MET requirements
    - INVALID_MAJOR_REVISION: any CRITICAL requirement is NOT_MET
    - INVALID_FIXABLE: has NOT_MET requirements, but none are CRITICAL

    Args:
        assessments: Aggregated list of 15 requirement assessments.

    Returns:
        One of Verdict.VALID, INVALID_FIXABLE, INVALID_MAJOR_REVISION.
    """
    not_met = [a for a in assessments if a.match_degree == MatchDegree.NOT_MET]
    if not not_met:
        return Verdict.VALID
    critical_gaps = [a for a in not_met if a.severity == Severity.CRITICAL]
    if critical_gaps:
        return Verdict.INVALID_MAJOR_REVISION
    return Verdict.INVALID_FIXABLE


def build_gap_report(
    session_id: str,
    letter_chunks: list[LetterChunk],
    assessments: list[RequirementAssessment],
    verdict: Verdict,
    metadata: dict | None = None,
) -> GapReport:
    """Assemble a GapReport from all analysis components.

    Args:
        session_id: UUID string for the analysis session.
        letter_chunks: All chunks parsed from the original letter.
        assessments: Aggregated 15-requirement assessments.
        verdict: Computed overall verdict.
        metadata: Optional dict with run metadata (timing, model versions, etc.)

    Returns:
        GapReport ready for serialization and storage.
    """
    return GapReport(
        session_id=session_id,
        verdict=verdict,
        assessments=assessments,
        letter_chunks=letter_chunks,
        metadata=metadata or {},
    )
