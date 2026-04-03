"""Generate actionable remediation guidance for compliance gaps.

For each NOT_MET or PARTIALLY_MET requirement, retrieves the corresponding
DAA clause template from graph traversal results and calls the LLM to
generate fill-in guidance tailored to the specific letter's context.
"""

from __future__ import annotations

import json
import logging
import time

import anthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.analysis.prompts import LETTER_CONTEXT_EXTRACTION_PROMPT, REMEDIATION_PROMPT
from src.config import get_settings
from src.models.schemas import (
    ManualField,
    MatchDegree,
    RemediationItem,
    RequirementAssessment,
    RetrievalResult,
    Severity,
)

logger = logging.getLogger(__name__)


def extract_letter_context(letter_text: str, client: anthropic.Anthropic) -> dict[str, str]:
    """Extract PI name, institution, and project title from the letter.

    Uses Claude Haiku for fast extraction. Falls back to 'Unknown' for any field
    that cannot be determined.

    Args:
        letter_text: Full text of the data-use letter.
        client: Anthropic API client.

    Returns:
        Dict with keys: pi_name, institution, project_title, dataset_id.
    """
    cfg = get_settings()
    defaults = {
        "pi_name": "Unknown",
        "institution": "Unknown",
        "project_title": "Unknown",
        "dataset_id": "Unknown",
    }
    try:
        message = client.messages.create(
            model=cfg.preprocessing_model,
            max_tokens=256,
            timeout=cfg.llm_timeout_seconds,
            messages=[
                {
                    "role": "user",
                    "content": LETTER_CONTEXT_EXTRACTION_PROMPT.format(letter_text=letter_text[:4000]),
                }
            ],
        )
        raw = message.content[0].text.strip()
        parsed = json.loads(raw)
        return {**defaults, **{k: v for k, v in parsed.items() if k in defaults}}
    except (json.JSONDecodeError, anthropic.APIError, Exception) as e:
        logger.warning("Letter context extraction failed: %s", e)
        return defaults


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((anthropic.APIError, anthropic.APIConnectionError)),
)
def _call_remediation_llm(
    assessment: RequirementAssessment,
    clause_text: str,
    letter_context: dict[str, str],
    client: anthropic.Anthropic,
) -> dict:
    """Call Claude Sonnet to generate remediation guidance for one gap.

    Args:
        assessment: The gap assessment for this requirement.
        clause_text: Full text of the corresponding DAA clause template.
        letter_context: Dict with pi_name, institution, project_title.
        client: Anthropic API client.

    Returns:
        Parsed JSON dict matching the RemediationItem schema.
    """
    user_message = REMEDIATION_PROMPT.format(
        requirement_description=assessment.description,
        severity=assessment.severity.value,
        match_degree=assessment.match_degree.value,
        evidence=assessment.evidence_from_letter[:500],
        clause_template_text=clause_text[:2000],
        pi_name=letter_context.get("pi_name", "Unknown"),
        institution=letter_context.get("institution", "Unknown"),
        project_title=letter_context.get("project_title", "Unknown"),
        requirement_id=assessment.requirement_id,
    )

    cfg = get_settings()
    t0 = time.perf_counter()
    message = client.messages.create(
        model=cfg.analysis_model,
        max_tokens=1024,
        timeout=cfg.llm_timeout_seconds,
        messages=[{"role": "user", "content": user_message}],
    )
    elapsed = time.perf_counter() - t0
    logger.debug("Remediation LLM call for %s: %.2fs", assessment.requirement_id, elapsed)

    raw = message.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse remediation JSON for %s. Raw: %r",
                     assessment.requirement_id, raw[:300])
        raise ValueError(f"LLM returned invalid JSON: {e}") from e


def _parse_remediation_response(raw: dict, assessment: RequirementAssessment) -> RemediationItem:
    """Convert raw LLM JSON dict to a RemediationItem Pydantic model."""
    manual_fields = [
        ManualField(
            field_name=mf.get("field_name", ""),
            example=mf.get("example", ""),
        )
        for mf in raw.get("manual_fields", [])
    ]
    return RemediationItem(
        gap_id=assessment.requirement_id,
        severity=assessment.severity,
        clause_category=raw.get("clause_category", assessment.description),
        suggested_text=raw.get("suggested_text", ""),
        auto_filled_fields=raw.get("auto_filled_fields", {}),
        manual_fields=manual_fields,
        explanation=raw.get("explanation", ""),
    )


def generate_remediations(
    assessments: list[RequirementAssessment],
    retrieval_results: list[RetrievalResult],
    letter_text: str,
    client: anthropic.Anthropic,
) -> list[RemediationItem]:
    """Generate remediation guidance for all NOT_MET and PARTIALLY_MET gaps.

    Matches each gapped requirement to its DAA clause template (from retrieval
    results) and calls the LLM to generate fill-in guidance.

    Args:
        assessments: All 15 requirement assessments.
        retrieval_results: Retrieval results containing form chunks (DAA clauses).
        letter_text: Full letter text (for context extraction).
        client: Anthropic API client.

    Returns:
        List of RemediationItem for each gap, ordered by severity (CRITICAL first).
    """
    # Extract letter context once
    letter_context = extract_letter_context(letter_text, client)
    logger.info("Letter context: PI=%s, Institution=%s, Project=%s",
                letter_context["pi_name"], letter_context["institution"], letter_context["project_title"])

    # Build a map from DAA clause ID / category to clause text
    clause_map = _build_clause_map(retrieval_results)

    # Filter to only gaps requiring remediation
    gaps = [
        a for a in assessments
        if a.match_degree in (MatchDegree.NOT_MET, MatchDegree.PARTIALLY_MET)
    ]

    # Sort: CRITICAL first, then MAJOR, then MINOR
    severity_order = {Severity.CRITICAL: 0, Severity.MAJOR: 1, Severity.MINOR: 2}
    gaps.sort(key=lambda a: severity_order[a.severity])

    remediations: list[RemediationItem] = []
    for assessment in gaps:
        clause_text = _find_clause_for_requirement(assessment, clause_map)

        try:
            raw = _call_remediation_llm(assessment, clause_text, letter_context, client)
            item = _parse_remediation_response(raw, assessment)
            remediations.append(item)
            logger.debug("Generated remediation for %s", assessment.requirement_id)
        except (ValueError, anthropic.APIError) as e:
            logger.error("Remediation generation failed for %s: %s", assessment.requirement_id, e)
            # Add a minimal fallback remediation
            remediations.append(RemediationItem(
                gap_id=assessment.requirement_id,
                severity=assessment.severity,
                clause_category=assessment.description,
                suggested_text=f"[Add language addressing: {assessment.description}]",
                auto_filled_fields={},
                manual_fields=[ManualField(
                    field_name="Required information",
                    example="Provide specific details as required by the GA4GH DAA clause.",
                )],
                explanation=f"This requirement ({assessment.requirement_id}) must be addressed. "
                            f"Refer to the GA4GH Model DAA Clauses for template language.",
            ))

    logger.info("Generated %d remediation items for %d gaps.", len(remediations), len(gaps))
    return remediations


def _build_clause_map(retrieval_results: list[RetrievalResult]) -> dict[str, str]:
    """Build a mapping from clause_category and chunk_id to clause text."""
    clause_map: dict[str, str] = {}
    for result in retrieval_results:
        for fc in result.form_chunks:
            clause_map[fc.chunk_id] = fc.text
            clause_map[fc.clause_category.lower()] = fc.text
            clause_map[f"daa_clause_{fc.clause_number.zfill(2)}"] = fc.text
    return clause_map


def _find_clause_for_requirement(
    assessment: RequirementAssessment,
    clause_map: dict[str, str],
) -> str:
    """Look up the DAA clause text for a given requirement assessment."""
    # Try by requirement ID → clause number pattern
    req_num = assessment.requirement_id.replace("REQ-", "")
    clause_id = f"daa_clause_{req_num.zfill(2)}"
    if clause_id in clause_map:
        return clause_map[clause_id]

    # Try by description keywords
    desc_lower = assessment.description.lower()
    for key, text in clause_map.items():
        if any(word in desc_lower for word in key.lower().split()):
            return text

    # Fallback: return generic guidance
    return (
        f"The researcher must address the following in their data-use letter: "
        f"{assessment.description}. Please refer to the GA4GH Model Data Access Agreement "
        f"Clauses for template language."
    )
