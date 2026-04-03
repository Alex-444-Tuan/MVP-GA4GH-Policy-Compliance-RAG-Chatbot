"""Cross-chunk contradiction detection for data-use letters.

Extracts claims from each letter chunk and uses the LLM to identify
contradictions between them (e.g., says data will be deleted in section 3
but plans indefinite retention in section 7).
"""

from __future__ import annotations

import json
import logging
import time

import anthropic
from tenacity import retry, retry_if_exception_type, stop_after_attempt, wait_exponential

from src.analysis.prompts import COHERENCE_CHECK_PROMPT
from src.config import get_settings
from src.models.schemas import Contradiction, LetterChunk, Severity

logger = logging.getLogger(__name__)


@retry(
    stop=stop_after_attempt(3),
    wait=wait_exponential(multiplier=1, min=2, max=15),
    retry=retry_if_exception_type((anthropic.APIError, anthropic.APIConnectionError)),
)
def _call_coherence_llm(claims_json: str, client: anthropic.Anthropic) -> dict:
    """Call Claude Sonnet to detect contradictions between claims.

    Args:
        claims_json: JSON array of claim objects with 'section' and 'text' fields.
        client: Anthropic API client.

    Returns:
        Parsed JSON dict with 'contradictions' list.
    """
    cfg = get_settings()
    t0 = time.perf_counter()
    message = client.messages.create(
        model=cfg.analysis_model,
        max_tokens=1024,
        timeout=cfg.llm_timeout_seconds,
        messages=[
            {
                "role": "user",
                "content": COHERENCE_CHECK_PROMPT.format(claims_json=claims_json[:4000]),
            }
        ],
    )
    elapsed = time.perf_counter() - t0
    logger.debug("Coherence check LLM call: %.2fs", elapsed)

    raw = message.content[0].text.strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError as e:
        logger.error("Failed to parse coherence check JSON. Raw: %r", raw[:300])
        raise ValueError(f"LLM returned invalid JSON: {e}") from e


def check_coherence(
    chunks: list[LetterChunk],
    client: anthropic.Anthropic,
) -> list[Contradiction]:
    """Detect contradictions across all chunks of a data-use letter.

    Extracts one representative claim per chunk (the first 300 chars),
    then asks the LLM to identify contradictions between them.

    Args:
        chunks: All letter chunks from the parsed document.
        client: Anthropic API client.

    Returns:
        List of Contradiction objects. Empty list if no contradictions found.
    """
    if len(chunks) < 2:
        logger.debug("Only %d chunk(s) — skipping coherence check.", len(chunks))
        return []

    claims = [
        {
            "section": chunk.section_title or f"Section {chunk.chunk_index + 1}",
            "text": chunk.text[:300],
        }
        for chunk in chunks
    ]
    claims_json = json.dumps(claims, indent=None)

    try:
        response = _call_coherence_llm(claims_json, client)
    except (ValueError, anthropic.APIError) as e:
        logger.error("Coherence check failed: %s", e)
        return []

    contradictions: list[Contradiction] = []
    for raw in response.get("contradictions", []):
        try:
            contradictions.append(Contradiction(
                claim_a=raw.get("claim_a", ""),
                claim_b=raw.get("claim_b", ""),
                nature_of_contradiction=raw.get("nature_of_contradiction", ""),
                severity=Severity(raw.get("severity", "MINOR")),
            ))
        except (ValueError, KeyError) as e:
            logger.warning("Skipping malformed contradiction: %s — %r", e, raw)
            continue

    logger.info("Coherence check found %d contradiction(s).", len(contradictions))
    return contradictions
