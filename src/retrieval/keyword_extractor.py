"""LLM-assisted keyword extraction from letter chunks, boosted by domain lexicon.

Uses Claude Haiku for fast extraction, then cross-references with
data/lexicon/policy_terms.json to boost exact domain matches.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path

import anthropic
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings

logger = logging.getLogger(__name__)
settings = get_settings()

_LEXICON_PATH = Path(__file__).parent.parent.parent / "data" / "lexicon" / "policy_terms.json"
_lexicon_flat: list[str] | None = None


def _load_lexicon() -> list[str]:
    """Load and flatten the domain keyword lexicon. Cached after first load."""
    global _lexicon_flat
    if _lexicon_flat is not None:
        return _lexicon_flat
    data = json.loads(_LEXICON_PATH.read_text(encoding="utf-8"))
    terms: list[str] = []
    for category_terms in data["terms"].values():
        terms.extend(category_terms)
    _lexicon_flat = [t.lower() for t in terms]
    logger.debug("Loaded lexicon with %d terms.", len(_lexicon_flat))
    return _lexicon_flat


_KEYWORD_EXTRACTION_PROMPT = """Extract 5-10 policy-relevant keywords from this text. Focus on terms related to: data governance, consent, security, IP rights, ethics oversight, access control, data breach, attribution, jurisdiction, re-identification, data destruction, third-party sharing, compliance monitoring, and sanctions.

Return ONLY a JSON array of keyword strings. No explanation, no markdown, no preamble.

Example output: ["IRB approval", "data encryption", "consent compatibility", "re-identification"]

Text to analyze:
{text}"""


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def extract_keywords(text: str, client: anthropic.Anthropic) -> list[str]:
    """Extract policy-relevant keywords from a letter chunk.

    Combines LLM extraction with lexicon boosting:
    1. Send chunk to Claude Haiku for keyword extraction.
    2. Cross-reference with policy_terms.json.
    3. Add any lexicon terms found verbatim in the text (boosted exact matches).

    Args:
        text: The letter chunk text to extract keywords from.
        client: Anthropic API client.

    Returns:
        Deduplicated list of policy-relevant keywords.
    """
    llm_keywords = _extract_via_llm(text, client)
    lexicon_matches = _boost_from_lexicon(text)

    # Merge and deduplicate (preserve order: LLM first, then lexicon boosts)
    seen: set[str] = set()
    combined: list[str] = []
    for kw in llm_keywords + lexicon_matches:
        kw_lower = kw.lower()
        if kw_lower not in seen:
            seen.add(kw_lower)
            combined.append(kw)

    logger.debug("Extracted %d keywords (%d LLM, %d lexicon boosts)",
                 len(combined), len(llm_keywords), len(lexicon_matches))
    return combined


def _extract_via_llm(text: str, client: anthropic.Anthropic) -> list[str]:
    """Call Claude Haiku to extract keywords. Parses JSON response."""
    try:
        message = client.messages.create(
            model=settings.preprocessing_model,
            max_tokens=256,
            timeout=settings.llm_timeout_seconds,
            messages=[
                {
                    "role": "user",
                    "content": _KEYWORD_EXTRACTION_PROMPT.format(text=text[:2000]),
                }
            ],
        )
        raw = message.content[0].text.strip()
        keywords = json.loads(raw)
        if isinstance(keywords, list):
            return [str(k) for k in keywords if k]
        logger.warning("LLM keyword response was not a list: %r", raw[:200])
        return []
    except json.JSONDecodeError:
        logger.warning("Failed to parse keyword JSON. Raw: %r", raw[:200] if "raw" in dir() else "unknown")
        return []
    except Exception as e:
        logger.error("Keyword extraction LLM call failed: %s", e)
        return []


def _boost_from_lexicon(text: str) -> list[str]:
    """Find domain lexicon terms that appear verbatim in the text."""
    text_lower = text.lower()
    lexicon = _load_lexicon()
    found = [term for term in lexicon if term in text_lower]
    return found
