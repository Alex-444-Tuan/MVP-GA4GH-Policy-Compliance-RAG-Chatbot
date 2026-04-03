"""Two-pass chunking strategy for researcher data-use letters.

Pass 1 — Structural: split on numbered section headers (e.g., "1. PURPOSE", "SECTION 2").
Pass 2 — Semantic fallback: for sections without structure (or over-long sections),
          split by sentence boundaries targeting ~350 tokens (~1400 chars) with
          50-token (~200 char) overlap.

Usage:
    from src.ingestion.chunker import chunk_letter
    chunks = chunk_letter(letter_text)
"""

from __future__ import annotations

import logging
import re
from uuid import uuid4

from src.models.schemas import LetterChunk

logger = logging.getLogger(__name__)

# Structural section header patterns
_NUMBERED_HEADER = re.compile(
    r"^(\d+[\.\)]\s+[A-Z][A-Z\s/&,-]{2,}|[A-Z]{3}[A-Z\s/&,-]{2,})$",
    re.MULTILINE,
)
_ALL_CAPS_HEADER = re.compile(
    r"^([A-Z]{3}[A-Z\s/&,-]{3,})$",
    re.MULTILINE,
)

TARGET_CHUNK_CHARS = 1400     # ≈ 350 tokens
OVERLAP_CHARS = 200           # ≈ 50 tokens


def chunk_letter(text: str) -> list[LetterChunk]:
    """Chunk a data-use letter into semantically coherent segments.

    Strategy:
    1. Split on structural section headers (numbered or ALL-CAPS).
    2. For each section, if it exceeds TARGET_CHUNK_CHARS, apply semantic
       sentence-boundary splitting with overlap.
    3. Filter out chunks that are too short (<50 chars) to be meaningful.

    Args:
        text: Full plain text of the researcher's letter.

    Returns:
        List of LetterChunk objects ordered by position in the letter.
    """
    sections = _structural_split(text)
    chunks: list[LetterChunk] = []
    char_offset = 0

    for section_title, section_text in sections:
        if len(section_text.strip()) < 50:
            char_offset += len(section_text)
            continue

        if len(section_text) <= TARGET_CHUNK_CHARS:
            # Section fits in a single chunk
            chunk = LetterChunk(
                chunk_id=str(uuid4()),
                text=section_text.strip(),
                section_title=section_title,
                chunk_index=len(chunks),
                start_char=char_offset,
                end_char=char_offset + len(section_text),
            )
            chunks.append(chunk)
        else:
            # Semantic fallback: split long section by sentence boundaries
            sub_chunks = _semantic_split(
                section_text,
                section_title=section_title,
                base_offset=char_offset,
                starting_index=len(chunks),
            )
            chunks.extend(sub_chunks)

        char_offset += len(section_text)

    logger.info("Produced %d chunks from letter (%d chars)", len(chunks), len(text))
    return chunks


def _structural_split(text: str) -> list[tuple[str | None, str]]:
    """Split text on structural section headers.

    Returns list of (section_title, section_text) tuples.
    The first tuple may have section_title=None for preamble text.
    """
    # Try numbered headers first
    matches = list(_NUMBERED_HEADER.finditer(text))

    # Fall back to ALL-CAPS headers if no numbered ones found
    if len(matches) < 2:
        matches = list(_ALL_CAPS_HEADER.finditer(text))

    if not matches:
        # No structural headers — treat whole text as one section
        return [(None, text)]

    sections: list[tuple[str | None, str]] = []

    # Preamble before the first header
    preamble = text[: matches[0].start()].strip()
    if preamble:
        sections.append((None, preamble))

    for i, match in enumerate(matches):
        section_title = match.group(0).strip()
        start = match.start()
        end = matches[i + 1].start() if i + 1 < len(matches) else len(text)
        section_text = text[start:end]
        sections.append((section_title, section_text))

    return sections


def _semantic_split(
    text: str,
    section_title: str | None,
    base_offset: int,
    starting_index: int,
) -> list[LetterChunk]:
    """Split a long text segment into chunks at sentence boundaries.

    Uses a greedy approach: accumulate sentences until TARGET_CHUNK_CHARS
    is reached, then start a new chunk with OVERLAP_CHARS lookback.

    Args:
        text: The text segment to split.
        section_title: Section header for all resulting chunks.
        base_offset: Character offset of this segment in the full letter.
        starting_index: Chunk index to start counting from.

    Returns:
        List of LetterChunk objects.
    """
    sentences = _split_sentences(text)
    chunks: list[LetterChunk] = []
    current_sentences: list[str] = []
    current_len = 0
    current_start = base_offset

    for sentence in sentences:
        sentence_len = len(sentence)

        if current_len + sentence_len > TARGET_CHUNK_CHARS and current_sentences:
            # Emit current chunk
            chunk_text = " ".join(current_sentences).strip()
            if chunk_text:
                chunks.append(LetterChunk(
                    chunk_id=str(uuid4()),
                    text=chunk_text,
                    section_title=section_title,
                    chunk_index=starting_index + len(chunks),
                    start_char=current_start,
                    end_char=current_start + len(chunk_text),
                ))

            # Overlap: keep last ~OVERLAP_CHARS worth of sentences
            overlap_sentences: list[str] = []
            overlap_len = 0
            for s in reversed(current_sentences):
                if overlap_len + len(s) > OVERLAP_CHARS:
                    break
                overlap_sentences.insert(0, s)
                overlap_len += len(s)

            current_sentences = overlap_sentences + [sentence]
            current_len = sum(len(s) for s in current_sentences)
            current_start = base_offset  # approximate
        else:
            current_sentences.append(sentence)
            current_len += sentence_len

    # Emit final chunk
    if current_sentences:
        chunk_text = " ".join(current_sentences).strip()
        if chunk_text:
            chunks.append(LetterChunk(
                chunk_id=str(uuid4()),
                text=chunk_text,
                section_title=section_title,
                chunk_index=starting_index + len(chunks),
                start_char=current_start,
                end_char=current_start + len(chunk_text),
            ))

    return chunks


def _split_sentences(text: str) -> list[str]:
    """Split text into sentences using a simple regex heuristic.

    Splits on '. ', '! ', '? ' followed by a capital letter or end of string,
    while preserving common abbreviations (Dr., Prof., etc.).
    """
    # Protect common abbreviations from being treated as sentence boundaries
    protected = re.sub(
        r"\b(Dr|Prof|Mr|Mrs|Ms|Jr|Sr|vs|et al|i\.e|e\.g|Inc|Ltd|Corp|Fig|No|Vol)\.",
        r"\1<DOT>",
        text,
    )
    # Split on sentence boundaries
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z\d])", protected)
    # Restore abbreviation dots
    sentences = [p.replace("<DOT>", ".").strip() for p in parts if p.strip()]
    return sentences
