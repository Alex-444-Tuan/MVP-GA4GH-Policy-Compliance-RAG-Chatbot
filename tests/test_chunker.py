"""Unit tests for the document parser and chunker."""

from __future__ import annotations

import pytest
from pathlib import Path

from src.ingestion.chunker import (
    LetterChunk,
    _structural_split,
    _semantic_split,
    chunk_letter,
)
from src.ingestion.document_parser import parse_document, _normalize_whitespace

DATA_DIR = Path(__file__).parent.parent / "data" / "test_letters"
SAMPLE_LETTER_PATH = DATA_DIR / "sample_letter_with_gaps.txt"
MINIMAL_LETTER_PATH = DATA_DIR / "minimal_sloppy_letter.txt"


# ── Document parser tests ─────────────────────────────────────────────────────


def test_parse_txt_file():
    """Parser should extract text from a plain TXT file."""
    text = SAMPLE_LETTER_PATH.read_bytes()
    result = parse_document(text, "letter.txt")
    assert isinstance(result, str)
    assert len(result) > 100
    assert "Dr. Sarah Chen" in result


def test_parse_unsupported_format():
    """Parser should raise ValueError for unsupported formats."""
    with pytest.raises(ValueError, match="Unsupported file format"):
        parse_document(b"content", "file.xyz")


def test_normalize_whitespace():
    """Whitespace normalizer should collapse multiple blank lines."""
    text = "Hello\n\n\n\nWorld\n  trailing  "
    result = _normalize_whitespace(text)
    assert "\n\n\n" not in result
    assert result.endswith("trailing")


# ── Structural split tests ────────────────────────────────────────────────────


def test_structural_split_numbered_sections():
    """Structural split should detect numbered sections."""
    text = """Dr. Jane Smith

1. PURPOSE AND OBJECTIVES

I want to study genomics.

2. DATA HANDLING AND SECURITY

Data will be encrypted."""
    sections = _structural_split(text)
    assert len(sections) >= 2
    titles = [t for t, _ in sections if t]
    assert any("PURPOSE" in t for t in titles)
    assert any("SECURITY" in t or "HANDLING" in t for t in titles)


def test_structural_split_no_headers():
    """Structural split should handle text with no section headers."""
    text = "This is a plain letter without any section headers. It should be returned as one section."
    sections = _structural_split(text)
    assert len(sections) == 1
    assert sections[0][0] is None  # No title


def test_structural_split_preserves_all_text():
    """Structural split should not drop any content."""
    text = SAMPLE_LETTER_PATH.read_text(encoding="utf-8")
    sections = _structural_split(text)
    combined = "".join(body for _, body in sections)
    # All text is preserved (modulo leading/trailing whitespace)
    assert len(combined) >= len(text) * 0.95


# ── Chunker integration tests ─────────────────────────────────────────────────


def test_chunk_letter_sample():
    """Chunker should produce multiple chunks for the sample letter."""
    text = SAMPLE_LETTER_PATH.read_text(encoding="utf-8")
    chunks = chunk_letter(text)
    assert len(chunks) >= 3
    assert all(isinstance(c, LetterChunk) for c in chunks)
    assert all(len(c.text) >= 50 for c in chunks)


def test_chunk_letter_indices_sequential():
    """Chunk indices should be sequential starting from 0."""
    text = SAMPLE_LETTER_PATH.read_text(encoding="utf-8")
    chunks = chunk_letter(text)
    for i, chunk in enumerate(chunks):
        assert chunk.chunk_index == i


def test_chunk_letter_minimal():
    """Chunker should handle short letters without error."""
    text = MINIMAL_LETTER_PATH.read_text(encoding="utf-8")
    chunks = chunk_letter(text)
    assert len(chunks) >= 1


def test_chunk_letter_no_chunks_too_short():
    """Chunker should filter out chunks shorter than 50 chars."""
    text = SAMPLE_LETTER_PATH.read_text(encoding="utf-8")
    chunks = chunk_letter(text)
    assert all(len(c.text) >= 50 for c in chunks)


def test_chunk_letter_ids_unique():
    """Each chunk should have a unique chunk_id."""
    text = SAMPLE_LETTER_PATH.read_text(encoding="utf-8")
    chunks = chunk_letter(text)
    ids = [c.chunk_id for c in chunks]
    assert len(ids) == len(set(ids))


def test_chunk_long_section():
    """Semantic fallback should split sections exceeding TARGET_CHUNK_CHARS."""
    long_section = "This is a sentence. " * 150  # ~3000 chars
    text = f"1. LONG SECTION\n\n{long_section}"
    chunks = chunk_letter(text)
    assert len(chunks) >= 2, "Long section should be split into multiple chunks"
