"""Parse PDF, DOCX, and plain-text documents into plain text strings.

Supported formats:
- PDF  — via PyMuPDF (imported as fitz)
- DOCX — via python-docx
- TXT  — direct UTF-8 read

Usage:
    from src.ingestion.document_parser import parse_document
    text = parse_document(file_bytes, filename="letter.pdf")
"""

from __future__ import annotations

import logging
from io import BytesIO
from pathlib import Path

logger = logging.getLogger(__name__)


def parse_document(file_bytes: bytes, filename: str) -> str:
    """Dispatch to the appropriate parser based on file extension.

    Args:
        file_bytes: Raw bytes of the uploaded file.
        filename: Original filename, used to detect format.

    Returns:
        Extracted plain text with whitespace normalized.

    Raises:
        ValueError: If the file format is not supported.
    """
    suffix = Path(filename).suffix.lower()
    logger.info("Parsing document: %s (%d bytes)", filename, len(file_bytes))

    if suffix == ".pdf":
        text = _parse_pdf(file_bytes)
    elif suffix in (".docx", ".doc"):
        text = _parse_docx(file_bytes)
    elif suffix in (".txt", ".md", ".text"):
        text = _parse_text(file_bytes)
    else:
        raise ValueError(f"Unsupported file format: {suffix!r}. Supported: .pdf, .docx, .txt")

    text = _normalize_whitespace(text)
    logger.info("Parsed %d characters from %s", len(text), filename)
    return text


def parse_file_path(path: str | Path) -> str:
    """Parse a document from a file path. Convenience wrapper for scripts.

    Args:
        path: Path to the document file.

    Returns:
        Extracted plain text.
    """
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Document not found: {path}")
    file_bytes = path.read_bytes()
    return parse_document(file_bytes, path.name)


def _parse_pdf(file_bytes: bytes) -> str:
    """Extract text from a PDF using PyMuPDF (fitz).

    Processes all pages and concatenates text with page separators.
    """
    try:
        import fitz  # PyMuPDF — imported as fitz per project convention
    except ImportError as e:
        raise ImportError("PyMuPDF is required for PDF parsing. Install with: pip install pymupdf") from e

    doc = fitz.open(stream=file_bytes, filetype="pdf")
    pages: list[str] = []
    for page_num, page in enumerate(doc, start=1):
        page_text = page.get_text()
        if page_text.strip():
            pages.append(page_text)
    doc.close()

    text = "\n\n".join(pages)
    logger.debug("Extracted %d pages from PDF", len(pages))
    return text


def _parse_docx(file_bytes: bytes) -> str:
    """Extract text from a DOCX file using python-docx.

    Preserves paragraph structure by joining with double newlines.
    """
    try:
        from docx import Document
    except ImportError as e:
        raise ImportError("python-docx is required for DOCX parsing. Install with: pip install python-docx") from e

    doc = Document(BytesIO(file_bytes))
    paragraphs = [para.text for para in doc.paragraphs if para.text.strip()]
    text = "\n\n".join(paragraphs)
    logger.debug("Extracted %d paragraphs from DOCX", len(paragraphs))
    return text


def _parse_text(file_bytes: bytes) -> str:
    """Decode plain text bytes as UTF-8, with latin-1 fallback."""
    try:
        return file_bytes.decode("utf-8")
    except UnicodeDecodeError:
        logger.warning("UTF-8 decoding failed, falling back to latin-1")
        return file_bytes.decode("latin-1")


def _normalize_whitespace(text: str) -> str:
    """Normalize whitespace: collapse multiple blank lines, strip trailing spaces."""
    import re
    # Collapse 3+ consecutive newlines to 2
    text = re.sub(r"\n{3,}", "\n\n", text)
    # Strip trailing whitespace from each line
    lines = [line.rstrip() for line in text.splitlines()]
    text = "\n".join(lines)
    return text.strip()
