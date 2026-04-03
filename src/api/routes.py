"""FastAPI route handlers for the compliance analysis API.

Endpoints:
- POST /analyze       — Upload letter (file or JSON), run full pipeline
- GET  /session/{id}  — Retrieve stored analysis session
- POST /followup      — Conversational follow-up on an existing session
"""

from __future__ import annotations

import json
import logging
import time
import uuid
from typing import Annotated

import anthropic
import asyncpg
import openai
from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status
from neo4j import AsyncDriver

from src.analysis.coherence_checker import check_coherence
from src.analysis.gap_detector import (
    aggregate_assessments,
    analyze_chunk,
    build_gap_report,
    compute_verdict,
)
from src.analysis.prompts import FOLLOWUP_SYSTEM_PROMPT
from src.analysis.remediation import generate_remediations
from src.api.dependencies import (
    get_anthropic_client,
    get_neo4j_driver,
    get_openai_client,
    get_pg_conn,
)
from src.config import get_settings
from src.ingestion.chunker import chunk_letter
from src.ingestion.document_parser import parse_document
from src.models.schemas import (
    AnalysisRequest,
    AnalysisResponse,
    FollowUpRequest,
    FollowUpResponse,
    GapReport,
    MatchDegree,
    RequirementAssessment,
    Severity,
    SessionResponse,
    Verdict,
)
from src.retrieval.fusion import retrieve_for_all_chunks
from src.retrieval.graph_search import get_all_requirements

router = APIRouter()
settings = get_settings()
logger = logging.getLogger(__name__)

# Build the canonical requirement metadata map from Neo4j on first use
_REQ_META_CACHE: dict[str, tuple[str, Severity]] | None = None


async def _get_req_meta(driver: AsyncDriver) -> dict[str, tuple[str, Severity]]:
    """Return cached requirement metadata (id → (description, severity))."""
    global _REQ_META_CACHE
    if _REQ_META_CACHE is None:
        all_reqs = await get_all_requirements(driver)
        _REQ_META_CACHE = {r.id: (r.description, r.severity) for r in all_reqs}
        logger.info("Loaded %d requirements from Neo4j.", len(_REQ_META_CACHE))
    return _REQ_META_CACHE


async def _run_full_pipeline(
    letter_text: str,
    rrf_lexical_weight: float,
    anthropic_client: anthropic.Anthropic,
    openai_client: openai.OpenAI,
    pg_conn: asyncpg.Connection,
    neo4j_driver: AsyncDriver,
) -> GapReport:
    """Execute the complete compliance analysis pipeline.

    Args:
        letter_text: Full text of the submitted data-use letter.
        rrf_lexical_weight: Lexical weight for RRF fusion (0.0–1.0).
        anthropic_client: Anthropic API client.
        openai_client: OpenAI API client.
        pg_conn: Active PostgreSQL connection.
        neo4j_driver: Neo4j async driver.

    Returns:
        Fully populated GapReport.
    """
    session_id = str(uuid.uuid4())
    t_start = time.perf_counter()

    # Phase 1: Chunk the letter
    chunks = chunk_letter(letter_text)
    logger.info("[%s] Letter chunked into %d chunks.", session_id, len(chunks))

    # Phase 2: Retrieval for all chunks
    retrieval_results = await retrieve_for_all_chunks(
        chunks,
        anthropic_client,
        openai_client,
        pg_conn,
        neo4j_driver,
        rrf_lexical_weight,
    )

    # Phase 3: Per-chunk gap analysis
    all_assessments: list[list[RequirementAssessment]] = []
    for result in retrieval_results:
        chunk_assessments = analyze_chunk(result, anthropic_client)
        all_assessments.append(chunk_assessments)

    # Phase 4: Aggregate to one assessment per requirement
    req_meta = await _get_req_meta(neo4j_driver)
    final_assessments = aggregate_assessments(all_assessments, req_meta)

    # Phase 5: Coherence check
    contradictions = check_coherence(chunks, anthropic_client)

    # Phase 6: Remediation for gaps
    remediations = generate_remediations(
        final_assessments, retrieval_results, letter_text, anthropic_client
    )

    # Phase 7: Compute verdict and build report
    verdict = compute_verdict(final_assessments)
    elapsed = time.perf_counter() - t_start

    gap_report = build_gap_report(
        session_id=session_id,
        letter_chunks=chunks,
        assessments=final_assessments,
        verdict=verdict,
        metadata={
            "elapsed_seconds": round(elapsed, 2),
            "chunk_count": len(chunks),
            "rrf_lexical_weight": rrf_lexical_weight,
            "analysis_model": settings.analysis_model,
            "embedding_model": settings.embedding_model,
        },
    )
    gap_report.contradictions = contradictions
    gap_report.remediations = remediations

    logger.info(
        "[%s] Analysis complete in %.2fs — verdict: %s, gaps: %d",
        session_id, elapsed, verdict.value,
        sum(1 for a in final_assessments if a.match_degree != MatchDegree.FULLY_MET),
    )
    return gap_report


async def _save_session(gap_report: GapReport, letter_text: str, pg_conn: asyncpg.Connection) -> None:
    """Persist the analysis session to PostgreSQL."""
    try:
        await pg_conn.execute(
            """
            INSERT INTO analysis_sessions (id, letter_text, letter_chunks, gap_report, verdict, metadata)
            VALUES ($1, $2, $3, $4, $5, $6)
            ON CONFLICT (id) DO NOTHING
            """,
            gap_report.session_id,
            letter_text,
            json.dumps([c.model_dump() for c in gap_report.letter_chunks], default=str),
            json.dumps(gap_report.model_dump(), default=str),
            gap_report.verdict.value,
            json.dumps(gap_report.metadata, default=str),
        )
    except Exception as e:
        logger.error("Failed to save session %s: %s", gap_report.session_id, e)


# ── Routes ────────────────────────────────────────────────────────────────────


@router.post("/analyze", response_model=AnalysisResponse, status_code=status.HTTP_200_OK)
async def analyze_letter_file(
    file: Annotated[UploadFile, File(description="PDF, DOCX, or TXT file of the data-use letter")],
    rrf_lexical_weight: Annotated[float, Form()] = 0.4,
    pg_conn: asyncpg.Connection = Depends(get_pg_conn),
    neo4j_driver: AsyncDriver = Depends(get_neo4j_driver),
    anthropic_client: anthropic.Anthropic = Depends(get_anthropic_client),
    openai_client: openai.OpenAI = Depends(get_openai_client),
) -> AnalysisResponse:
    """Analyze an uploaded data-use letter file for GA4GH policy compliance.

    Accepts PDF, DOCX, or TXT. Returns a full gap report with remediation.
    """
    if not file.filename:
        raise HTTPException(status_code=400, detail="No filename provided.")

    file_bytes = await file.read()
    if len(file_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    try:
        letter_text = parse_document(file_bytes, file.filename)
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    if len(letter_text.strip()) < 50:
        raise HTTPException(status_code=400, detail="Document text is too short to analyze.")

    gap_report = await _run_full_pipeline(
        letter_text, rrf_lexical_weight,
        anthropic_client, openai_client, pg_conn, neo4j_driver,
    )
    await _save_session(gap_report, letter_text, pg_conn)
    return AnalysisResponse(session_id=gap_report.session_id, gap_report=gap_report)


@router.post("/analyze/text", response_model=AnalysisResponse, status_code=status.HTTP_200_OK)
async def analyze_letter_text(
    request: AnalysisRequest,
    pg_conn: asyncpg.Connection = Depends(get_pg_conn),
    neo4j_driver: AsyncDriver = Depends(get_neo4j_driver),
    anthropic_client: anthropic.Anthropic = Depends(get_anthropic_client),
    openai_client: openai.OpenAI = Depends(get_openai_client),
) -> AnalysisResponse:
    """Analyze a data-use letter submitted as plain text JSON.

    Useful for testing and for the Streamlit UI.
    """
    gap_report = await _run_full_pipeline(
        request.letter_text, request.rrf_lexical_weight,
        anthropic_client, openai_client, pg_conn, neo4j_driver,
    )
    await _save_session(gap_report, request.letter_text, pg_conn)
    return AnalysisResponse(session_id=gap_report.session_id, gap_report=gap_report)


@router.get("/session/{session_id}", response_model=SessionResponse)
async def get_session(
    session_id: str,
    pg_conn: asyncpg.Connection = Depends(get_pg_conn),
) -> SessionResponse:
    """Retrieve a previously analyzed session by ID."""
    row = await pg_conn.fetchrow(
        "SELECT id, gap_report, created_at FROM analysis_sessions WHERE id = $1",
        session_id,
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"Session {session_id!r} not found.")

    gap_report = GapReport.model_validate(json.loads(row["gap_report"]))
    return SessionResponse(
        session_id=row["id"],
        gap_report=gap_report,
        created_at=row["created_at"],
    )


@router.post("/followup", response_model=FollowUpResponse)
async def followup(
    request: FollowUpRequest,
    pg_conn: asyncpg.Connection = Depends(get_pg_conn),
    anthropic_client: anthropic.Anthropic = Depends(get_anthropic_client),
) -> FollowUpResponse:
    """Handle a conversational follow-up question about an analysis session.

    Provides context-aware responses using the stored gap report.
    """
    row = await pg_conn.fetchrow(
        "SELECT gap_report FROM analysis_sessions WHERE id = $1",
        request.session_id,
    )
    if row is None:
        raise HTTPException(status_code=404, detail=f"Session {request.session_id!r} not found.")

    gap_report_json = row["gap_report"]

    user_context = (
        f"Here is the gap report summary for context:\n\n"
        f"Verdict: {json.loads(gap_report_json).get('verdict', 'Unknown')}\n"
        f"Number of gaps: {len([a for a in json.loads(gap_report_json).get('assessments', []) if a.get('match_degree') != 'FULLY_MET'])}\n\n"
        f"Researcher's question: {request.message}"
    )

    try:
        message = anthropic_client.messages.create(
            model=settings.analysis_model,
            max_tokens=1024,
            timeout=settings.llm_timeout_seconds,
            system=FOLLOWUP_SYSTEM_PROMPT,
            messages=[{"role": "user", "content": user_context}],
        )
        response_text = message.content[0].text.strip()
    except anthropic.APIError as e:
        logger.error("Follow-up LLM call failed: %s", e)
        raise HTTPException(status_code=503, detail="LLM service temporarily unavailable.")

    return FollowUpResponse(session_id=request.session_id, response=response_text)


@router.get("/health")
async def health_check() -> dict:
    """Basic health check endpoint."""
    return {"status": "ok", "version": "0.1.0"}
