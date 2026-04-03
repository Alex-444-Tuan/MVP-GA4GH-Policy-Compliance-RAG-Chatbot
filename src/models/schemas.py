"""Pydantic models for all data structures in the GA4GH compliance pipeline."""

from __future__ import annotations

from datetime import datetime
from enum import Enum
from typing import Any
from uuid import uuid4

from pydantic import BaseModel, Field


# ── Enumerations ──────────────────────────────────────────────────────────────


class MatchDegree(str, Enum):
    """Per-requirement compliance classification."""

    FULLY_MET = "FULLY_MET"
    PARTIALLY_MET = "PARTIALLY_MET"
    NOT_MET = "NOT_MET"


class Severity(str, Enum):
    """Importance level of a compliance requirement."""

    CRITICAL = "CRITICAL"
    MAJOR = "MAJOR"
    MINOR = "MINOR"


class Verdict(str, Enum):
    """Overall compliance verdict for a letter."""

    VALID = "VALID"
    INVALID_FIXABLE = "INVALID_FIXABLE"
    INVALID_MAJOR_REVISION = "INVALID_MAJOR_REVISION"


# ── Ingestion models ──────────────────────────────────────────────────────────


class LetterChunk(BaseModel):
    """A semantically coherent segment of a researcher's data-use letter."""

    chunk_id: str = Field(default_factory=lambda: str(uuid4()))
    text: str
    section_title: str | None = None
    chunk_index: int
    start_char: int = 0
    end_char: int = 0


# ── Knowledge graph / retrieval models ───────────────────────────────────────


class RequirementNode(BaseModel):
    """A compliance requirement from the GA4GH knowledge graph."""

    id: str
    description: str
    severity: Severity
    keywords: list[str]
    daa_clause_category: str
    daa_clause_number: str | None = None


class PolicyChunkResult(BaseModel):
    """A policy chunk returned by lexical or semantic search."""

    chunk_id: str
    policy_id: str
    section_title: str | None
    text: str
    lexical_rank: int = 999
    semantic_rank: int = 999
    rrf_score: float = 0.0


class PolicyFormChunkResult(BaseModel):
    """A DAA clause chunk returned from graph traversal."""

    chunk_id: str
    clause_category: str
    clause_number: str
    text: str
    relevance_weight: float = 0.0


class RetrievalResult(BaseModel):
    """All retrieval results for a single letter chunk."""

    letter_chunk: LetterChunk
    policy_chunks: list[PolicyChunkResult] = []
    requirements: list[RequirementNode] = []
    form_chunks: list[PolicyFormChunkResult] = []


# ── Analysis models ───────────────────────────────────────────────────────────


class RequirementAssessment(BaseModel):
    """LLM-generated assessment of one requirement against a letter."""

    requirement_id: str
    description: str = ""
    match_degree: MatchDegree
    evidence_from_letter: str
    evidence_from_policy: str
    reasoning: str
    severity: Severity = Severity.MAJOR


class Contradiction(BaseModel):
    """A detected contradiction between two sections of a letter."""

    claim_a: str
    claim_b: str
    nature_of_contradiction: str
    severity: Severity


class ManualField(BaseModel):
    """A field the researcher must fill in manually in the suggested text."""

    field_name: str
    example: str


class RemediationItem(BaseModel):
    """Actionable remediation guidance for a single gap."""

    gap_id: str
    severity: Severity
    clause_category: str
    suggested_text: str
    auto_filled_fields: dict[str, str] = {}
    manual_fields: list[ManualField] = []
    explanation: str


class GapReport(BaseModel):
    """Full compliance gap report for a researcher's data-use letter."""

    session_id: str
    verdict: Verdict
    assessments: list[RequirementAssessment] = []
    remediations: list[RemediationItem] = []
    contradictions: list[Contradiction] = []
    letter_chunks: list[LetterChunk] = []
    metadata: dict[str, Any] = {}
    created_at: datetime = Field(default_factory=datetime.utcnow)

    @property
    def gaps(self) -> list[RequirementAssessment]:
        """Return assessments that are NOT_MET or PARTIALLY_MET."""
        return [
            a for a in self.assessments
            if a.match_degree != MatchDegree.FULLY_MET
        ]

    @property
    def critical_gaps(self) -> list[RequirementAssessment]:
        """Return NOT_MET assessments with CRITICAL severity."""
        return [
            a for a in self.assessments
            if a.match_degree == MatchDegree.NOT_MET and a.severity == Severity.CRITICAL
        ]


# ── API request / response models ────────────────────────────────────────────


class AnalysisRequest(BaseModel):
    """Request body for POST /analyze (JSON upload path)."""

    letter_text: str = Field(..., min_length=50)
    rrf_lexical_weight: float = Field(default=0.4, ge=0.0, le=1.0)


class AnalysisResponse(BaseModel):
    """Response from POST /analyze."""

    session_id: str
    gap_report: GapReport


class SessionResponse(BaseModel):
    """Response from GET /session/{id}."""

    session_id: str
    gap_report: GapReport
    created_at: datetime


class FollowUpRequest(BaseModel):
    """Request body for POST /followup."""

    session_id: str
    message: str = Field(..., min_length=1)


class FollowUpResponse(BaseModel):
    """Response from POST /followup."""

    session_id: str
    response: str


# ── Seeding / internal models ─────────────────────────────────────────────────


class PolicyChunkSeed(BaseModel):
    """Internal model for seeding a policy chunk into both databases."""

    id: str
    policy_id: str
    section_title: str
    chunk_index: int
    text: str
    embedding: list[float] = []


class PolicyFormChunkSeed(BaseModel):
    """Internal model for seeding a DAA clause chunk into Neo4j."""

    id: str
    policy_form_id: str
    clause_category: str
    clause_number: str
    text: str
    embedding: list[float] = []
