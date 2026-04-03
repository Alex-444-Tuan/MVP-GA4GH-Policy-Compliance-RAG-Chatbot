"""SQLAlchemy ORM models for PostgreSQL tables."""

from __future__ import annotations

import uuid
from datetime import datetime

from pgvector.sqlalchemy import Vector
from sqlalchemy import DateTime, Integer, String, Text
from sqlalchemy.dialects.postgresql import JSONB, UUID
from sqlalchemy.ext.asyncio import AsyncAttrs
from sqlalchemy.orm import DeclarativeBase, Mapped, mapped_column


class Base(AsyncAttrs, DeclarativeBase):
    """Base class for all SQLAlchemy models."""

    pass


class PolicyChunkModel(Base):
    """Stores policy text chunks with tsvector and pgvector columns.

    The tsvector_col is a GENERATED ALWAYS column — managed by PostgreSQL,
    not by SQLAlchemy. Do not write to it from Python.
    """

    __tablename__ = "policy_chunks"

    id: Mapped[str] = mapped_column(String, primary_key=True)
    policy_id: Mapped[str] = mapped_column(String, nullable=False, index=True)
    section_title: Mapped[str | None] = mapped_column(String, nullable=True)
    chunk_index: Mapped[int] = mapped_column(Integer, nullable=False)
    text: Mapped[str] = mapped_column(Text, nullable=False)
    # text-embedding-3-large = 3072 dimensions
    embedding: Mapped[list[float] | None] = mapped_column(Vector(3072), nullable=True)
    # tsvector_col: GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
    # Not declared here — created via raw DDL in seed_postgres.py


class AnalysisSessionModel(Base):
    """Stores full analysis sessions including gap reports for retrieval."""

    __tablename__ = "analysis_sessions"

    id: Mapped[str] = mapped_column(
        String,
        primary_key=True,
        default=lambda: str(uuid.uuid4()),
    )
    created_at: Mapped[datetime] = mapped_column(
        DateTime(timezone=True),
        default=datetime.utcnow,
        nullable=False,
    )
    letter_text: Mapped[str] = mapped_column(Text, nullable=False)
    letter_chunks: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    gap_report: Mapped[dict | None] = mapped_column(JSONB, nullable=True)
    verdict: Mapped[str | None] = mapped_column(String, nullable=True)
    metadata_: Mapped[dict | None] = mapped_column(
        "metadata", JSONB, nullable=True
    )
