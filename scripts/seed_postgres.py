"""Seed PostgreSQL with policy chunks — tsvector for lexical search + pgvector for semantic search.

Run after seed_knowledge_graph.py (chunks must already be created there first,
and embeddings can be reused from Neo4j or regenerated here).

Run once:
    python scripts/seed_postgres.py

Requires:
    - PostgreSQL running with pgvector extension (see docker-compose.yml)
    - .env file with POSTGRES_DSN and OPENAI_API_KEY
    - data/policies/ga4gh_framework.md and ga4gh_daa_clauses.md
"""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent.parent))

import asyncpg
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import configure_logging, get_settings
from scripts.seed_knowledge_graph import (
    chunk_daa_clauses,
    chunk_framework_text,
    embed_chunks_in_batches,
)

configure_logging()
logger = logging.getLogger(__name__)
settings = get_settings()
DATA_DIR = Path(__file__).parent.parent / "data"

DDL = """
CREATE EXTENSION IF NOT EXISTS vector;

CREATE TABLE IF NOT EXISTS policy_chunks (
    id TEXT PRIMARY KEY,
    policy_id TEXT NOT NULL,
    section_title TEXT,
    chunk_index INTEGER,
    text TEXT NOT NULL,
    embedding vector(3072),
    tsvector_col tsvector GENERATED ALWAYS AS (to_tsvector('english', text)) STORED
);

CREATE INDEX IF NOT EXISTS idx_policy_chunks_tsvector
    ON policy_chunks USING GIN (tsvector_col);

-- ivfflat index omitted: requires ≤2000 dims, but text-embedding-3-large = 3072 dims.
-- With ~35 chunks, sequential scan via <=> is instant. Add hnsw index when pgvector ≥0.6.0.

CREATE INDEX IF NOT EXISTS idx_policy_chunks_policy_id
    ON policy_chunks (policy_id);

CREATE TABLE IF NOT EXISTS analysis_sessions (
    id TEXT PRIMARY KEY,
    created_at TIMESTAMPTZ DEFAULT now(),
    letter_text TEXT NOT NULL,
    letter_chunks JSONB,
    gap_report JSONB,
    verdict TEXT,
    metadata JSONB
);
"""


async def create_schema(conn: asyncpg.Connection) -> None:
    """Create tables and indexes if they don't exist."""
    logger.info("Creating schema...")
    await conn.execute(DDL)
    logger.info("Schema ready.")


async def upsert_chunk(conn: asyncpg.Connection, chunk: dict) -> None:
    """Insert or update a single policy chunk row."""
    embedding_str = f"[{','.join(str(x) for x in chunk['embedding'])}]" if chunk.get("embedding") else None
    await conn.execute(
        """
        INSERT INTO policy_chunks (id, policy_id, section_title, chunk_index, text, embedding)
        VALUES ($1, $2, $3, $4, $5, $6::vector)
        ON CONFLICT (id) DO UPDATE
          SET policy_id = EXCLUDED.policy_id,
              section_title = EXCLUDED.section_title,
              chunk_index = EXCLUDED.chunk_index,
              text = EXCLUDED.text,
              embedding = EXCLUDED.embedding
        """,
        chunk["id"],
        chunk.get("policy_id", chunk.get("policy_form_id", "unknown")),
        chunk.get("section_title") or chunk.get("clause_category"),
        chunk.get("chunk_index", 0),
        chunk["text"],
        embedding_str,
    )


async def seed_postgres(framework_chunks: list[dict], daa_chunks: list[dict]) -> None:
    """Connect to PostgreSQL and seed all chunks."""
    # asyncpg needs postgresql:// not postgresql+asyncpg://
    dsn = settings.postgres_dsn.replace("postgresql+asyncpg://", "postgresql://")
    logger.info("Connecting to PostgreSQL...")
    conn = await asyncpg.connect(dsn)

    try:
        await create_schema(conn)

        all_chunks = framework_chunks + daa_chunks
        logger.info("Inserting %d chunks...", len(all_chunks))

        for i, chunk in enumerate(all_chunks):
            await upsert_chunk(conn, chunk)
            if (i + 1) % 10 == 0:
                logger.info("  Inserted %d/%d", i + 1, len(all_chunks))

        logger.info("All chunks inserted.")

        # Verify counts
        fw_count = await conn.fetchval(
            "SELECT COUNT(*) FROM policy_chunks WHERE policy_id = $1",
            "ga4gh_framework_v1",
        )
        daa_count = await conn.fetchval(
            "SELECT COUNT(*) FROM policy_chunks WHERE policy_id = $1",
            "ga4gh_daa_clauses_v1",
        )
        logger.info("PostgreSQL row counts — framework: %d, DAA clauses: %d", fw_count, daa_count)
    finally:
        await conn.close()


async def main_async() -> None:
    """Async entry point."""
    logger.info("=== Seeding PostgreSQL ===")

    framework_path = DATA_DIR / "policies" / "ga4gh_framework.md"
    daa_path = DATA_DIR / "policies" / "ga4gh_daa_clauses.md"

    framework_text = framework_path.read_text(encoding="utf-8")
    daa_text = daa_path.read_text(encoding="utf-8")

    logger.info("Chunking policy texts...")
    framework_chunks = chunk_framework_text(framework_text)
    daa_chunks = chunk_daa_clauses(daa_text)

    # Re-use DAA chunks with policy_id field for PostgreSQL (consistent IDs)
    for chunk in daa_chunks:
        chunk["policy_id"] = chunk.get("policy_form_id", "ga4gh_daa_clauses_v1")
        chunk.setdefault("chunk_index", int(chunk["clause_number"]))

    logger.info("Generating embeddings...")
    openai_client = OpenAI(api_key=settings.openai_api_key)
    framework_chunks = embed_chunks_in_batches(framework_chunks, openai_client)
    daa_chunks = embed_chunks_in_batches(daa_chunks, openai_client)

    await seed_postgres(framework_chunks, daa_chunks)
    logger.info("=== PostgreSQL seeding complete ===")


def main() -> None:
    asyncio.run(main_async())


if __name__ == "__main__":
    main()
