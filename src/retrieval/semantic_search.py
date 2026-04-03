"""pgvector-based semantic (cosine similarity) search for policy chunks.

Embeds the letter chunk using text-embedding-3-large and finds the
most similar policy chunks by cosine distance.
"""

from __future__ import annotations

import logging

import asyncpg
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_exponential

from src.config import get_settings
from src.models.schemas import PolicyChunkResult

logger = logging.getLogger(__name__)
settings = get_settings()


@retry(stop=stop_after_attempt(3), wait=wait_exponential(multiplier=1, min=1, max=8))
def embed_text(text: str, client: OpenAI) -> list[float]:
    """Embed a single text string using text-embedding-3-large (3072 dims).

    Args:
        text: Text to embed. Truncated to 8000 chars if longer.
        client: OpenAI client instance.

    Returns:
        3072-dimensional embedding vector.
    """
    response = client.embeddings.create(
        model=settings.embedding_model,
        input=text[:8000],
        dimensions=settings.embedding_dims,
    )
    return response.data[0].embedding


async def semantic_search(
    query_text: str,
    openai_client: OpenAI,
    pg_conn: asyncpg.Connection,
    top_k: int | None = None,
) -> list[PolicyChunkResult]:
    """Find policy chunks most semantically similar to the query text.

    Uses cosine similarity via pgvector (<=> operator) against
    pre-computed embeddings in the policy_chunks table.

    Args:
        query_text: The letter chunk text to search against.
        openai_client: OpenAI client for embedding generation.
        pg_conn: Active asyncpg connection.
        top_k: Maximum results. Defaults to settings.retrieval_top_k * 2.

    Returns:
        List of PolicyChunkResult ordered by cosine similarity (best first).
        Ranks are 1-based.
    """
    if top_k is None:
        top_k = settings.retrieval_top_k * 2

    # Generate query embedding
    try:
        embedding = embed_text(query_text, openai_client)
    except Exception as e:
        logger.error("Failed to generate query embedding: %s", e)
        return []

    # Format embedding as PostgreSQL vector literal
    embedding_str = f"[{','.join(str(x) for x in embedding)}]"

    sql = f"""
        SELECT
            id,
            policy_id,
            section_title,
            text,
            1 - (embedding <=> $1::vector) AS cosine_similarity
        FROM policy_chunks
        WHERE embedding IS NOT NULL
        ORDER BY embedding <=> $1::vector
        LIMIT {top_k}
    """

    try:
        rows = await pg_conn.fetch(sql, embedding_str)
    except Exception as e:
        logger.error("Semantic search query failed: %s", e)
        return []

    results: list[PolicyChunkResult] = []
    for rank, row in enumerate(rows, start=1):
        results.append(PolicyChunkResult(
            chunk_id=row["id"],
            policy_id=row["policy_id"],
            section_title=row["section_title"],
            text=row["text"],
            lexical_rank=999,
            semantic_rank=rank,
            rrf_score=0.0,
        ))

    logger.debug("Semantic search returned %d results for query (%d chars)", len(results), len(query_text))
    return results
