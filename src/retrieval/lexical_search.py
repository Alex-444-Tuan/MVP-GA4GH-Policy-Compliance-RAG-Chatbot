"""PostgreSQL tsvector-based lexical (full-text) search for policy chunks.

Uses ts_rank to score matches against the policy_chunks.tsvector_col index.
"""

from __future__ import annotations

import logging

import asyncpg

from src.config import get_settings
from src.models.schemas import PolicyChunkResult

logger = logging.getLogger(__name__)
settings = get_settings()


async def lexical_search(
    keywords: list[str],
    conn: asyncpg.Connection,
    top_k: int | None = None,
) -> list[PolicyChunkResult]:
    """Search policy chunks using PostgreSQL full-text search (tsvector).

    Builds a tsquery from the extracted keywords using OR-logic (|),
    then ranks results by ts_rank.

    Args:
        keywords: List of policy-relevant keywords to search for.
        conn: Active asyncpg connection.
        top_k: Maximum results to return. Defaults to settings.retrieval_top_k * 2.

    Returns:
        List of PolicyChunkResult ordered by lexical relevance (best first).
        Ranks are 1-based (rank 1 = best match).
    """
    if not keywords:
        logger.debug("No keywords provided for lexical search.")
        return []

    if top_k is None:
        top_k = settings.retrieval_top_k * 2

    # Build tsquery: join all keywords with OR operator
    # Each keyword is converted to a lexeme-safe form for plainto_tsquery
    # We use plainto_tsquery per term and combine with || (OR in tsquery)
    tsquery_parts = [f"plainto_tsquery('english', ${i + 1})" for i in range(len(keywords))]
    combined_query = " || ".join(tsquery_parts)

    sql = f"""
        SELECT
            id,
            policy_id,
            section_title,
            text,
            ts_rank(tsvector_col, ({combined_query})) AS score
        FROM policy_chunks
        WHERE tsvector_col @@ ({combined_query})
        ORDER BY score DESC
        LIMIT {top_k}
    """

    try:
        rows = await conn.fetch(sql, *keywords)
    except Exception as e:
        logger.error("Lexical search failed: %s", e)
        return []

    results: list[PolicyChunkResult] = []
    for rank, row in enumerate(rows, start=1):
        results.append(PolicyChunkResult(
            chunk_id=row["id"],
            policy_id=row["policy_id"],
            section_title=row["section_title"],
            text=row["text"],
            lexical_rank=rank,
            semantic_rank=999,
            rrf_score=0.0,
        ))

    logger.debug("Lexical search returned %d results for %d keywords", len(results), len(keywords))
    return results
