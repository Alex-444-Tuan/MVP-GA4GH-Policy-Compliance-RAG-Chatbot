"""Hybrid retrieval pipeline: RRF fusion of lexical + semantic + graph traversal.

Implements the full retrieval pipeline for one letter chunk:
1. Keyword extraction (Claude Haiku)
2. Lexical search (PostgreSQL tsvector)
3. Semantic search (pgvector cosine similarity)
4. Graph traversal (Neo4j: PolicyChunk → Requirements → PolicyFormChunks)
5. RRF fusion of lexical and semantic results
6. Assemble final RetrievalResult

RRF formula:
    score(d) = α * (1 / (k + lexical_rank)) + β * (1 / (k + semantic_rank))
    where α = rrf_lexical_weight, β = 1 - α, k = rrf_k (default 60)
    rank = 999 if document not in that result set
"""

from __future__ import annotations

import logging
import time

import anthropic
import asyncpg
from neo4j import AsyncDriver
from openai import OpenAI

from src.config import get_settings
from src.models.schemas import (
    LetterChunk,
    PolicyChunkResult,
    RetrievalResult,
)
from src.retrieval.graph_search import graph_traversal
from src.retrieval.keyword_extractor import extract_keywords
from src.retrieval.lexical_search import lexical_search
from src.retrieval.semantic_search import semantic_search

logger = logging.getLogger(__name__)
settings = get_settings()


async def retrieve_for_chunk(
    chunk: LetterChunk,
    anthropic_client: anthropic.Anthropic,
    openai_client: OpenAI,
    pg_conn: asyncpg.Connection,
    neo4j_driver: AsyncDriver,
    rrf_lexical_weight: float | None = None,
) -> RetrievalResult:
    """Run the full hybrid retrieval pipeline for a single letter chunk.

    Args:
        chunk: The letter chunk to retrieve policy context for.
        anthropic_client: Anthropic client for keyword extraction.
        openai_client: OpenAI client for embedding generation.
        pg_conn: Active PostgreSQL connection.
        neo4j_driver: Async Neo4j driver.
        rrf_lexical_weight: Override for RRF lexical weight (0.0–1.0).
                            Defaults to settings.rrf_lexical_weight.

    Returns:
        RetrievalResult containing fused policy chunks, requirements, and DAA clauses.
    """
    alpha = rrf_lexical_weight if rrf_lexical_weight is not None else settings.rrf_lexical_weight
    beta = 1.0 - alpha
    k = settings.rrf_k
    top_k_final = settings.retrieval_top_k

    t0 = time.perf_counter()

    # Step 1: Keyword extraction
    keywords = extract_keywords(chunk.text, anthropic_client)
    logger.debug("[chunk %d] Keywords: %s", chunk.chunk_index, keywords[:5])

    # Step 2 & 3: Lexical and semantic search (run concurrently via asyncio)
    import asyncio
    lexical_task = asyncio.create_task(
        lexical_search(keywords, pg_conn)
    )
    semantic_task = asyncio.create_task(
        semantic_search(chunk.text, openai_client, pg_conn)
    )
    lexical_results, semantic_results = await asyncio.gather(lexical_task, semantic_task)

    # Step 4: RRF fusion
    fused_chunks = _rrf_fuse(lexical_results, semantic_results, alpha, beta, k, top_k_final)
    logger.debug("[chunk %d] RRF top-%d: %s", chunk.chunk_index, top_k_final, [c.chunk_id for c in fused_chunks])

    # Step 5: Graph traversal from top-3 semantic results (by semantic rank)
    top_semantic_ids = [r.chunk_id for r in sorted(semantic_results, key=lambda x: x.semantic_rank)[:3]]
    requirements, form_chunks = await graph_traversal(top_semantic_ids, neo4j_driver)

    elapsed = time.perf_counter() - t0
    logger.info(
        "[chunk %d] Retrieval complete in %.2fs — %d policy chunks, %d requirements, %d DAA clauses",
        chunk.chunk_index, elapsed, len(fused_chunks), len(requirements), len(form_chunks),
    )

    return RetrievalResult(
        letter_chunk=chunk,
        policy_chunks=fused_chunks,
        requirements=requirements,
        form_chunks=form_chunks,
    )


def _rrf_fuse(
    lexical_results: list[PolicyChunkResult],
    semantic_results: list[PolicyChunkResult],
    alpha: float,
    beta: float,
    k: int,
    top_k: int,
) -> list[PolicyChunkResult]:
    """Apply Reciprocal Rank Fusion to combine lexical and semantic results.

    Args:
        lexical_results: Results ranked by tsvector score (rank 1 = best).
        semantic_results: Results ranked by cosine similarity (rank 1 = best).
        alpha: Weight for lexical score.
        beta: Weight for semantic score.
        k: RRF constant (standard: 60).
        top_k: Number of top results to return.

    Returns:
        List of PolicyChunkResult sorted by RRF score descending, length ≤ top_k.
    """
    # Build lookup: chunk_id → rank in each result set
    lexical_rank_map: dict[str, int] = {r.chunk_id: r.lexical_rank for r in lexical_results}
    semantic_rank_map: dict[str, int] = {r.chunk_id: r.semantic_rank for r in semantic_results}
    chunk_map: dict[str, PolicyChunkResult] = {}

    for r in lexical_results + semantic_results:
        if r.chunk_id not in chunk_map:
            chunk_map[r.chunk_id] = r

    # Compute RRF scores
    scored: list[tuple[float, PolicyChunkResult]] = []
    for chunk_id, chunk in chunk_map.items():
        lex_rank = lexical_rank_map.get(chunk_id, 999)
        sem_rank = semantic_rank_map.get(chunk_id, 999)
        rrf_score = alpha * (1.0 / (k + lex_rank)) + beta * (1.0 / (k + sem_rank))

        result = PolicyChunkResult(
            chunk_id=chunk.chunk_id,
            policy_id=chunk.policy_id,
            section_title=chunk.section_title,
            text=chunk.text,
            lexical_rank=lex_rank,
            semantic_rank=sem_rank,
            rrf_score=rrf_score,
        )
        scored.append((rrf_score, result))

    scored.sort(key=lambda x: x[0], reverse=True)
    return [r for _, r in scored[:top_k]]


async def retrieve_for_all_chunks(
    chunks: list[LetterChunk],
    anthropic_client: anthropic.Anthropic,
    openai_client: OpenAI,
    pg_conn: asyncpg.Connection,
    neo4j_driver: AsyncDriver,
    rrf_lexical_weight: float | None = None,
) -> list[RetrievalResult]:
    """Run retrieval for all letter chunks sequentially.

    Sequential (not parallel) to avoid overwhelming the LLM APIs and DB
    with concurrent requests in the MVP.

    Args:
        chunks: All chunks from the parsed letter.
        anthropic_client: Anthropic API client.
        openai_client: OpenAI API client.
        pg_conn: Active PostgreSQL connection.
        neo4j_driver: Async Neo4j driver.
        rrf_lexical_weight: Optional RRF weight override.

    Returns:
        List of RetrievalResult, one per chunk.
    """
    results: list[RetrievalResult] = []
    for chunk in chunks:
        result = await retrieve_for_chunk(
            chunk,
            anthropic_client,
            openai_client,
            pg_conn,
            neo4j_driver,
            rrf_lexical_weight,
        )
        results.append(result)
    return results
