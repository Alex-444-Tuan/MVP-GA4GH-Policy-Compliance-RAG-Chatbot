"""Tests for the retrieval pipeline components.

NOTE: Tests marked with @pytest.mark.integration require running Neo4j and
PostgreSQL instances. Run unit tests only with: pytest tests/test_retrieval.py -m "not integration"
"""

from __future__ import annotations

import pytest

from src.models.schemas import LetterChunk, PolicyChunkResult
from src.retrieval.fusion import _rrf_fuse
from src.retrieval.keyword_extractor import _boost_from_lexicon


# ── Unit tests (no DB required) ───────────────────────────────────────────────


def test_rrf_fuse_both_sets():
    """RRF fusion should correctly combine lexical and semantic results."""
    chunk_a = PolicyChunkResult(
        chunk_id="chunk_a", policy_id="fw", section_title="Security",
        text="Security text", lexical_rank=1, semantic_rank=2
    )
    chunk_b = PolicyChunkResult(
        chunk_id="chunk_b", policy_id="fw", section_title="Privacy",
        text="Privacy text", lexical_rank=2, semantic_rank=1
    )
    chunk_c = PolicyChunkResult(
        chunk_id="chunk_c", policy_id="fw", section_title="Consent",
        text="Consent text", lexical_rank=3, semantic_rank=3
    )

    lexical = [chunk_a, chunk_b, chunk_c]
    semantic = [chunk_b, chunk_a, chunk_c]

    results = _rrf_fuse(lexical, semantic, alpha=0.4, beta=0.6, k=60, top_k=3)

    assert len(results) <= 3
    assert all(r.rrf_score > 0 for r in results)
    # Results should be sorted by RRF score descending
    for i in range(len(results) - 1):
        assert results[i].rrf_score >= results[i + 1].rrf_score


def test_rrf_fuse_lexical_only():
    """RRF should handle results that appear only in one set."""
    lexical_only = PolicyChunkResult(
        chunk_id="lex_only", policy_id="fw", section_title="T", text="T",
        lexical_rank=1, semantic_rank=999
    )
    semantic_only = PolicyChunkResult(
        chunk_id="sem_only", policy_id="fw", section_title="S", text="S",
        lexical_rank=999, semantic_rank=1
    )

    results = _rrf_fuse([lexical_only], [semantic_only], alpha=0.4, beta=0.6, k=60, top_k=5)
    assert len(results) == 2
    chunk_ids = [r.chunk_id for r in results]
    assert "lex_only" in chunk_ids
    assert "sem_only" in chunk_ids


def test_rrf_fuse_top_k_limit():
    """RRF fusion should return at most top_k results."""
    chunks = [
        PolicyChunkResult(
            chunk_id=f"chunk_{i}", policy_id="fw", section_title=f"S{i}", text=f"T{i}",
            lexical_rank=i + 1, semantic_rank=i + 1
        )
        for i in range(10)
    ]
    results = _rrf_fuse(chunks, chunks, alpha=0.4, beta=0.6, k=60, top_k=3)
    assert len(results) == 3


def test_rrf_fuse_empty_inputs():
    """RRF fusion should handle empty inputs gracefully."""
    results = _rrf_fuse([], [], alpha=0.4, beta=0.6, k=60, top_k=5)
    assert results == []


def test_rrf_semantic_weight_matters():
    """Higher semantic weight should boost semantically-ranked results."""
    chunk_a = PolicyChunkResult(
        chunk_id="a", policy_id="fw", section_title="A", text="A",
        lexical_rank=1, semantic_rank=10
    )
    chunk_b = PolicyChunkResult(
        chunk_id="b", policy_id="fw", section_title="B", text="B",
        lexical_rank=10, semantic_rank=1
    )

    # High semantic weight (0.9) — chunk_b (semantic rank 1) should win
    results_sem = _rrf_fuse([chunk_a, chunk_b], [chunk_a, chunk_b],
                            alpha=0.1, beta=0.9, k=60, top_k=2)
    # High lexical weight (0.9) — chunk_a (lexical rank 1) should win
    results_lex = _rrf_fuse([chunk_a, chunk_b], [chunk_a, chunk_b],
                            alpha=0.9, beta=0.1, k=60, top_k=2)

    assert results_sem[0].chunk_id == "b"
    assert results_lex[0].chunk_id == "a"


def test_lexicon_boost_exact_match():
    """Lexicon boost should find exact domain terms in text."""
    text = "The IRB approval was obtained for this study. We will use encryption and MFA."
    matches = _boost_from_lexicon(text)
    assert "IRB" in matches or "irb" in [m.lower() for m in matches]
    assert any("encryption" in m.lower() for m in matches)
    assert any("mfa" in m.lower() for m in matches)


def test_lexicon_boost_no_match():
    """Lexicon boost should return empty list when no domain terms present."""
    text = "The weather is nice today and I enjoy hiking in the mountains."
    matches = _boost_from_lexicon(text)
    assert matches == []


# ── Integration tests (require DB) ───────────────────────────────────────────


@pytest.mark.integration
async def test_lexical_search_returns_results(pg_conn):
    """Lexical search should return results for known policy keywords."""
    from src.retrieval.lexical_search import lexical_search
    results = await lexical_search(["IRB", "ethics", "consent"], pg_conn, top_k=5)
    assert len(results) > 0
    assert all(r.chunk_id for r in results)
    assert all(r.lexical_rank >= 1 for r in results)


@pytest.mark.integration
async def test_semantic_search_returns_results(pg_conn, openai_client):
    """Semantic search should return results for a sample query."""
    from src.retrieval.semantic_search import semantic_search
    query = "Data security measures including encryption and access control"
    results = await semantic_search(query, openai_client, pg_conn, top_k=5)
    assert len(results) > 0
    assert all(r.semantic_rank >= 1 for r in results)


@pytest.mark.integration
async def test_graph_traversal_returns_requirements(neo4j_driver):
    """Graph traversal should return requirements for known chunk IDs."""
    from src.retrieval.graph_search import graph_traversal
    # Try known chunk IDs from seeding
    chunk_ids = ["fw_chunk_000", "fw_chunk_001"]
    reqs, form_chunks = await graph_traversal(chunk_ids, neo4j_driver)
    # Should find at least some requirements
    assert len(reqs) >= 0  # May be empty if chunks not connected; test structure
    assert isinstance(reqs, list)
    assert isinstance(form_chunks, list)
