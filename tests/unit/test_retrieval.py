"""
Unit tests for RRF fusion.

run_hybrid_search is an integration concern (needs live Qdrant and
models) so we test it minimally here. The RRF logic itself is pure
and gets thorough unit testing.
"""

from __future__ import annotations

import pytest

from finsight.models.base import Chunk, ChunkMetadata
from finsight.services.retrieval import RRF_K, reciprocal_rank_fusion
from datetime import date


def make_chunk(chunk_id: str, score: float = 1.0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-001",
        content=f"content for {chunk_id}",
        score=score,
        token_count=50,
        metadata=ChunkMetadata(
            doc_id="doc-001",
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_type="10-K",
            filing_date=date(2023, 10, 27),
            section="Item 1A",
            chunk_index=0,
            token_count=50,
            embedding_model="nomic-embed-text",
            scopes=["public"],
        ),
    )


def test_rrf_single_list_preserves_order():
    chunks = [make_chunk("a"), make_chunk("b"), make_chunk("c")]
    result = reciprocal_rank_fusion([chunks])
    assert [c.chunk_id for c in result] == ["a", "b", "c"]


def test_rrf_chunk_in_both_lists_ranks_higher():
    dense = [make_chunk("shared"), make_chunk("dense-only")]
    sparse = [make_chunk("shared"), make_chunk("sparse-only")]
    result = reciprocal_rank_fusion([dense, sparse])
    assert result[0].chunk_id == "shared"


def test_rrf_deduplicates_chunks():
    dense = [make_chunk("a"), make_chunk("b")]
    sparse = [make_chunk("a"), make_chunk("c")]
    result = reciprocal_rank_fusion([dense, sparse])
    ids = [c.chunk_id for c in result]
    assert len(ids) == len(set(ids))


def test_rrf_empty_lists_return_empty():
    result = reciprocal_rank_fusion([[], []])
    assert result == []


def test_rrf_single_empty_list_uses_other():
    chunks = [make_chunk("a"), make_chunk("b")]
    result = reciprocal_rank_fusion([chunks, []])
    assert len(result) == 2


def test_rrf_scores_use_rank_not_original_score():
    high_score_chunk = make_chunk("low-rank", score=0.99)
    low_score_chunk = make_chunk("high-rank", score=0.01)
    result = reciprocal_rank_fusion([[low_score_chunk, high_score_chunk]])
    assert result[0].chunk_id == "high-rank"


def test_rrf_score_formula():
    chunks = [make_chunk("only")]
    result = reciprocal_rank_fusion([chunks])
    expected = 1.0 / (0 + RRF_K)
    assert abs(result[0].score - expected) < 1e-6


def test_rrf_multiple_lists_accumulate_score():
    chunk = make_chunk("everywhere")
    lists = [[chunk], [chunk], [chunk]]
    result = reciprocal_rank_fusion(lists)
    expected = 3.0 / (0 + RRF_K)
    assert abs(result[0].score - expected) < 1e-6