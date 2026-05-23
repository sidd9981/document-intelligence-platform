"""
Unit tests for the cross-encoder reranker.

The actual model is not loaded. We mock torch and transformers
and verify the reranking logic in isolation.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch
from datetime import date

import pytest
import torch

from finsight.models.base import Chunk, ChunkMetadata
from finsight.services import reranker


@pytest.fixture(autouse=True)
def reset_reranker():
    reranker._tokenizer = None
    reranker._model = None
    yield
    reranker._tokenizer = None
    reranker._model = None


def make_chunk(chunk_id: str, score: float = 0.5) -> Chunk:
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


def make_mock_model(scores: list[float]) -> MagicMock:
    model = MagicMock()
    output = MagicMock()
    output.logits = torch.tensor(scores)
    model.return_value = output
    return model


def make_mock_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.zeros(2, 10, dtype=torch.long),
        "attention_mask": torch.ones(2, 10, dtype=torch.long),
    }
    return tokenizer


def test_get_tokenizer_raises_before_init():
    with pytest.raises(RuntimeError, match="not initialized"):
        reranker.get_tokenizer()


def test_get_model_raises_before_init():
    with pytest.raises(RuntimeError, match="not initialized"):
        reranker.get_model()


def test_rerank_raises_before_init():
    with pytest.raises(RuntimeError, match="not initialized"):
        reranker.rerank("query", [make_chunk("a")], top_k=1)


def test_rerank_returns_empty_for_empty_input():
    reranker._tokenizer = make_mock_tokenizer()
    reranker._model = make_mock_model([])
    result = reranker.rerank("query", [], top_k=5)
    assert result == []


def test_rerank_sorts_by_cross_encoder_score():
    reranker._tokenizer = make_mock_tokenizer()
    reranker._model = make_mock_model([0.2, 0.9, 0.5])

    chunks = [make_chunk("a"), make_chunk("b"), make_chunk("c")]
    result = reranker.rerank("what are the supply chain risks", chunks, top_k=3)

    assert result[0].chunk_id == "b"
    assert result[1].chunk_id == "c"
    assert result[2].chunk_id == "a"


def test_rerank_respects_top_k():
    reranker._tokenizer = make_mock_tokenizer()
    reranker._model = make_mock_model([0.9, 0.8, 0.7, 0.6, 0.5])

    chunks = [make_chunk(str(i)) for i in range(5)]
    result = reranker.rerank("query", chunks, top_k=3)
    assert len(result) == 3


def test_rerank_replaces_original_scores():
    reranker._tokenizer = make_mock_tokenizer()
    reranker._model = make_mock_model([0.75, 0.25])

    chunks = [make_chunk("a", score=0.99), make_chunk("b", score=0.99)]
    result = reranker.rerank("query", chunks, top_k=2)

    assert abs(result[0].score - 0.75) < 1e-5
    assert abs(result[1].score - 0.25) < 1e-5


def test_close_reranker_resets_state():
    reranker._tokenizer = MagicMock()
    reranker._model = MagicMock()
    reranker.close_reranker()
    assert reranker._tokenizer is None
    assert reranker._model is None