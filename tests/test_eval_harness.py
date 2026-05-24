"""
Unit tests for the eval harness.

No Langfuse connection needed. Mocks the client so we test
sampling logic, metric computation, and logging calls only.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import MagicMock, patch

import pytest

from finsight.harness.eval_harness import (
    _score_answer_relevance,
    _score_context_recall,
    maybe_run_eval,
)
from finsight.models.base import Chunk, ChunkMetadata
from finsight.models.synthesis import SynthesisResult


def make_chunk(score: float = 0.8) -> Chunk:
    return Chunk(
        chunk_id="abc",
        doc_id="doc1",
        content="Apple depends on TSMC.",
        score=score,
        token_count=10,
        metadata=ChunkMetadata(
            doc_id="doc1",
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_type="10-K",
            filing_date=date(2023, 10, 27),
            section="Risk Factors",
            chunk_index=0,
            token_count=10,
            embedding_model="nomic-embed-text",
            scopes=["public"],
        ),
    )


def make_result(**kwargs) -> SynthesisResult:
    defaults = dict(
        answer="Apple revenue was $383B.",
        faithfulness_score=0.91,
        tokens_used=100,
        model_used="llama3.2:3b",
        prompt_version="synthesis_v1",
        latency_ms=1200.0,
    )
    defaults.update(kwargs)
    return SynthesisResult(**defaults)


def test_score_answer_relevance_full_match():
    score = _score_answer_relevance("Apple revenue", "Apple revenue was $383B.")
    assert score == 1.0


def test_score_answer_relevance_no_match():
    score = _score_answer_relevance("Apple revenue", "Microsoft reported strong earnings.")
    assert score == 0.0


def test_score_answer_relevance_partial_match():
    score = _score_answer_relevance("Apple TSMC supply chain", "Apple depends on TSMC.")
    assert 0.0 < score < 1.0


def test_score_answer_relevance_empty_query():
    score = _score_answer_relevance("", "some answer")
    assert score == 0.0


def test_score_context_recall_all_above_threshold():
    chunks = [make_chunk(0.8), make_chunk(0.9), make_chunk(0.7)]
    assert _score_context_recall(chunks) == 1.0


def test_score_context_recall_none_above_threshold():
    chunks = [make_chunk(0.3), make_chunk(0.2)]
    assert _score_context_recall(chunks) == 0.0


def test_score_context_recall_empty():
    assert _score_context_recall([]) == 0.0


def test_score_context_recall_mixed():
    chunks = [make_chunk(0.8), make_chunk(0.3)]
    assert _score_context_recall(chunks) == 0.5


@pytest.mark.asyncio
async def test_maybe_run_eval_skips_when_not_sampled():
    """When random() > sample rate the harness must not call Langfuse."""
    with (
        patch("finsight.harness.eval_harness.random.random", return_value=0.99),
        patch("finsight.harness.eval_harness.get_langfuse") as mock_lf,
    ):
        await maybe_run_eval(
            query="What is Apple revenue?",
            result=make_result(),
            chunks=[make_chunk()],
            trace_id="trace-001",
            team_id="ops",
        )
        mock_lf.assert_not_called()


@pytest.mark.asyncio
async def test_maybe_run_eval_logs_when_sampled():
    """When sampled the harness must create a Langfuse trace."""
    mock_trace = MagicMock()
    mock_lf = MagicMock()
    mock_lf.trace.return_value = mock_trace

    with (
        patch("finsight.harness.eval_harness.random.random", return_value=0.01),
        patch("finsight.harness.eval_harness.get_langfuse", return_value=mock_lf),
    ):
        await maybe_run_eval(
            query="What is Apple revenue?",
            result=make_result(),
            chunks=[make_chunk()],
            trace_id="trace-001",
            team_id="ops",
        )
        mock_lf.trace.assert_called_once()
        mock_trace.score.assert_called()
        mock_lf.flush.assert_called_once()


@pytest.mark.asyncio
async def test_maybe_run_eval_never_raises_on_langfuse_error():
    """Langfuse being down must never affect the user response."""
    with (
        patch("finsight.harness.eval_harness.random.random", return_value=0.01),
        patch("finsight.harness.eval_harness.get_langfuse", side_effect=Exception("langfuse down")),
    ):
        await maybe_run_eval(
            query="What is Apple revenue?",
            result=make_result(),
            chunks=[make_chunk()],
            trace_id="trace-001",
            team_id="ops",
        )


@pytest.mark.asyncio
async def test_maybe_run_eval_logs_three_scores():
    """Must log faithfulness, answer_relevance, and context_recall."""
    mock_trace = MagicMock()
    mock_lf = MagicMock()
    mock_lf.trace.return_value = mock_trace

    with (
        patch("finsight.harness.eval_harness.random.random", return_value=0.01),
        patch("finsight.harness.eval_harness.get_langfuse", return_value=mock_lf),
    ):
        await maybe_run_eval(
            query="Apple revenue",
            result=make_result(),
            chunks=[make_chunk()],
            trace_id="trace-001",
            team_id="ops",
        )

        score_names = [call.kwargs["name"] for call in mock_trace.score.call_args_list]
        assert "faithfulness" in score_names
        assert "answer_relevance" in score_names
        assert "context_recall" in score_names