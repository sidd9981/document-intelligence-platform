"""
Unit tests for the drift detector.

No live services needed. Injects a fake retrieval function.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock

import pytest

from finsight.mlops.drift_detector import (
    DRIFT_THRESHOLD,
    DriftReport,
    _compute_recall,
    run_drift_check,
)
from finsight.models.base import Chunk, ChunkMetadata


def make_chunk(score: float) -> Chunk:
    return Chunk(
        chunk_id="abc",
        doc_id="doc1",
        content="test content",
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


def test_compute_recall_all_above_threshold():
    chunks = [make_chunk(0.8), make_chunk(0.9), make_chunk(0.7)]
    assert _compute_recall([chunks]) == 1.0


def test_compute_recall_none_above_threshold():
    chunks = [make_chunk(0.3), make_chunk(0.2)]
    assert _compute_recall([chunks]) == 0.0


def test_compute_recall_empty_list():
    assert _compute_recall([]) == 0.0


def test_compute_recall_empty_chunks():
    assert _compute_recall([[]]) == 0.0


def test_compute_recall_mixed():
    chunks = [make_chunk(0.8), make_chunk(0.3)]
    assert _compute_recall([chunks]) == 0.5


@pytest.mark.asyncio
async def test_drift_check_no_drift():
    """When recent and baseline recall are similar, drifted must be False."""
    good_chunks = [make_chunk(0.8), make_chunk(0.9)]

    async def fake_retrieval(query, team_id):
        return good_chunks

    report = await run_drift_check(
        recent_queries=["query 1", "query 2"],
        baseline_queries=["query 3", "query 4"],
        team_id="ops",
        retrieval_fn=fake_retrieval,
    )

    assert report.drifted is False
    assert report.drift <= DRIFT_THRESHOLD


@pytest.mark.asyncio
async def test_drift_check_detects_drift():
    """When recent recall drops significantly vs baseline, drifted must be True."""
    good_chunks = [make_chunk(0.9), make_chunk(0.85)]
    bad_chunks = [make_chunk(0.2), make_chunk(0.1)]

    call_count = 0

    async def fake_retrieval(query, team_id):
        nonlocal call_count
        call_count += 1
        if call_count <= 2:
            return bad_chunks
        return good_chunks

    report = await run_drift_check(
        recent_queries=["recent 1", "recent 2"],
        baseline_queries=["baseline 1", "baseline 2"],
        team_id="ops",
        retrieval_fn=fake_retrieval,
    )

    assert report.drifted is True
    assert report.drift > DRIFT_THRESHOLD


@pytest.mark.asyncio
async def test_drift_check_handles_retrieval_failure():
    """Retrieval failures must not raise — they count as zero recall."""
    async def fake_retrieval(query, team_id):
        raise RuntimeError("qdrant down")

    report = await run_drift_check(
        recent_queries=["query 1"],
        baseline_queries=["query 2"],
        team_id="ops",
        retrieval_fn=fake_retrieval,
    )

    assert report.current_recall == 0.0
    assert report.baseline_recall == 0.0
    assert report.drifted is False


@pytest.mark.asyncio
async def test_drift_report_has_correct_sample_size():
    async def fake_retrieval(query, team_id):
        return [make_chunk(0.8)]

    report = await run_drift_check(
        recent_queries=["q1", "q2", "q3"],
        baseline_queries=["q4", "q5"],
        team_id="ops",
        retrieval_fn=fake_retrieval,
    )

    assert report.sample_size == 3


@pytest.mark.asyncio
async def test_drift_check_returns_drift_report():
    async def fake_retrieval(query, team_id):
        return [make_chunk(0.8)]

    report = await run_drift_check(
        recent_queries=["query 1"],
        baseline_queries=["query 2"],
        team_id="ops",
        retrieval_fn=fake_retrieval,
    )

    assert isinstance(report, DriftReport)
    assert report.checked_at is not None
    assert report.baseline_weeks == 4