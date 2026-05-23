"""
Unit tests for the retrieval agent.

run_hybrid_search and rerank are mocked so we test the agent's
orchestration logic without needing live services.
"""

from __future__ import annotations

import json
import hashlib
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finsight.agents.retrieval_agent import RetrievalAgent
from finsight.models.base import Chunk, ChunkMetadata
from finsight.models.tenant import TenantConfig


def make_tenant_config(team_id: str = "ops", retrieval_k: int = 5) -> TenantConfig:
    return TenantConfig(
        team_id=team_id,
        daily_token_budget=200_000,
        max_context_tokens=8_000,
        max_output_tokens=500,
        requests_per_minute=20,
        priority=3,
        allowed_models=["small"],
        retrieval_k=retrieval_k,
        data_scopes=["public"],
    )


def make_chunk(chunk_id: str, score: float = 0.8) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-001",
        content=f"content for {chunk_id}",
        score=score,
        token_count=100,
        metadata=ChunkMetadata(
            doc_id="doc-001",
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_type="10-K",
            filing_date=date(2023, 10, 27),
            section="Item 1A",
            chunk_index=0,
            token_count=100,
            embedding_model="nomic-embed-text",
            scopes=["public"],
        ),
    )


@pytest.fixture
def mock_redis() -> AsyncMock:
    client = AsyncMock()
    client.get = AsyncMock(return_value=None)
    client.set = AsyncMock()
    return client


@pytest.fixture
def agent(mock_redis: AsyncMock) -> RetrievalAgent:
    return RetrievalAgent(redis_client=mock_redis)


async def test_retrieve_returns_retrieval_result(agent: RetrievalAgent) -> None:
    chunks = [make_chunk("a"), make_chunk("b")]

    with patch("finsight.agents.retrieval_agent.run_hybrid_search", new=AsyncMock(return_value=chunks)):
        with patch("finsight.agents.retrieval_agent.rerank", return_value=chunks):
            result = await agent.retrieve("what are apple risk factors", make_tenant_config(), "trace-001")

    assert result.chunks == chunks
    assert result.cache_hit is False
    assert result.retrieval_method == "hybrid"


async def test_retrieve_returns_cache_hit_when_cached(
    agent: RetrievalAgent,
    mock_redis: AsyncMock,
) -> None:
    chunks = [make_chunk("cached-chunk")]
    serialized = json.dumps([c.model_dump(mode="json") for c in chunks])
    mock_redis.get.return_value = serialized

    with patch("finsight.agents.retrieval_agent.run_hybrid_search", new=AsyncMock()) as mock_search:
        result = await agent.retrieve("query", make_tenant_config(), "trace-001")

    assert result.cache_hit is True
    assert result.retrieval_method == "cached"
    mock_search.assert_not_called()


async def test_retrieve_returns_empty_result_on_no_chunks(agent: RetrievalAgent) -> None:
    with patch("finsight.agents.retrieval_agent.run_hybrid_search", new=AsyncMock(return_value=[])):
        result = await agent.retrieve("query", make_tenant_config(), "trace-001")

    assert result.chunks == []
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "empty_result"


async def test_retrieve_never_raises_on_exception(agent: RetrievalAgent) -> None:
    with patch("finsight.agents.retrieval_agent.run_hybrid_search", new=AsyncMock(side_effect=RuntimeError("qdrant down"))):
        result = await agent.retrieve("query", make_tenant_config(), "trace-001")

    assert result.chunks == []
    assert result.errors[0].error_type == "service_unavailable"


async def test_retrieve_respects_retrieval_k(agent: RetrievalAgent) -> None:
    chunks = [make_chunk(str(i)) for i in range(10)]
    top_3 = chunks[:3]

    with patch("finsight.agents.retrieval_agent.run_hybrid_search", new=AsyncMock(return_value=chunks)):
        with patch("finsight.agents.retrieval_agent.rerank", return_value=top_3) as mock_rerank:
            await agent.retrieve("query", make_tenant_config(retrieval_k=3), "trace-001")

    mock_rerank.assert_called_once()
    assert mock_rerank.call_args[1]["top_k"] == 3


async def test_write_cache_stores_result(
    agent: RetrievalAgent,
    mock_redis: AsyncMock,
) -> None:
    chunks = [make_chunk("a")]
    await agent.write_cache("query", "ops", chunks, ttl_seconds=3600)
    mock_redis.set.assert_called_once()
    call_args = mock_redis.set.call_args
    assert call_args[1]["ex"] == 3600


async def test_write_cache_does_not_raise_on_redis_error(
    agent: RetrievalAgent,
    mock_redis: AsyncMock,
) -> None:
    mock_redis.set.side_effect = RuntimeError("redis down")
    chunks = [make_chunk("a")]
    await agent.write_cache("query", "ops", chunks)