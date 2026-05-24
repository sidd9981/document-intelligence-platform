"""
Chaos engineering tests.

Verify that the system degrades gracefully when individual services
fail. Every test follows the same pattern:
    1. Confirm the system works normally
    2. Break something
    3. Confirm the system returns a degraded but valid response
    4. Confirm no unhandled exceptions escape

These tests require all services running. Run with:
    pytest tests/chaos/test_chaos.py -v -m chaos
"""

from __future__ import annotations

import asyncio
from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finsight.models.base import AgentError, Chunk, ChunkMetadata
from finsight.models.tenant import TenantConfig
from finsight.services.circuit_breaker import CircuitBreaker

pytestmark = pytest.mark.chaos


def make_tenant_config(team_id: str = "ops") -> TenantConfig:
    return TenantConfig(
        team_id=team_id,
        daily_token_budget=200_000,
        max_context_tokens=8_000,
        max_output_tokens=500,
        requests_per_minute=20,
        priority=3,
        allowed_models=["small"],
        retrieval_k=5,
        data_scopes=["public"],
    )


def make_chunk(chunk_id: str = "abc", score: float = 0.8) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc1",
        content="Apple depends on TSMC for chip manufacturing.",
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


@pytest.mark.asyncio
async def test_retrieval_agent_survives_qdrant_down():
    """When Qdrant is down the retrieval agent must return empty chunks, not raise."""
    from finsight.agents.retrieval_agent import RetrievalAgent

    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)

    agent = RetrievalAgent(redis_client=mock_redis)

    with patch(
        "finsight.agents.retrieval_agent.run_hybrid_search",
        new=AsyncMock(side_effect=ConnectionError("qdrant connection refused")),
    ):
        result = await agent.retrieve("Apple risk factors", make_tenant_config(), "trace-001")

    assert result.chunks == []
    assert len(result.errors) > 0
    assert result.errors[0].error_type == "service_unavailable"


@pytest.mark.asyncio
async def test_graph_agent_survives_neo4j_down():
    """When Neo4j is down the graph agent must return fallback=True, not raise."""
    from finsight.agents.graph_agent import GraphAgent

    mock_driver = MagicMock()
    mock_driver.session.side_effect = ConnectionError("neo4j connection refused")

    agent = GraphAgent(driver=mock_driver)

    result = await agent.query(["AAPL"], make_tenant_config(), "trace-001")

    assert result.fallback is True
    assert result.entities == []
    assert len(result.errors) > 0


@pytest.mark.asyncio
async def test_circuit_breaker_opens_after_repeated_qdrant_failures():
    """After threshold failures the breaker opens and subsequent calls fail fast."""
    from finsight.agents.retrieval_agent import RetrievalAgent
    from finsight.services.circuit_breaker import CircuitState

    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)

    breaker = CircuitBreaker(name="qdrant_chaos", failure_threshold=3, recovery_timeout=9999)
    agent = RetrievalAgent(redis_client=mock_redis, breaker=breaker)

    with patch(
        "finsight.agents.retrieval_agent.run_hybrid_search",
        new=AsyncMock(side_effect=ConnectionError("qdrant down")),
    ):
        for _ in range(3):
            await agent.retrieve("query", make_tenant_config(), "trace-001")

    assert breaker.state == CircuitState.OPEN

    with patch(
        "finsight.agents.retrieval_agent.run_hybrid_search",
        new=AsyncMock(return_value=[make_chunk()]),
    ):
        result = await agent.retrieve("query", make_tenant_config(), "trace-002")

    assert result.chunks == []
    assert result.errors[0].error_type == "service_unavailable"


@pytest.mark.asyncio
async def test_circuit_breaker_recovers_after_timeout():
    from finsight.agents.retrieval_agent import RetrievalAgent
    from finsight.services.circuit_breaker import CircuitState

    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(return_value=None)

    breaker = CircuitBreaker(name="qdrant_recover", failure_threshold=2, recovery_timeout=0.0)
    agent = RetrievalAgent(redis_client=mock_redis, breaker=breaker)

    with patch(
        "finsight.agents.retrieval_agent.run_hybrid_search",
        new=AsyncMock(side_effect=ConnectionError("qdrant down")),
    ):
        for _ in range(2):
            await agent.retrieve("query", make_tenant_config(), "trace-001")

    assert breaker.state == CircuitState.OPEN

    with (
        patch(
            "finsight.agents.retrieval_agent.run_hybrid_search",
            new=AsyncMock(return_value=[make_chunk()]),
        ),
        patch(
            "finsight.agents.retrieval_agent.rerank",
            return_value=[make_chunk()],
        ),
    ):
        result = await agent.retrieve("query", make_tenant_config(), "trace-002")

    assert breaker.state == CircuitState.CLOSED
    assert len(result.chunks) > 0

@pytest.mark.asyncio
async def test_redis_down_does_not_block_retrieval():
    """When Redis is down cache misses gracefully and retrieval continues."""
    from finsight.agents.retrieval_agent import RetrievalAgent

    mock_redis = AsyncMock()
    mock_redis.get = AsyncMock(side_effect=ConnectionError("redis down"))

    agent = RetrievalAgent(redis_client=mock_redis)

    with patch(
        "finsight.agents.retrieval_agent.run_hybrid_search",
        new=AsyncMock(return_value=[make_chunk()]),
    ):
        with patch("finsight.agents.retrieval_agent.rerank", return_value=[make_chunk()]):
            result = await agent.retrieve("Apple risk", make_tenant_config(), "trace-001")

    assert len(result.chunks) > 0
    assert result.cache_hit is False


@pytest.mark.asyncio
async def test_orchestrator_continues_without_graph_context():
    """When graph agent returns fallback the orchestrator still produces an answer."""
    from finsight.agents.orchestrator import Orchestrator
    from finsight.agents.retrieval_agent import RetrievalAgent
    from finsight.agents.graph_agent import GraphAgent
    from finsight.agents.synthesis_agent import SynthesisAgent
    from finsight.models.graph import GraphResult
    from finsight.models.retrieval import RetrievalResult
    from finsight.models.synthesis import SynthesisResult, Citation

    mock_retrieval = AsyncMock()
    mock_retrieval.retrieve = AsyncMock(return_value=RetrievalResult(
        chunks=[make_chunk()],
        cache_hit=False,
        retrieval_method="hybrid",
        total_tokens=10,
        latency_ms=100.0,
    ))
    mock_retrieval.write_cache = AsyncMock()
    mock_retrieval._redis = AsyncMock()

    mock_graph = AsyncMock()
    mock_graph.query = AsyncMock(return_value=GraphResult(fallback=True))

    mock_synthesis = AsyncMock()
    mock_synthesis.synthesize = AsyncMock(return_value=SynthesisResult(
        answer="Apple depends on TSMC.",
        faithfulness_score=0.9,
        tokens_used=100,
        model_used="llama3.2:3b",
        prompt_version="synthesis_v1",
        latency_ms=1000.0,
    ))

    orchestrator = Orchestrator(
        retrieval_agent=mock_retrieval,
        graph_agent=mock_graph,
        synthesis_agent=mock_synthesis,
    )

    result = await orchestrator.run(
        query="What are Apple risk factors?",
        tenant_config=make_tenant_config(),
    )

    assert result.answer != ""
    assert result.warning is None or "graph" not in result.warning.lower()


@pytest.mark.asyncio
async def test_no_context_returns_structured_error():
    """When retrieval returns nothing the orchestrator must not call the LLM."""
    from finsight.agents.orchestrator import Orchestrator
    from finsight.agents.retrieval_agent import RetrievalAgent
    from finsight.agents.graph_agent import GraphAgent
    from finsight.agents.synthesis_agent import SynthesisAgent
    from finsight.models.graph import GraphResult
    from finsight.models.retrieval import RetrievalResult

    mock_retrieval = AsyncMock()
    mock_retrieval.retrieve = AsyncMock(return_value=RetrievalResult(
        chunks=[],
        cache_hit=False,
        retrieval_method="hybrid",
        total_tokens=0,
        latency_ms=100.0,
    ))
    mock_retrieval.write_cache = AsyncMock()
    mock_retrieval._redis = AsyncMock()

    mock_graph = AsyncMock()
    mock_graph.query = AsyncMock(return_value=GraphResult())

    mock_synthesis = AsyncMock()

    orchestrator = Orchestrator(
        retrieval_agent=mock_retrieval,
        graph_agent=mock_graph,
        synthesis_agent=mock_synthesis,
    )

    result = await orchestrator.run(
        query="What are Apple risk factors?",
        tenant_config=make_tenant_config(),
    )

    assert result.answer != ""
    mock_synthesis.synthesize.assert_not_called()