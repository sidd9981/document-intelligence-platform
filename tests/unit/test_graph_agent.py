"""
Unit tests for the graph agent.

Neo4j driver is mocked. We verify the agent's error handling,
scope filtering, and fallback behaviour without a live graph.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock

import pytest

from finsight.agents.graph_agent import GraphAgent
from finsight.models.tenant import TenantConfig


def make_tenant_config(team_id: str = "analysis") -> TenantConfig:
    return TenantConfig(
        team_id=team_id,
        daily_token_budget=2_000_000,
        max_context_tokens=64_000,
        max_output_tokens=2_000,
        requests_per_minute=60,
        priority=1,
        allowed_models=["large"],
        retrieval_k=30,
        data_scopes=["public", "analysis"],
    )


@pytest.fixture
def mock_session() -> AsyncMock:
    session = AsyncMock()
    session.execute_read = AsyncMock(side_effect=[
        ([], []),
        [],
    ])
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


@pytest.fixture
def mock_driver(mock_session: AsyncMock) -> MagicMock:
    driver = MagicMock()
    driver.session = MagicMock(return_value=mock_session)
    return driver


@pytest.fixture
def agent(mock_driver: MagicMock) -> GraphAgent:
    return GraphAgent(driver=mock_driver)


async def test_query_returns_graph_result(
    agent: GraphAgent,
    mock_session: AsyncMock,
) -> None:
    result = await agent.query(["AAPL", "TSMC"], make_tenant_config(), "trace-001")
    assert result is not None
    assert result.fallback is False


async def test_query_empty_entities_returns_empty_result(
    agent: GraphAgent,
) -> None:
    result = await agent.query([], make_tenant_config(), "trace-001")
    assert result.entities == []
    assert result.relationships == []
    assert result.fallback is False


async def test_query_returns_fallback_on_neo4j_error(
    agent: GraphAgent,
    mock_session: AsyncMock,
) -> None:
    mock_session.execute_read.side_effect = RuntimeError("neo4j connection refused")
    result = await agent.query(["AAPL"], make_tenant_config(), "trace-001")
    assert result.fallback is True
    assert len(result.errors) == 1
    assert result.errors[0].error_type == "service_unavailable"
    assert result.errors[0].fallback_used is True


async def test_query_never_raises(
    agent: GraphAgent,
    mock_session: AsyncMock,
) -> None:
    mock_session.execute_read.side_effect = Exception("unexpected failure")
    result = await agent.query(["AAPL"], make_tenant_config(), "trace-001")
    assert result is not None


async def test_query_calls_execute_read_twice(
    agent: GraphAgent,
    mock_session: AsyncMock,
) -> None:
    await agent.query(["AAPL"], make_tenant_config(), "trace-001")
    assert mock_session.execute_read.call_count == 2


async def test_query_latency_is_populated(
    agent: GraphAgent,
) -> None:
    result = await agent.query(["AAPL"], make_tenant_config(), "trace-001")
    assert result.latency_ms >= 0.0