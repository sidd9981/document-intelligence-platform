"""
Unit tests for the orchestrator.

All agents are mocked. We test the state machine routing logic —
correct nodes fire, conditional edges route correctly, error states
are handled gracefully.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock

import pytest

from finsight.agents.orchestrator import Orchestrator, _initial_state
from finsight.models.base import AgentError, Chunk, ChunkMetadata
from finsight.models.graph import GraphResult
from finsight.models.retrieval import RetrievalResult
from finsight.models.synthesis import SynthesisResult
from finsight.models.tenant import TenantConfig


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


def make_chunk(chunk_id: str = "a" * 32, score: float = 0.8) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-001",
        content="Apple depends on TSMC for chip manufacturing and faces supply chain risks.",
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


def make_retrieval_result(chunks: list[Chunk] | None = None) -> RetrievalResult:
    if chunks is None:
        chunks = [make_chunk()]
    return RetrievalResult(
        chunks=chunks,
        cache_hit=False,
        retrieval_method="hybrid",
        total_tokens=50,
        latency_ms=100.0,
    )


def make_synthesis_result(answer: str = "Apple faces supply chain risks.") -> SynthesisResult:
    return SynthesisResult(
        answer=answer,
        tokens_used=150,
        model_used="llama3.2:3b",
        prompt_version="synthesis_v1",
        latency_ms=1200.0,
        faithfulness_score=0.9,
    )


@pytest.fixture
def mock_retrieval_agent() -> AsyncMock:
    agent = AsyncMock()
    agent.retrieve = AsyncMock(return_value=make_retrieval_result())
    agent.write_cache = AsyncMock()
    return agent


@pytest.fixture
def mock_graph_agent() -> AsyncMock:
    agent = AsyncMock()
    agent.query = AsyncMock(return_value=GraphResult())
    return agent


@pytest.fixture
def mock_synthesis_agent() -> AsyncMock:
    agent = AsyncMock()
    agent.synthesize = AsyncMock(return_value=make_synthesis_result())
    return agent


@pytest.fixture
def orchestrator(
    mock_retrieval_agent: AsyncMock,
    mock_graph_agent: AsyncMock,
    mock_synthesis_agent: AsyncMock,
) -> Orchestrator:
    return Orchestrator(
        retrieval_agent=mock_retrieval_agent,
        graph_agent=mock_graph_agent,
        synthesis_agent=mock_synthesis_agent,
    )


async def test_run_returns_query_response(orchestrator: Orchestrator) -> None:
    from finsight.services import llm
    import tiktoken
    llm._tokenizer = tiktoken.get_encoding("cl100k_base")

    from unittest.mock import patch, AsyncMock as AM
    with patch("finsight.harness.output_harness.complete", new=AM(return_value=("SCORE: 0.95\nUNSUPPORTED: NONE", 50, 20))):
        result = await orchestrator.run("what are apple risks", make_tenant_config())

    assert result.answer
    assert result.trace_id
    llm._tokenizer = None


async def test_run_returns_error_response_when_no_chunks() -> None:
    retrieval_agent = AsyncMock()
    retrieval_agent.retrieve = AsyncMock(return_value=make_retrieval_result(chunks=[]))

    graph_agent = AsyncMock()
    graph_agent.query = AsyncMock(return_value=GraphResult())

    synthesis_agent = AsyncMock()
    synthesis_agent.synthesize = AsyncMock(return_value=make_synthesis_result())

    orc = Orchestrator(
        retrieval_agent=retrieval_agent,
        graph_agent=graph_agent,
        synthesis_agent=synthesis_agent,
    )

    result = await orc.run("query with no results", make_tenant_config())
    assert "No relevant context" in result.answer

async def test_run_never_raises(orchestrator: Orchestrator) -> None:
    from finsight.services import llm
    import tiktoken
    llm._tokenizer = tiktoken.get_encoding("cl100k_base")

    from unittest.mock import patch, AsyncMock as AM
    with patch("finsight.harness.output_harness.complete", new=AM(side_effect=RuntimeError("llm down"))):
        result = await orchestrator.run("query", make_tenant_config())

    assert result is not None
    llm._tokenizer = None


def test_classify_intent_factual():
    state = _initial_state("what is apple revenue", make_tenant_config(), "t-001")
    assert state["intent"] == "factual"


def test_initial_state_has_correct_defaults():
    config = make_tenant_config()
    state = _initial_state("test query", config, "trace-001")
    assert state["retry_count"] == 0
    assert state["tokens_used"] == 0
    assert state["cache_hit"] is False
    assert state["errors"] == []
    assert state["final_response"] is None

