"""
Unit tests for the synthesis agent.

complete() is mocked so prompt building, numbered citations, model tier
resolution and error handling are tested without a running Ollama.
Chunk order is decided by the input harness, so these tests also check
that the agent uses chunks in the order it receives them.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, patch

import pytest
import tiktoken

from finsight.agents.synthesis_agent import (
    PROMPT_VERSION,
    SynthesisAgent,
    _build_context,
    _build_prompt,
    _extract_citations,
)
from finsight.models.base import Chunk, ChunkMetadata
from finsight.models.graph import EntityNode, GraphResult
from finsight.models.tenant import TenantConfig
from finsight.services import llm
from finsight.services.reranker import relevance_probability


@pytest.fixture(autouse=True)
def init_tokenizer():
    llm._tokenizer = tiktoken.get_encoding("cl100k_base")
    yield
    llm._tokenizer = None


def make_tenant_config(allowed_models: list[str] | None = None) -> TenantConfig:
    return TenantConfig(
        team_id="analysis",
        daily_token_budget=2_000_000,
        max_context_tokens=8_000,
        max_output_tokens=500,
        requests_per_minute=60,
        priority=1,
        allowed_models=allowed_models or ["small"],
        retrieval_k=10,
        data_scopes=["public", "analysis"],
    )


def make_chunk(chunk_id: str, score: float = 2.0) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id=f"doc-{chunk_id}",
        content=f"Apple obtains certain components from single or limited sources. chunk {chunk_id}",
        score=score,
        token_count=50,
        metadata=ChunkMetadata(
            doc_id=f"doc-{chunk_id}",
            ticker="AAPL",
            company_name="Apple Inc.",
            filing_type="10-K",
            filing_date=date(2024, 11, 1),
            section="Item 1A",
            chunk_index=0,
            token_count=50,
            embedding_model="nomic-embed-text",
            scopes=["public"],
        ),
    )


def test_build_context_numbers_sources_in_order():
    context = _build_context([make_chunk("a"), make_chunk("b")], None)
    assert context.index("[1]") < context.index("chunk a")
    assert context.index("[2]") < context.index("chunk b")


def test_build_context_includes_source_metadata():
    context = _build_context([make_chunk("a")], None)
    assert "Apple Inc." in context
    assert "10-K" in context
    assert "Item 1A" in context


def test_build_context_includes_graph_entities():
    graph_result = GraphResult(
        entities=[EntityNode(id="0000320193", cik="0000320193", name="Apple Inc.", entity_type="Company")],
    )
    context = _build_context([make_chunk("a")], graph_result)
    assert "Graph context" in context
    assert "0000320193" in context


def test_build_context_omits_empty_graph_section():
    assert "Graph context" not in _build_context([make_chunk("a")], GraphResult())


def test_build_prompt_contains_query_and_context():
    prompt = _build_prompt("what are the risks", "some context here")
    assert "what are the risks" in prompt
    assert "some context here" in prompt


def test_extract_citations_maps_number_to_chunk():
    chunks = [make_chunk("a"), make_chunk("b")]
    citations = _extract_citations("Apple relies on limited sources [2].", chunks)
    assert len(citations) == 1
    assert citations[0].source_chunk_id == "b"
    assert citations[0].source_doc_id == "doc-b"


def test_extract_citations_handles_adjacent_markers():
    chunks = [make_chunk("a"), make_chunk("b")]
    citations = _extract_citations("Both sources agree [1][2].", chunks)
    assert [c.source_chunk_id for c in citations] == ["a", "b"]


def test_extract_citations_ignores_out_of_range_numbers():
    citations = _extract_citations("Some claim [0] and [5].", [make_chunk("a")])
    assert citations == []


def test_extract_citations_deduplicates():
    citations = _extract_citations("First claim [1]. Second claim [1].", [make_chunk("a")])
    assert len(citations) == 1


def test_extract_citations_captures_claim_text():
    citations = _extract_citations("Apple relies on limited sources [1].", [make_chunk("a")])
    assert "limited sources" in citations[0].claim


def test_extract_citations_confidence_is_probability():
    citations = _extract_citations("Claim [1].", [make_chunk("a", score=2.0)])
    assert citations[0].confidence == pytest.approx(relevance_probability(2.0))
    assert 0.0 < citations[0].confidence < 1.0


async def test_synthesize_returns_synthesis_result():
    agent = SynthesisAgent()
    mock = AsyncMock(return_value=("Apple relies on limited sources [1].", 100, 50))

    with patch("finsight.agents.synthesis_agent.complete", new=mock):
        result = await agent.synthesize(
            query="what are apple supply risks",
            chunks=[make_chunk("a")],
            graph_result=None,
            tenant_config=make_tenant_config(),
            trace_id="trace-001",
        )

    assert result.answer == "Apple relies on limited sources [1]."
    assert result.tokens_used == 150
    assert result.prompt_version == PROMPT_VERSION
    assert [c.source_chunk_id for c in result.citations] == ["a"]


async def test_synthesize_resolves_model_tier():
    agent = SynthesisAgent()
    mock = AsyncMock(return_value=("answer", 10, 5))

    with patch("finsight.agents.synthesis_agent.complete", new=mock):
        result = await agent.synthesize(
            query="query",
            chunks=[make_chunk("a")],
            graph_result=None,
            tenant_config=make_tenant_config(["medium", "small"]),
            trace_id="trace-001",
        )

    assert mock.call_args.kwargs["model"] == llm.MODEL_TIERS["medium"]
    assert result.model_used == llm.MODEL_TIERS["medium"]


async def test_synthesize_uses_chunks_in_given_order():
    agent = SynthesisAgent()
    mock = AsyncMock(return_value=("Claim [1].", 10, 5))
    chunks = [make_chunk("x", score=-1.0), make_chunk("y", score=5.0)]

    with patch("finsight.agents.synthesis_agent.complete", new=mock):
        result = await agent.synthesize(
            query="query",
            chunks=chunks,
            graph_result=None,
            tenant_config=make_tenant_config(),
            trace_id="trace-001",
        )

    prompt = mock.call_args.kwargs["prompt"]
    assert prompt.index("chunk x") < prompt.index("chunk y")
    assert result.citations[0].source_chunk_id == "x"


async def test_synthesize_never_raises():
    agent = SynthesisAgent()

    with patch("finsight.agents.synthesis_agent.complete", new=AsyncMock(side_effect=RuntimeError("ollama down"))):
        result = await agent.synthesize(
            query="query",
            chunks=[make_chunk("a")],
            graph_result=None,
            tenant_config=make_tenant_config(),
            trace_id="trace-001",
        )

    assert result.answer == ""
    assert result.tokens_used == 0