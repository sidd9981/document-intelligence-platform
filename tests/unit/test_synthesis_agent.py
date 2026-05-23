"""
Unit tests for the synthesis agent.

The LLM complete() function is mocked so we test prompt building,
position bias reordering, citation extraction, and error handling
without a running Ollama instance.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, patch

import pytest

from finsight.agents.synthesis_agent import (
    SynthesisAgent,
    _build_context,
    _build_prompt,
    _extract_citations,
    _reorder_for_position_bias,
)
from finsight.models.base import Chunk, ChunkMetadata
from finsight.models.graph import GraphResult
from finsight.models.tenant import TenantConfig
import tiktoken
from finsight.services import llm


@pytest.fixture(autouse=True)
def init_tokenizer():
    llm._tokenizer = tiktoken.get_encoding("cl100k_base")
    yield
    llm._tokenizer = None


def make_tenant_config() -> TenantConfig:
    return TenantConfig(
        team_id="analysis",
        daily_token_budget=2_000_000,
        max_context_tokens=64_000,
        max_output_tokens=500,
        requests_per_minute=60,
        priority=1,
        allowed_models=["large"],
        retrieval_k=10,
        data_scopes=["public", "analysis"],
    )


def make_chunk(chunk_id: str, score: float = 0.8) -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-001",
        content=f"Apple depends on TSMC for chip manufacturing. chunk {chunk_id}",
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


def test_reorder_for_position_bias_puts_best_first():
    chunks = [make_chunk("a", 0.9), make_chunk("b", 0.8), make_chunk("c", 0.7)]
    result = _reorder_for_position_bias(chunks)
    assert result[0].chunk_id == "a"


def test_reorder_for_position_bias_puts_second_best_last():
    chunks = [make_chunk("a", 0.9), make_chunk("b", 0.8), make_chunk("c", 0.7)]
    result = _reorder_for_position_bias(chunks)
    assert result[-1].chunk_id == "b"


def test_reorder_for_position_bias_handles_two_chunks():
    chunks = [make_chunk("a"), make_chunk("b")]
    result = _reorder_for_position_bias(chunks)
    assert len(result) == 2


def test_reorder_for_position_bias_handles_single_chunk():
    chunks = [make_chunk("a")]
    result = _reorder_for_position_bias(chunks)
    assert result[0].chunk_id == "a"


def test_build_context_includes_chunk_ids():
    chunks = [make_chunk("abc123" + "0" * 26)]
    context = _build_context(chunks, None)
    assert "abc123" in context


def test_build_context_includes_graph_entities():
    from finsight.models.graph import EntityNode
    graph_result = GraphResult(
        entities=[EntityNode(id="0000320193", cik="0000320193", name="Apple Inc.", entity_type="Company")],
    )
    chunks = [make_chunk("a" * 32)]
    context = _build_context(chunks, graph_result)
    assert "Apple Inc." in context


def test_build_prompt_contains_query_and_context():
    prompt = _build_prompt("what are the risks", "some context here")
    assert "what are the risks" in prompt
    assert "some context here" in prompt


def test_extract_citations_finds_chunk_ids():
    chunk_id = "a" * 32
    chunks = [make_chunk(chunk_id)]
    answer = f"Apple relies on TSMC for chips [chunk_id: {chunk_id}]."
    citations = _extract_citations(answer, chunks)
    assert len(citations) == 1
    assert citations[0].source_chunk_id == chunk_id


def test_extract_citations_ignores_unknown_ids():
    chunks = [make_chunk("a" * 32)]
    answer = "Some claim [chunk_id: " + "b" * 32 + "]."
    citations = _extract_citations(answer, chunks)
    assert citations == []


def test_extract_citations_deduplicates():
    chunk_id = "a" * 32
    chunks = [make_chunk(chunk_id)]
    answer = f"First claim [chunk_id: {chunk_id}]. Second claim [chunk_id: {chunk_id}]."
    citations = _extract_citations(answer, chunks)
    assert len(citations) == 1


async def test_synthesize_returns_synthesis_result():
    agent = SynthesisAgent()
    chunks = [make_chunk("a" * 32)]

    with patch("finsight.agents.synthesis_agent.complete", new=AsyncMock(return_value=("answer text", 100, 50))):
        result = await agent.synthesize(
            query="what are apple risks",
            chunks=chunks,
            graph_result=None,
            tenant_config=make_tenant_config(),
            trace_id="trace-001",
        )

    assert result.answer == "answer text"
    assert result.tokens_used == 150
    assert result.prompt_version == "synthesis_v1"


async def test_synthesize_never_raises():
    agent = SynthesisAgent()

    with patch("finsight.agents.synthesis_agent.complete", new=AsyncMock(side_effect=RuntimeError("ollama down"))):
        result = await agent.synthesize(
            query="query",
            chunks=[make_chunk("a" * 32)],
            graph_result=None,
            tenant_config=make_tenant_config(),
            trace_id="trace-001",
        )

    assert result.answer == ""
    assert result.tokens_used == 0