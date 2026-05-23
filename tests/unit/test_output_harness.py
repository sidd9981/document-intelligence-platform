"""
Unit tests for the output harness.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, patch

import tiktoken
import pytest

from finsight.services import llm
from finsight.harness.output_harness import (
    FAITHFULNESS_THRESHOLD,
    _flag_hallucinations,
    _parse_faithfulness_response,
    _scrub_pii,
    run_output_harness,
)
from finsight.models.base import Chunk, ChunkMetadata
from finsight.models.synthesis import SynthesisResult


@pytest.fixture(autouse=True)
def init_tokenizer():
    llm._tokenizer = tiktoken.get_encoding("cl100k_base")
    yield
    llm._tokenizer = None


def make_chunk(chunk_id: str, content: str = "") -> Chunk:
    return Chunk(
        chunk_id=chunk_id,
        doc_id="doc-001",
        content=content or "Apple revenue was $89.5 billion in Q4 2023.",
        score=0.8,
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


def make_synthesis_result(answer: str = "Apple revenue was $89.5 billion.") -> SynthesisResult:
    return SynthesisResult(
        answer=answer,
        tokens_used=150,
        model_used="llama3.2:3b",
        prompt_version="synthesis_v1",
        latency_ms=1200.0,
    )


def test_parse_faithfulness_response_extracts_score():
    response = "SCORE: 0.92\nUNSUPPORTED: NONE"
    score, claims = _parse_faithfulness_response(response)
    assert abs(score - 0.92) < 1e-6
    assert claims == []


def test_parse_faithfulness_response_extracts_unsupported_claims():
    response = "SCORE: 0.60\nUNSUPPORTED: revenue was 100B, growth was 20%"
    score, claims = _parse_faithfulness_response(response)
    assert len(claims) == 2
    assert "revenue was 100B" in claims


def test_parse_faithfulness_response_defaults_on_bad_input():
    score, claims = _parse_faithfulness_response("completely unparseable output")
    assert score == 1.0
    assert claims == []


def test_parse_faithfulness_clamps_score_to_valid_range():
    response = "SCORE: 1.5\nUNSUPPORTED: NONE"
    score, _ = _parse_faithfulness_response(response)
    assert score <= 1.0


def test_flag_hallucinations_catches_number_not_in_context():
    chunks = [make_chunk("a", content="Revenue was 89.5 billion.")]
    flags = _flag_hallucinations("Revenue was 99.9 billion.", chunks)
    assert "99.9" in flags


def test_flag_hallucinations_no_flags_when_grounded():
    chunks = [make_chunk("a", content="Revenue was 89.5 billion.")]
    flags = _flag_hallucinations("Revenue was 89.5 billion.", chunks)
    assert flags == []


def test_scrub_pii_removes_ssn():
    text = "Employee SSN is 123-45-6789 per the filing."
    result = _scrub_pii(text)
    assert "123-45-6789" not in result
    assert "[REDACTED]" in result


def test_scrub_pii_removes_email():
    text = "Contact john.doe@example.com for details."
    result = _scrub_pii(text)
    assert "john.doe@example.com" not in result


def test_scrub_pii_leaves_clean_text_unchanged():
    text = "Apple reported strong earnings this quarter."
    assert _scrub_pii(text) == text


async def test_run_output_harness_returns_result():
    result = make_synthesis_result()
    chunks = [make_chunk("a")]

    with patch(
        "finsight.harness.output_harness.complete",
        new=AsyncMock(return_value=("SCORE: 0.95\nUNSUPPORTED: NONE", 50, 20)),
    ):
        harness_result = await run_output_harness(result, chunks, "what is apple revenue")

    assert harness_result.faithfulness_score == 0.95
    assert harness_result.low_faithfulness is False


async def test_run_output_harness_flags_low_faithfulness():
    result = make_synthesis_result()
    chunks = [make_chunk("a")]

    with patch(
        "finsight.harness.output_harness.complete",
        new=AsyncMock(return_value=("SCORE: 0.50\nUNSUPPORTED: revenue figure", 50, 20)),
    ):
        harness_result = await run_output_harness(result, chunks, "query")

    assert harness_result.low_faithfulness is True
    assert len(harness_result.unsupported_claims) == 1


async def test_run_output_harness_handles_empty_answer():
    result = make_synthesis_result(answer="")
    chunks = [make_chunk("a")]

    harness_result = await run_output_harness(result, chunks, "query")
    assert harness_result.answer == ""
    assert harness_result.low_faithfulness is True