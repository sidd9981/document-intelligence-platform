"""
Unit tests for the embedding worker.

No running services needed. We mock the stream backend and the embed
function to verify routing and publishing logic without network calls.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
import tiktoken
from finsight.services import llm

@pytest.fixture(autouse=True)
def init_tokenizer():
    llm._tokenizer = tiktoken.get_encoding("cl100k_base")
    yield
    llm._tokenizer = None

from finsight.ingestion.embedder import (
    EmbeddingWorker,
    _chunk_document,
    _doc_id_from_message,
)
from finsight.ingestion.stream_backend import STREAM_EMBEDDED


@pytest.fixture
def mock_backend() -> AsyncMock:
    backend = AsyncMock()
    backend.consume = AsyncMock(return_value=[])
    backend.publish = AsyncMock(return_value="111-0")
    backend.ack = AsyncMock()
    return backend


@pytest.fixture
def worker(mock_backend: AsyncMock) -> EmbeddingWorker:
    return EmbeddingWorker(backend=mock_backend, consumer_name="embedder-test")


def make_filing_message(filing_type: str = "10-K", text: str = "") -> dict:
    return {
        "id": "msg-001",
        "data": {
            "ticker": "AAPL",
            "cik": "0000320193",
            "company_name": "Apple Inc.",
            "filing_type": filing_type,
            "filing_date": "2023-10-27",
            "text": text or (
                "Item 1A. Risk Factors\n\n"
                "Apple depends on sole-source suppliers for certain components including "
                "Taiwan Semiconductor Manufacturing Company Limited which manufactures "
                "substantially all of the Company's custom silicon chips. This concentration "
                "creates significant supply chain risk that could adversely affect the "
                "Company's ability to deliver products on schedule and at acceptable cost.\n\n"
                "The Company also faces intense competition across all its markets. Competitors "
                "include Samsung Electronics, Alphabet, and Microsoft, each with significant "
                "resources and global distribution. Failure to compete effectively on product "
                "quality, pricing, or innovation could result in loss of market share and "
                "materially harm the Company's revenue and financial condition.\n\n"
                "Item 7. Management Discussion and Analysis\n\n"
                "Net sales increased eight percent or thirty two billion dollars during fiscal "
                "2023 compared to fiscal 2022. The increase was driven by higher net sales of "
                "Services which reached a record high, partially offset by lower net sales of "
                "Mac and iPad due to macroeconomic headwinds affecting consumer spending."
            ),
            "scopes": ["public", "analysis"],
        },
    }


def test_chunk_document_routes_sec_filing_types():
    for filing_type in ("10-K", "10-Q", "8-K", "DEF 14A"):
        data = make_filing_message(filing_type)["data"]
        chunks = _chunk_document(data, "doc-001")
        assert len(chunks) >= 1, f"expected chunks for {filing_type}"


def test_chunk_document_routes_transcript():
    data = {
        "filing_type": "transcript",
        "filing_date": "2023-10-26",
        "company_name": "Apple Inc.",
        "text": (
            "Tim Cook:\n"
            "We had a great quarter with revenue of 89.5 billion dollars. "
            "Services reached an all time record and we are very pleased with "
            "the performance across all our major product categories this period.\n\n"
            "Luca Maestri:\n"
            "Gross margin came in at 45.2 percent for the quarter which was above "
            "our guidance range. We generated strong operating cash flow and returned "
            "capital to shareholders through dividends and buybacks as planned."
        ),
    }
    chunks = _chunk_document(data, "doc-002")
    assert len(chunks) >= 1


def test_chunk_document_routes_news():
    data = {
        "filing_type": "news",
        "filing_date": "2023-10-26",
        "source": "Reuters",
        "headline": "Apple Beats Estimates",
        "text": "Apple reported quarterly earnings that exceeded analyst expectations driven by strong services growth.",
    }
    chunks = _chunk_document(data, "doc-003")
    assert len(chunks) >= 1


def test_chunk_document_returns_empty_for_unknown_type():
    data = {"filing_type": "unknown", "text": "some text here"}
    chunks = _chunk_document(data, "doc-004")
    assert chunks == []


def test_chunk_document_returns_empty_for_missing_text():
    data = {"filing_type": "10-K", "text": ""}
    chunks = _chunk_document(data, "doc-005")
    assert chunks == []


def test_doc_id_is_deterministic():
    data = {"ticker": "AAPL", "filing_type": "10-K", "filing_date": "2023-10-27"}
    assert _doc_id_from_message(data) == _doc_id_from_message(data)


def test_doc_id_differs_for_different_filings():
    data_a = {"ticker": "AAPL", "filing_type": "10-K", "filing_date": "2023-10-27"}
    data_b = {"ticker": "AAPL", "filing_type": "10-K", "filing_date": "2022-10-28"}
    assert _doc_id_from_message(data_a) != _doc_id_from_message(data_b)


async def test_process_message_publishes_to_embedded_stream(
    worker: EmbeddingWorker,
    mock_backend: AsyncMock,
) -> None:
    message = make_filing_message()

    with patch("finsight.ingestion.embedder.embed", new=AsyncMock(return_value=[0.1] * 768)):
        await worker._process_message(message)

    assert mock_backend.publish.called
    for call in mock_backend.publish.call_args_list:
        assert call[0][0] == STREAM_EMBEDDED


async def test_process_message_published_event_has_required_fields(
    worker: EmbeddingWorker,
    mock_backend: AsyncMock,
) -> None:
    message = make_filing_message()

    with patch("finsight.ingestion.embedder.embed", new=AsyncMock(return_value=[0.1] * 768)):
        await worker._process_message(message)

    assert mock_backend.publish.called
    published = mock_backend.publish.call_args_list[0][0][1]
    for field in ("chunk_id", "doc_id", "embedding", "content", "token_count", "metadata"):
        assert field in published, f"missing field: {field}"


async def test_process_message_skips_unknown_filing_type(
    worker: EmbeddingWorker,
    mock_backend: AsyncMock,
) -> None:
    message = make_filing_message(filing_type="unknown")

    with patch("finsight.ingestion.embedder.embed", new=AsyncMock(return_value=[0.1] * 768)):
        await worker._process_message(message)

    mock_backend.publish.assert_not_called()


async def test_embed_failure_does_not_stop_other_chunks(
    worker: EmbeddingWorker,
    mock_backend: AsyncMock,
) -> None:
    message = make_filing_message()
    call_count = 0

    async def flaky_embed(text: str) -> list[float]:
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            raise RuntimeError("embed timeout")
        return [0.1] * 768

    with patch("finsight.ingestion.embedder.embed", new=flaky_embed):
        await worker._process_message(message)

    assert mock_backend.publish.call_count >= 1