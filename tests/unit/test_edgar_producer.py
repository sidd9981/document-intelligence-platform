"""
Unit tests for the EDGAR producer.

No network calls, no Redis. We mock the HTTP client and stream backend
and verify the producer routes data correctly between them.
"""

from __future__ import annotations

import json
from unittest.mock import AsyncMock, MagicMock, patch

import httpx
import pytest

from finsight.ingestion.producers.edgar_producer import (
    SUPPORTED_FILING_TYPES,
    EdgarProducer,
)
from finsight.ingestion.stream_backend import STREAM_FILINGS


def make_tickers_response(ticker: str, cik: int, title: str) -> dict:
    return {"0": {"cik_str": cik, "ticker": ticker, "title": title}}


def make_submissions_response(filing_type: str, dates: list[str]) -> dict:
    n = len(dates)
    return {
        "filings": {
            "recent": {
                "form": [filing_type] * n,
                "filingDate": dates,
                "accessionNumber": [f"0000320193-23-{str(i).zfill(6)}" for i in range(n)],
                "primaryDocument": [f"filing-{i}.htm" for i in range(n)],
            }
        }
    }


@pytest.fixture
def mock_backend() -> AsyncMock:
    backend = AsyncMock()
    backend.publish = AsyncMock(return_value="111-0")
    return backend


@pytest.fixture
def mock_http() -> AsyncMock:
    client = AsyncMock(spec=httpx.AsyncClient)

    tickers_resp = MagicMock()
    tickers_resp.json.return_value = make_tickers_response("AAPL", 320193, "Apple Inc.")
    tickers_resp.raise_for_status = MagicMock()

    submissions_resp = MagicMock()
    submissions_resp.json.return_value = make_submissions_response("10-K", ["2023-10-27", "2022-10-28"])
    submissions_resp.raise_for_status = MagicMock()

    filing_resp = MagicMock()
    filing_resp.text = "Apple risk factors content here."
    filing_resp.raise_for_status = MagicMock()

    client.get = AsyncMock(side_effect=[
        tickers_resp,       # company_tickers.json
        tickers_resp,       # unused second call in _resolve_cik
        submissions_resp,   # submissions/CIK.json
        filing_resp,        # first filing text
        filing_resp,        # second filing text
    ])

    return client


@pytest.fixture
def producer(mock_backend: AsyncMock, mock_http: AsyncMock) -> EdgarProducer:
    return EdgarProducer(backend=mock_backend, http_client=mock_http)


async def test_produce_filings_publishes_to_correct_stream(
    producer: EdgarProducer,
    mock_backend: AsyncMock,
    mock_http: AsyncMock,
) -> None:
    await producer.produce_filings("AAPL", "10-K", start_year=2022, end_year=2023)

    for call in mock_backend.publish.call_args_list:
        assert call[0][0] == STREAM_FILINGS


async def test_produce_filings_message_contains_required_fields(
    producer: EdgarProducer,
    mock_backend: AsyncMock,
    mock_http: AsyncMock,
) -> None:
    await producer.produce_filings("AAPL", "10-K", start_year=2022, end_year=2023)

    assert mock_backend.publish.called
    message = mock_backend.publish.call_args_list[0][0][1]

    for field in ("ticker", "cik", "company_name", "filing_type", "filing_date", "text", "scopes", "source_url"):
        assert field in message, f"missing field: {field}"


async def test_produce_filings_returns_count_of_published(
    producer: EdgarProducer,
    mock_http: AsyncMock,
    mock_backend: AsyncMock,
) -> None:
    count = await producer.produce_filings("AAPL", "10-K", start_year=2022, end_year=2023)
    assert count == 2


async def test_produce_filings_rejects_unsupported_filing_type(
    producer: EdgarProducer,
) -> None:
    with pytest.raises(ValueError, match="not a supported filing type"):
        await producer.produce_filings("AAPL", "S-1", start_year=2023)


async def test_produce_filings_skips_empty_text(
    mock_backend: AsyncMock,
    mock_http: AsyncMock,
) -> None:
    empty_resp = MagicMock()
    empty_resp.text = "   "
    empty_resp.raise_for_status = MagicMock()

    tickers_resp = MagicMock()
    tickers_resp.json.return_value = make_tickers_response("AAPL", 320193, "Apple Inc.")
    tickers_resp.raise_for_status = MagicMock()

    submissions_resp = MagicMock()
    submissions_resp.json.return_value = make_submissions_response("10-K", ["2023-10-27"])
    submissions_resp.raise_for_status = MagicMock()

    mock_http.get = AsyncMock(side_effect=[tickers_resp, tickers_resp, submissions_resp, empty_resp])

    producer = EdgarProducer(backend=mock_backend, http_client=mock_http)
    count = await producer.produce_filings("AAPL", "10-K", start_year=2023)

    assert count == 0
    mock_backend.publish.assert_not_called()


async def test_produce_filings_uses_default_public_scope(
    producer: EdgarProducer,
    mock_backend: AsyncMock,
    mock_http: AsyncMock,
) -> None:
    await producer.produce_filings("AAPL", "10-K", start_year=2023, end_year=2023)

    message = mock_backend.publish.call_args_list[0][0][1]
    assert "public" in message["scopes"]


async def test_resolve_cik_raises_for_unknown_ticker(
    mock_backend: AsyncMock,
) -> None:
    unknown_resp = MagicMock()
    unknown_resp.json.return_value = make_tickers_response("AAPL", 320193, "Apple Inc.")
    unknown_resp.raise_for_status = MagicMock()

    http = AsyncMock(spec=httpx.AsyncClient)
    http.get = AsyncMock(side_effect=[unknown_resp, unknown_resp])

    producer = EdgarProducer(backend=mock_backend, http_client=http)

    with pytest.raises(ValueError, match="not found in EDGAR"):
        await producer._resolve_cik("ZZZZ")