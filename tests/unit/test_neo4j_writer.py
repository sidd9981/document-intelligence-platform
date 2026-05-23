"""
Unit tests for the Neo4j writer.

No running Neo4j needed. We mock the driver and session to verify
Cypher queries are called with the right arguments.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finsight.ingestion.neo4j_writer import (
    Neo4jWriter,
    _merge_company_node,
    _merge_filing_node,
    _merge_mentioned_in,
)
from finsight.ingestion.stream_backend import STREAM_ENTITIES


def make_entity_message(cik: str = "0000320193", doc_id: str = "doc-001") -> dict:
    return {
        "id": "msg-001",
        "data": {
            "doc_id": doc_id,
            "ticker": "AAPL",
            "filing_type": "10-K",
            "filing_date": "2023-10-27",
            "scopes": ["public", "analysis"],
            "entity": {
                "raw_text": "AAPL",
                "entity_type": "company",
                "canonical_cik": cik,
                "canonical_name": "Apple Inc.",
                "confidence": 1.0,
            },
        },
    }


@pytest.fixture
def mock_backend() -> AsyncMock:
    backend = AsyncMock()
    backend.consume = AsyncMock(return_value=[])
    backend.ack = AsyncMock()
    return backend


@pytest.fixture
def mock_session() -> AsyncMock:
    session = AsyncMock()
    session.execute_write = AsyncMock()
    session.__aenter__ = AsyncMock(return_value=session)
    session.__aexit__ = AsyncMock(return_value=False)
    return session


@pytest.fixture
def mock_driver(mock_session: AsyncMock) -> MagicMock:
    driver = MagicMock()
    driver.session = MagicMock(return_value=mock_session)
    return driver


@pytest.fixture
def writer(mock_backend: AsyncMock, mock_driver: MagicMock) -> Neo4jWriter:
    return Neo4jWriter(
        backend=mock_backend,
        driver=mock_driver,
        consumer_name="writer-test",
    )


async def test_process_message_calls_execute_write_three_times(
    writer: Neo4jWriter,
    mock_session: AsyncMock,
) -> None:
    message = make_entity_message()
    await writer._process_message(message)
    assert mock_session.execute_write.call_count == 3


async def test_process_message_skips_entity_without_cik(
    writer: Neo4jWriter,
    mock_session: AsyncMock,
) -> None:
    message = make_entity_message()
    message["data"]["entity"]["canonical_cik"] = None
    await writer._process_message(message)
    mock_session.execute_write.assert_not_called()


async def test_process_message_passes_correct_cik(
    writer: Neo4jWriter,
    mock_session: AsyncMock,
) -> None:
    message = make_entity_message(cik="0000320193")
    await writer._process_message(message)

    first_call = mock_session.execute_write.call_args_list[0]
    assert first_call[1]["cik"] == "0000320193"


async def test_process_message_passes_correct_doc_id(
    writer: Neo4jWriter,
    mock_session: AsyncMock,
) -> None:
    message = make_entity_message(doc_id="abc-123")
    await writer._process_message(message)

    second_call = mock_session.execute_write.call_args_list[1]
    assert second_call[1]["doc_id"] == "abc-123"


async def test_process_message_passes_scopes(
    writer: Neo4jWriter,
    mock_session: AsyncMock,
) -> None:
    message = make_entity_message()
    await writer._process_message(message)

    first_call = mock_session.execute_write.call_args_list[0]
    assert "public" in first_call[1]["scopes"]


async def test_merge_company_node_runs_merge_cypher() -> None:
    tx = AsyncMock()
    await _merge_company_node(tx, "0000320193", "Apple Inc.", "AAPL", ["public"])
    tx.run.assert_called_once()
    cypher = tx.run.call_args[0][0]
    assert "MERGE" in cypher
    assert "Company" in cypher
    assert "cik" in cypher


async def test_merge_filing_node_runs_merge_cypher() -> None:
    tx = AsyncMock()
    await _merge_filing_node(tx, "doc-001", "AAPL", "10-K", "2023-10-27", ["public"])
    tx.run.assert_called_once()
    cypher = tx.run.call_args[0][0]
    assert "MERGE" in cypher
    assert "Filing" in cypher


async def test_merge_mentioned_in_runs_merge_cypher() -> None:
    tx = AsyncMock()
    await _merge_mentioned_in(tx, "0000320193", "doc-001")
    tx.run.assert_called_once()
    cypher = tx.run.call_args[0][0]
    assert "MERGE" in cypher
    assert "MENTIONED_IN" in cypher