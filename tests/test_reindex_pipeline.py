"""
Unit tests for the blue/green re-indexing pipeline.

No live Qdrant needed. Mocks the client and injected functions.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finsight.mlops.reindex_pipeline import (
    MIN_EVAL_RECALL,
    ReindexStatus,
    run_reindex,
    swap_alias,
    create_collection,
    delete_collection,
)


def make_client():
    client = AsyncMock()
    client.create_collection = AsyncMock()
    client.upsert = AsyncMock()
    client.update_collection_aliases = AsyncMock()
    client.delete_collection = AsyncMock()
    return client


async def fake_embed(text: str) -> list[float]:
    return [0.1] * 768


async def fake_get_documents():
    return [
        {"id": "doc1", "text": "Apple risk factors", "metadata": {"ticker": "AAPL"}},
        {"id": "doc2", "text": "TSMC supply chain", "metadata": {"ticker": "TSM"}},
    ]


@pytest.mark.asyncio
async def test_reindex_swaps_alias_when_eval_passes():
    client = make_client()

    async def good_eval(collection):
        return 0.80

    result = await run_reindex(
        client=client,
        old_collection="filings_dense_v1",
        new_collection="filings_dense_v2",
        embedding_dim=768,
        embed_fn=fake_embed,
        get_documents_fn=fake_get_documents,
        eval_fn=good_eval,
    )

    assert result.status == ReindexStatus.SWAPPED
    assert result.eval_recall == 0.80
    assert result.docs_reindexed == 2
    client.update_collection_aliases.assert_called_once()


@pytest.mark.asyncio
async def test_reindex_aborts_when_eval_fails():
    client = make_client()

    async def bad_eval(collection):
        return 0.50

    result = await run_reindex(
        client=client,
        old_collection="filings_dense_v1",
        new_collection="filings_dense_v2",
        embedding_dim=768,
        embed_fn=fake_embed,
        get_documents_fn=fake_get_documents,
        eval_fn=bad_eval,
    )

    assert result.status == ReindexStatus.FAILED
    assert result.eval_recall == 0.50
    client.update_collection_aliases.assert_not_called()
    client.delete_collection.assert_called_once_with("filings_dense_v2")


@pytest.mark.asyncio
async def test_reindex_deletes_new_collection_on_eval_failure():
    client = make_client()

    async def bad_eval(collection):
        return 0.60

    result = await run_reindex(
        client=client,
        old_collection="filings_dense_v1",
        new_collection="filings_dense_v2",
        embedding_dim=768,
        embed_fn=fake_embed,
        get_documents_fn=fake_get_documents,
        eval_fn=bad_eval,
    )

    client.delete_collection.assert_called_once_with("filings_dense_v2")


@pytest.mark.asyncio
async def test_reindex_returns_failed_on_exception():
    client = make_client()
    client.create_collection.side_effect = RuntimeError("qdrant down")

    result = await run_reindex(
        client=client,
        old_collection="filings_dense_v1",
        new_collection="filings_dense_v2",
        embedding_dim=768,
        embed_fn=fake_embed,
        get_documents_fn=fake_get_documents,
        eval_fn=AsyncMock(return_value=0.80),
    )

    assert result.status == ReindexStatus.FAILED
    assert "qdrant down" in result.message


@pytest.mark.asyncio
async def test_reindex_indexes_all_documents():
    client = make_client()

    async def good_eval(collection):
        return 0.75

    result = await run_reindex(
        client=client,
        old_collection="filings_dense_v1",
        new_collection="filings_dense_v2",
        embedding_dim=768,
        embed_fn=fake_embed,
        get_documents_fn=fake_get_documents,
        eval_fn=good_eval,
    )

    assert result.docs_reindexed == 2
    client.upsert.assert_called_once()


@pytest.mark.asyncio
async def test_swap_alias_calls_update():
    client = make_client()
    await swap_alias(client, "filings_current", "filings_dense_v2")
    client.update_collection_aliases.assert_called_once()


@pytest.mark.asyncio
async def test_delete_collection_calls_client():
    client = make_client()
    await delete_collection(client, "filings_dense_v2")
    client.delete_collection.assert_called_once_with("filings_dense_v2")