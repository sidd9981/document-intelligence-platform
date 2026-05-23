"""
Unit tests for the vector store module.

These tests do not require a running Qdrant instance. They verify
the module's interface and error handling in isolation.
"""

import pytest

from finsight.services import vector_store


@pytest.fixture(autouse=True)
def reset_client():
    """Reset client state between tests."""
    vector_store._client = None
    yield
    vector_store._client = None


def test_get_client_raises_before_init():
    """get_client() must raise RuntimeError if called before init_client()."""
    with pytest.raises(RuntimeError, match="not initialized"):
        vector_store.get_client()


async def test_search_sparse_raises_before_init():
    with pytest.raises(RuntimeError, match="not initialized"):
        await vector_store.search_sparse({100: 1.5, 200: 0.8}, "ops", k=10)


async def test_upsert_sparse_chunks_raises_before_init():
    with pytest.raises(RuntimeError, match="not initialized"):
        await vector_store.upsert_sparse_chunks([])