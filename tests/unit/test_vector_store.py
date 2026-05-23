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