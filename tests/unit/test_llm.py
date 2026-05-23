"""
Unit tests for the LLM client module.
"""

import pytest

from finsight.services import llm


@pytest.fixture(autouse=True)
def reset_client():
    """Reset client state between tests."""
    llm._client = None
    llm._tokenizer = None
    yield
    llm._client = None
    llm._tokenizer = None


def test_get_client_raises_before_init():
    """get_client() must raise RuntimeError if called before init_client()."""
    with pytest.raises(RuntimeError, match="not initialized"):
        llm.get_client()


def test_get_tokenizer_raises_before_init():
    """get_tokenizer() must raise RuntimeError if called before init_client()."""
    with pytest.raises(RuntimeError, match="not initialized"):
        llm.get_tokenizer()


def test_count_tokens_raises_before_init():
    """count_tokens() must raise RuntimeError if tokenizer not initialized."""
    with pytest.raises(RuntimeError, match="not initialized"):
        llm.count_tokens("hello world")