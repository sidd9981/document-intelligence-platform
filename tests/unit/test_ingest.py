"""
Unit tests for the ingestion module.

Tests chunking logic without requiring any running services.
The embed and upsert functions are not tested here — those are
covered by integration tests that require live services.
"""

import tiktoken
import pytest

from finsight.services import llm
from finsight.ingestion.ingest import chunk_text, _stable_chunk_id


@pytest.fixture(autouse=True)
def init_tokenizer():
    """Initialize the tokenizer for tests that use count_tokens.

    The tokenizer does not require any network calls or running
    services. It is initialized here directly rather than calling
    init_client() which would also try to connect to Ollama.
    """
    llm._tokenizer = tiktoken.get_encoding("cl100k_base")
    yield
    llm._tokenizer = None


def test_chunk_text_returns_at_least_one_chunk():
    """Any non-empty text must produce at least one chunk."""
    text = "This is a simple document with one paragraph."
    chunks = chunk_text(text, doc_id="test-doc-001")
    assert len(chunks) >= 1


def test_chunk_text_all_chunks_have_required_fields():
    """Every chunk must have chunk_id, content, token_count, chunk_index."""
    text = "First paragraph.\n\nSecond paragraph.\n\nThird paragraph."
    chunks = chunk_text(text, doc_id="test-doc-002")

    for chunk in chunks:
        assert "chunk_id" in chunk
        assert "content" in chunk
        assert "token_count" in chunk
        assert "chunk_index" in chunk
        assert len(chunk["content"]) > 0
        assert chunk["token_count"] > 0


def test_chunk_text_ids_are_deterministic():
    """Same input must always produce the same chunk IDs."""
    text = "Paragraph one.\n\nParagraph two.\n\nParagraph three."
    chunks_first = chunk_text(text, doc_id="test-doc-003")
    chunks_second = chunk_text(text, doc_id="test-doc-003")

    ids_first = [c["chunk_id"] for c in chunks_first]
    ids_second = [c["chunk_id"] for c in chunks_second]
    assert ids_first == ids_second


def test_chunk_text_different_docs_have_different_ids():
    """Same text with different doc_ids must produce different chunk IDs."""
    text = "Same content in both documents."
    chunks_a = chunk_text(text, doc_id="doc-a")
    chunks_b = chunk_text(text, doc_id="doc-b")

    ids_a = {c["chunk_id"] for c in chunks_a}
    ids_b = {c["chunk_id"] for c in chunks_b}
    assert ids_a.isdisjoint(ids_b)


def test_stable_chunk_id_is_deterministic():
    """Same doc_id and chunk_index must always produce the same ID."""
    id1 = _stable_chunk_id("doc-001", 0)
    id2 = _stable_chunk_id("doc-001", 0)
    assert id1 == id2


def test_stable_chunk_id_length():
    """Chunk IDs must be exactly 32 characters."""
    chunk_id = _stable_chunk_id("doc-001", 0)
    assert len(chunk_id) == 32