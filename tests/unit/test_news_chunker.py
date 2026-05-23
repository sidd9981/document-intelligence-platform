"""
Unit tests for the news chunker.

No services needed. All tests run against in-memory text.
"""

import tiktoken
import pytest
from datetime import date

from finsight.services import llm
from finsight.ingestion.chunkers.news_chunker import (
    CHUNK_MIN_TOKENS,
    CHUNK_TARGET_TOKENS,
    NewsChunk,
    chunk_news_article,
    _article_header,
)


@pytest.fixture(autouse=True)
def init_tokenizer():
    llm._tokenizer = tiktoken.get_encoding("cl100k_base")
    yield
    llm._tokenizer = None


SAMPLE_ARTICLE = """Apple Inc. reported quarterly earnings that exceeded analyst expectations,
driven by strong performance in its services division and steady iPhone demand
in key markets including the United States and Europe.

Chief Executive Tim Cook said the company was pleased with the results and
remained optimistic about the holiday quarter. Services revenue reached a new
record, growing 16 percent year over year to $21.2 billion.

Analysts had expected earnings per share of $1.39 on revenue of $89.3 billion.
Apple reported earnings per share of $1.46 on revenue of $89.5 billion, beating
estimates on both measures.

The company said it would continue to invest in artificial intelligence and
machine learning capabilities across its product lineup. Cook highlighted
improvements to Siri and new features in the latest iPhone models as evidence
of the company's commitment to the technology.
"""

PUB_DATE = date(2023, 10, 26)


def test_chunk_news_article_returns_chunks():
    chunks = chunk_news_article(SAMPLE_ARTICLE, "doc-001", "Reuters", "Apple Beats Estimates", PUB_DATE)
    assert len(chunks) >= 1


def test_all_chunks_have_required_fields():
    chunks = chunk_news_article(SAMPLE_ARTICLE, "doc-002", "Reuters", "Apple Beats Estimates", PUB_DATE)
    for chunk in chunks:
        assert chunk.chunk_id
        assert chunk.content
        assert chunk.token_count > 0
        assert chunk.source
        assert chunk.headline


def test_header_appears_in_every_chunk():
    chunks = chunk_news_article(SAMPLE_ARTICLE, "doc-003", "Reuters", "Apple Beats Estimates", PUB_DATE)
    for chunk in chunks:
        assert "Reuters" in chunk.content
        assert "2023-10-26" in chunk.content
        assert "Apple Beats Estimates" in chunk.content


def test_chunk_ids_are_deterministic():
    chunks_a = chunk_news_article(SAMPLE_ARTICLE, "doc-004", "Reuters", "Apple Beats Estimates", PUB_DATE)
    chunks_b = chunk_news_article(SAMPLE_ARTICLE, "doc-004", "Reuters", "Apple Beats Estimates", PUB_DATE)
    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]


def test_different_doc_ids_produce_different_chunk_ids():
    chunks_a = chunk_news_article(SAMPLE_ARTICLE, "doc-a", "Reuters", "Apple Beats Estimates", PUB_DATE)
    chunks_b = chunk_news_article(SAMPLE_ARTICLE, "doc-b", "Reuters", "Apple Beats Estimates", PUB_DATE)
    ids_a = {c.chunk_id for c in chunks_a}
    ids_b = {c.chunk_id for c in chunks_b}
    assert ids_a.isdisjoint(ids_b)


def test_short_article_produces_single_chunk():
    short = "Apple reported strong earnings this quarter, beating analyst expectations on revenue and EPS."
    chunks = chunk_news_article(short, "doc-005", "Reuters", "Apple Earnings", PUB_DATE)
    assert len(chunks) == 1


def test_article_header_format():
    header = _article_header("Reuters", "Apple Beats Estimates", PUB_DATE)
    assert header == "[Source: Reuters, Date: 2023-10-26, Headline: Apple Beats Estimates]"


def test_source_and_headline_stored_on_chunk():
    chunks = chunk_news_article(SAMPLE_ARTICLE, "doc-006", "Financial Times", "Apple Q4 Results", PUB_DATE)
    for chunk in chunks:
        assert chunk.source == "Financial Times"
        assert chunk.headline == "Apple Q4 Results"


def test_empty_paragraphs_are_ignored():
    text_with_gaps = "First paragraph with enough content to pass the minimum token threshold.\n\n\n\n\nSecond paragraph with enough content to also pass the minimum token threshold here."
    chunks = chunk_news_article(text_with_gaps, "doc-007", "Reuters", "Test", PUB_DATE)
    assert len(chunks) >= 1
    for chunk in chunks:
        assert chunk.token_count > 0