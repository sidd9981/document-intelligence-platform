"""
Unit tests for the SEC chunker.

No services needed. All tests run against in-memory text.
"""

import tiktoken
import pytest

from finsight.services import llm
from finsight.ingestion.chunkers.sec_chunker import (
    CHUNK_MIN_TOKENS,
    CHUNK_TARGET_TOKENS,
    SecChunk,
    chunk_sec_filing,
    _strip_html,
    _split_into_sections,
    _extract_tables,
)


@pytest.fixture(autouse=True)
def init_tokenizer():
    llm._tokenizer = tiktoken.get_encoding("cl100k_base")
    yield
    llm._tokenizer = None


SAMPLE_FILING = """
Item 1. Business

Apple Inc. designs, manufactures, and markets smartphones, personal computers,
tablets, wearables, and accessories worldwide. The Company also sells a variety
of related services.

Item 1A. Risk Factors

The Company depends on the performance of distributors, operators, and
resellers of the Company's products. The Company also relies on sole-source
suppliers for certain components, including Taiwan Semiconductor Manufacturing
Company Limited, which manufactures substantially all of the Company's custom
silicon chips.

Item 7. Management's Discussion and Analysis

Net sales increased 8% or $32.1 billion during 2023 compared to 2022. The
increase was driven by higher net sales of Services, partially offset by
lower net sales of Mac and iPad.
"""


def test_chunk_sec_filing_returns_chunks():
    chunks = chunk_sec_filing(SAMPLE_FILING, doc_id="test-001")
    assert len(chunks) >= 1


def test_all_chunks_have_required_fields():
    chunks = chunk_sec_filing(SAMPLE_FILING, doc_id="test-002")
    for chunk in chunks:
        assert chunk.chunk_id
        assert chunk.content
        assert chunk.token_count > 0
        assert chunk.section


def test_section_labels_are_detected():
    chunks = chunk_sec_filing(SAMPLE_FILING, doc_id="test-003")
    sections = {c.section for c in chunks}
    assert any("1A" in s for s in sections), f"expected Item 1A in sections, got {sections}"
    assert any("1." in s or "1 " in s or s.startswith("Item 1") for s in sections), f"expected Item 1 in sections, got {sections}"


def test_chunk_ids_are_deterministic():
    chunks_a = chunk_sec_filing(SAMPLE_FILING, doc_id="test-004")
    chunks_b = chunk_sec_filing(SAMPLE_FILING, doc_id="test-004")
    assert [c.chunk_id for c in chunks_a] == [c.chunk_id for c in chunks_b]


def test_different_doc_ids_produce_different_chunk_ids():
    chunks_a = chunk_sec_filing(SAMPLE_FILING, doc_id="doc-a")
    chunks_b = chunk_sec_filing(SAMPLE_FILING, doc_id="doc-b")
    ids_a = {c.chunk_id for c in chunks_a}
    ids_b = {c.chunk_id for c in chunks_b}
    assert ids_a.isdisjoint(ids_b)


def test_no_chunk_below_min_token_count():
    chunks = chunk_sec_filing(SAMPLE_FILING, doc_id="test-005")
    for chunk in chunks:
        assert chunk.token_count >= CHUNK_MIN_TOKENS, (
            f"chunk in section '{chunk.section}' has {chunk.token_count} tokens, below minimum"
        )


def test_strip_html_removes_tags():
    html = "<p>Apple Inc. <b>risk factors</b> apply here.</p>"
    result = _strip_html(html)
    assert "<" not in result
    assert "Apple Inc." in result
    assert "risk factors" in result


def test_strip_html_removes_xbrl_tags():
    html = '<ix:nonfraction>42.5</ix:nonfraction> billion in revenue'
    result = _strip_html(html)
    assert "ix:" not in result
    assert "42.5" in result


def test_strip_html_decodes_entities():
    html = "AT&amp;T earned &lt;10% growth"
    result = _strip_html(html)
    assert "AT&T" in result
    assert "<10%" in result


def test_split_into_sections_detects_item_headers():
    sections = _split_into_sections(SAMPLE_FILING)
    names = [s[0] for s in sections]
    assert any("1A" in n for n in names)


def test_split_into_sections_unstructured_text():
    text = "This is a short filing with no item headers at all."
    sections = _split_into_sections(text)
    assert len(sections) == 1
    assert sections[0][0] == "Full Document"


def test_extract_tables_separates_table_from_text():
    text = "Some narrative text here.\n\n| Revenue | 42 |\n| Costs | 10 |\n\nMore narrative."
    tables, remaining = _extract_tables(text)
    assert len(tables) == 1
    assert "Revenue" in tables[0]
    assert "Revenue" not in remaining


def test_table_chunks_are_flagged():
    big_table = "\n".join(
        [f"| Company {i} | Revenue {i}B | Growth {i}% | Risk Level {i} |" for i in range(1, 15)]
    )
    filing_with_table = SAMPLE_FILING + f"\n\nItem 8. Financial Statements\n\n{big_table}"
    chunks = chunk_sec_filing(filing_with_table, doc_id="test-006")
    table_chunks = [c for c in chunks if c.is_table]
    assert len(table_chunks) >= 1