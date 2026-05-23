"""
Chunker for SEC EDGAR filings (10-K, 10-Q, 8-K, DEF 14A).

SEC filings have known section structure. We preserve that structure
in chunk metadata so downstream queries can filter by section without
reading chunk content. A chunk tagged section="Item 1A" is a risk
factor. That label matters for retrieval quality.

The generic Phase 1 splitter is replaced by this for all SEC filing
types from Phase 2 onwards.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass, field

from finsight.services.llm import count_tokens

CHUNK_TARGET_TOKENS = 400
CHUNK_MIN_TOKENS = 50

# Matches SEC section headers like:
#   "Item 1A. Risk Factors"
#   "ITEM 1A - RISK FACTORS"
#   "Item 7:"
# Permissive on punctuation and case because filings are inconsistent.
SECTION_HEADER_RE = re.compile(
    r"^\s*item\s+\d+[a-z]?\s*[.\-:)]\s*.+",
    re.IGNORECASE,
)

# Matches inline XBRL tags that appear in modern SEC filings.
# These carry no semantic content and confuse the tokenizer.
XBRL_TAG_RE = re.compile(r"<ix:[^>]+>|</ix:[^>]+>", re.IGNORECASE)


@dataclass
class SecChunk:
    """A single chunk produced by the SEC chunker.

    section is the most important field here. It tells downstream
    components where in the filing this text came from without them
    having to parse the content themselves.
    """

    chunk_id: str
    content: str
    token_count: int
    chunk_index: int
    section: str
    is_table: bool = False


def chunk_sec_filing(text: str, doc_id: str) -> list[SecChunk]:
    """Split a SEC filing into structured chunks with section labels.

    Args:
        text: Raw filing text, either plain text or HTML.
        doc_id: Used to generate stable chunk IDs. Same doc always
                produces the same chunk IDs so ingestion is idempotent.

    Returns:
        List of SecChunk objects ordered by appearance in the document.
    """
    clean = _strip_html(text)
    sections = _split_into_sections(clean)

    chunks: list[SecChunk] = []
    chunk_index = 0

    for section_name, section_text in sections:
        tables, text_without_tables = _extract_tables(section_text)

        for table_text in tables:
            token_count = count_tokens(table_text)
            if token_count < CHUNK_MIN_TOKENS:
                continue

            chunks.append(SecChunk(
                chunk_id=_chunk_id(doc_id, chunk_index),
                content=table_text,
                token_count=token_count,
                chunk_index=chunk_index,
                section=section_name,
                is_table=True,
            ))
            chunk_index += 1

        text_chunks = _split_text_section(text_without_tables, doc_id, section_name, chunk_index)
        chunks.extend(text_chunks)
        chunk_index += len(text_chunks)

    return chunks


def _strip_html(text: str) -> str:
    """Remove HTML tags and decode common entities.

    Strips XBRL inline tags first because they sometimes wrap content
    that would otherwise look like a regular HTML tag, which confuses
    a naive tag stripper.
    """
    text = XBRL_TAG_RE.sub("", text)
    text = re.sub(r"<[^>]+>", " ", text)
    text = text.replace("&nbsp;", " ")
    text = text.replace("&amp;", "&")
    text = text.replace("&lt;", "<")
    text = text.replace("&gt;", ">")
    text = text.replace("&quot;", '"')
    text = re.sub(r"\s{3,}", "\n\n", text)
    return text.strip()


def _split_into_sections(text: str) -> list[tuple[str, str]]:
    """Split filing text into (section_name, section_text) pairs.

    Uses the Item header regex to detect section boundaries. Text
    before the first Item header goes into a "Preamble" section.
    If no headers are detected the whole document is one section
    called "Full Document" — this handles 8-Ks which are often
    short and unstructured.
    """
    lines = text.splitlines()
    sections: list[tuple[str, str]] = []
    current_name = "Preamble"
    current_lines: list[str] = []
    found_item_header = False

    for line in lines:
        if SECTION_HEADER_RE.match(line):
            found_item_header = True
            if current_lines:
                content = "\n".join(current_lines).strip()
                if content:
                    sections.append((current_name, content))
            current_name = line.strip()
            current_lines = []
        else:
            current_lines.append(line)

    if current_lines:
        content = "\n".join(current_lines).strip()
        if content:
            sections.append((current_name, content))

    if not found_item_header:
        return [("Full Document", text)]

    return sections if sections else [("Full Document", text)]


def _extract_tables(text: str) -> tuple[list[str], str]:
    """Pull tables out of section text and return them separately.

    Tables are never split. Each table becomes exactly one chunk
    regardless of size. The remaining text has the table placeholders
    removed so the text splitter doesn't see gaps.

    This is a best-effort heuristic. Real SEC tables are HTML and
    would be caught by the HTML stripper. Plain-text tables (rows of
    numbers separated by whitespace) are detected by the pipe character
    pattern common in converted filings.
    """
    table_pattern = re.compile(
        r"(\|.+\|(?:\n\|.+\|)+)",
        re.MULTILINE,
    )

    tables = []
    text_without_tables = text

    for match in table_pattern.finditer(text):
        tables.append(match.group(0).strip())

    text_without_tables = table_pattern.sub("", text_without_tables).strip()
    return tables, text_without_tables


def _split_text_section(
    text: str,
    doc_id: str,
    section_name: str,
    start_index: int,
) -> list[SecChunk]:
    """Split a section's text into chunks at paragraph boundaries.

    Targets CHUNK_TARGET_TOKENS per chunk. Never splits mid-paragraph.
    Skips paragraphs below CHUNK_MIN_TOKENS so we don't produce
    single-sentence chunks from whitespace artifacts in the filing.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[SecChunk] = []
    current_paragraphs: list[str] = []
    current_tokens = 0
    chunk_index = start_index

    for paragraph in paragraphs:
        paragraph_tokens = count_tokens(paragraph)

        if paragraph_tokens < CHUNK_MIN_TOKENS:
            continue

        if current_tokens + paragraph_tokens > CHUNK_TARGET_TOKENS and current_paragraphs:
            content = "\n\n".join(current_paragraphs)
            chunks.append(SecChunk(
                chunk_id=_chunk_id(doc_id, chunk_index),
                content=content,
                token_count=current_tokens,
                chunk_index=chunk_index,
                section=section_name,
            ))
            chunk_index += 1
            current_paragraphs = current_paragraphs[-1:]
            current_tokens = count_tokens(current_paragraphs[0]) if current_paragraphs else 0

        current_paragraphs.append(paragraph)
        current_tokens += paragraph_tokens

    if current_paragraphs:
        content = "\n\n".join(current_paragraphs)
        token_count = count_tokens(content)
        if token_count >= CHUNK_MIN_TOKENS:
            chunks.append(SecChunk(
                chunk_id=_chunk_id(doc_id, chunk_index),
                content=content,
                token_count=token_count,
                chunk_index=chunk_index,
                section=section_name,
            ))

    return chunks


def _chunk_id(doc_id: str, chunk_index: int) -> str:
    raw = f"{doc_id}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]