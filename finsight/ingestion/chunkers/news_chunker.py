"""
Chunker for financial news articles.

Paragraph-level splitting with article metadata prepended to each chunk.
News articles are typically short enough that most produce one or two
chunks. The metadata header is what lets the synthesis agent attribute
claims to a specific source and date.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import date

from finsight.services.llm import count_tokens

CHUNK_TARGET_TOKENS = 300
CHUNK_MIN_TOKENS = 15


@dataclass
class NewsChunk:
    chunk_id: str
    content: str
    token_count: int
    chunk_index: int
    source: str
    headline: str


def chunk_news_article(
    text: str,
    doc_id: str,
    source: str,
    headline: str,
    published_date: date,
) -> list[NewsChunk]:
    """Split a news article into paragraph-level chunks.

    Each chunk gets a metadata header with source, headline, and date.
    Paragraphs below CHUNK_MIN_TOKENS are merged into the next one
    rather than becoming their own chunk.

    Args:
        text: Raw article text, plain text or lightly formatted.
        doc_id: Used to generate stable chunk IDs.
        source: Publication name, e.g. 'Reuters'.
        headline: Article headline, included in every chunk header so
                  the synthesis agent knows the article context even
                  when only reading a single paragraph.
        published_date: Publication date.

    Returns:
        List of NewsChunk objects in order of appearance.
    """
    header = _article_header(source, headline, published_date)
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks: list[NewsChunk] = []
    current_paragraphs: list[str] = []
    current_tokens = 0
    chunk_index = 0

    for paragraph in paragraphs:
        paragraph_tokens = count_tokens(paragraph)

        if paragraph_tokens < CHUNK_MIN_TOKENS:
            current_paragraphs.append(paragraph)
            current_tokens += paragraph_tokens
            continue

        if current_tokens + paragraph_tokens > CHUNK_TARGET_TOKENS and current_paragraphs:
            content = header + "\n\n" + "\n\n".join(current_paragraphs)
            chunks.append(NewsChunk(
                chunk_id=_chunk_id(doc_id, chunk_index),
                content=content,
                token_count=count_tokens(content),
                chunk_index=chunk_index,
                source=source,
                headline=headline,
            ))
            chunk_index += 1
            current_paragraphs = []
            current_tokens = 0

        current_paragraphs.append(paragraph)
        current_tokens += paragraph_tokens

    if current_paragraphs:
        content = header + "\n\n" + "\n\n".join(current_paragraphs)
        chunks.append(NewsChunk(
            chunk_id=_chunk_id(doc_id, chunk_index),
            content=content,
            token_count=count_tokens(content),
            chunk_index=chunk_index,
            source=source,
            headline=headline,
        ))

    return chunks


def _article_header(source: str, headline: str, published_date: date) -> str:
    return f"[Source: {source}, Date: {published_date.isoformat()}, Headline: {headline}]"


def _chunk_id(doc_id: str, chunk_index: int) -> str:
    raw = f"{doc_id}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]