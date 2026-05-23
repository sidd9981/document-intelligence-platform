"""
Phase 1 document ingestion.

Takes a raw text document, splits it into chunks, embeds each chunk,
writes vectors to Qdrant, and writes document metadata to Postgres.

This is the simplified Phase 1 ingestion path. It runs synchronously
in a single process with no event streaming. Phase 2 replaces this
with a Redis Streams pipeline where chunking, embedding, and entity
extraction run as separate consumer groups.

The interface is kept intentionally simple so Phase 2 can swap the
transport layer without changing the chunking or embedding logic.
"""

import hashlib
import uuid
from datetime import date

import asyncpg

from finsight.config.settings import settings
from finsight.gateway.db import get_pool
from finsight.services.llm import count_tokens, embed
from finsight.services.vector_store import upsert_chunks
from finsight.telemetry.tracing import get_tracer

tracer = get_tracer(__name__)

CHUNK_TARGET_TOKENS = 400
CHUNK_OVERLAP_TOKENS = 50


def chunk_text(text: str, doc_id: str) -> list[dict]:
    """Split text into overlapping chunks of approximately target size.

    Uses a simple paragraph-aware splitting strategy for Phase 1.
    Phase 2 replaces this with document-type-specific chunkers:
    SECChunker for filings, TranscriptChunker for earnings calls.

    Overlap between chunks preserves context at boundaries. A sentence
    that spans a chunk boundary appears in both chunks so neither chunk
    loses the full context of that sentence.

    Args:
        text: Raw document text.
        doc_id: Document identifier, used to generate stable chunk IDs.

    Returns:
        List of dicts, each with chunk_id, content, and token_count.
    """
    paragraphs = [p.strip() for p in text.split("\n\n") if p.strip()]

    chunks = []
    current_paragraphs: list[str] = []
    current_tokens = 0
    chunk_index = 0

    for paragraph in paragraphs:
        paragraph_tokens = count_tokens(paragraph)

        if current_tokens + paragraph_tokens > CHUNK_TARGET_TOKENS and current_paragraphs:
            content = "\n\n".join(current_paragraphs)
            chunk_id = _stable_chunk_id(doc_id, chunk_index)

            chunks.append({
                "chunk_id": chunk_id,
                "content": content,
                "token_count": current_tokens,
                "chunk_index": chunk_index,
            })

            chunk_index += 1
            current_paragraphs = current_paragraphs[-2:] if len(current_paragraphs) > 2 else current_paragraphs
            current_tokens = sum(count_tokens(p) for p in current_paragraphs)

        current_paragraphs.append(paragraph)
        current_tokens += paragraph_tokens

    if current_paragraphs:
        content = "\n\n".join(current_paragraphs)
        chunk_id = _stable_chunk_id(doc_id, chunk_index)
        chunks.append({
            "chunk_id": chunk_id,
            "content": content,
            "token_count": current_tokens,
            "chunk_index": chunk_index,
        })

    return chunks


def _stable_chunk_id(doc_id: str, chunk_index: int) -> str:
    """Generate a deterministic chunk ID from doc_id and chunk index.

    Deterministic IDs make ingestion idempotent — running the same
    document through ingestion twice produces the same chunk IDs and
    Qdrant upserts overwrite rather than duplicate.
    """
    raw = f"{doc_id}:{chunk_index}"
    return hashlib.sha256(raw.encode()).hexdigest()[:32]


async def ingest_document(
    text: str,
    ticker: str,
    company_name: str,
    filing_type: str,
    filing_date: date,
    section: str,
    scopes: list[str],
    source_url: str | None = None,
) -> str:
    """Ingest a single document into Qdrant and Postgres.

    Chunks the document, embeds each chunk, writes vectors to Qdrant,
    and writes document metadata to Postgres. Returns the doc_id.

    Args:
        text: Raw document text.
        ticker: Company ticker symbol.
        company_name: Canonical company name.
        filing_type: One of 10-K, 10-Q, 8-K, transcript, news.
        filing_date: Date of the filing or publication.
        section: Document section name, e.g. "Risk Factors".
        scopes: List of team IDs that can access this document,
                plus "public" if accessible to all teams.
        source_url: Original URL of the document, if available.

    Returns:
        The generated doc_id as a string.
    """
    doc_id = str(uuid.uuid4())

    with tracer.start_as_current_span("ingest.document") as span:
        span.set_attribute("ticker", ticker)
        span.set_attribute("filing_type", filing_type)
        span.set_attribute("section", section)

        chunks = chunk_text(text, doc_id)
        span.set_attribute("chunks.count", len(chunks))

        embedded_chunks = []
        for chunk in chunks:
            embedding = await embed(chunk["content"])

            embedded_chunks.append({
                "chunk_id": chunk["chunk_id"],
                "embedding": embedding,
                "content": chunk["content"],
                "metadata": {
                    "doc_id": doc_id,
                    "ticker": ticker,
                    "company_name": company_name,
                    "filing_type": filing_type,
                    "filing_date": str(filing_date),
                    "section": section,
                    "chunk_index": chunk["chunk_index"],
                    "token_count": chunk["token_count"],
                    "embedding_model": settings.ollama.embedding_model,
                    "scopes": scopes,
                },
            })

        await upsert_chunks(embedded_chunks)

        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO documents (
                    doc_id, ticker, company_name, filing_type,
                    filing_date, source_url, chunk_count,
                    embedding_model, scopes
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8, $9)
                ON CONFLICT (doc_id) DO UPDATE SET
                    chunk_count = EXCLUDED.chunk_count,
                    updated_at = NOW()
                """,
                uuid.UUID(doc_id),
                ticker,
                company_name,
                filing_type,
                filing_date,
                source_url,
                len(chunks),
                settings.ollama.embedding_model,
                scopes,
            )

        span.set_attribute("doc_id", doc_id)
        return doc_id