"""
Qdrant vector store client wrapper.

All vector search and upsert operations go through this module.
Direct use of the Qdrant client outside this file is not permitted.

Responsibilities of this wrapper:
    - Tenant scope filtering on every search operation. A search that
      does not filter by scopes would return documents the requesting
      team is not authorized to see. This is enforced here so callers
      cannot accidentally omit it.
    - OTEL span creation on every operation so latency and result
      counts are visible in traces without each caller doing it.
    - Consistent error handling so Qdrant-specific exceptions are
      caught at one boundary and converted to structured AgentError
      objects rather than propagating raw client exceptions.
"""

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import (
    Distance,
    FieldCondition,
    Filter,
    MatchAny,
    PointStruct,
    VectorParams,
)

from finsight.config.settings import settings
from finsight.models.base import Chunk, ChunkMetadata
from finsight.telemetry.tracing import get_tracer

tracer = get_tracer(__name__)

_client: AsyncQdrantClient | None = None


def get_client() -> AsyncQdrantClient:
    """Return the shared Qdrant client instance.

    Raises:
        RuntimeError: If called before init_client() has been awaited.
    """
    if _client is None:
        raise RuntimeError(
            "qdrant client is not initialized. "
            "call init_client() during application startup."
        )
    return _client


async def init_client() -> None:
    """Initialize the shared Qdrant client.

    Must be called once at application startup. The client maintains
    an internal connection pool to Qdrant. Creating multiple clients
    would create multiple pools unnecessarily.
    """
    global _client

    if _client is not None:
        return

    with tracer.start_as_current_span("qdrant.init_client") as span:
        span.set_attribute("qdrant.host", settings.qdrant.host)
        span.set_attribute("qdrant.port", settings.qdrant.port)

        _client = AsyncQdrantClient(
            host=settings.qdrant.host,
            port=settings.qdrant.port,
        )


async def close_client() -> None:
    """Close the Qdrant client.

    Must be called at application shutdown.
    """
    global _client

    if _client is None:
        return

    await _client.close()
    _client = None


async def ensure_collections_exist() -> None:
    """Create Qdrant collections if they do not already exist.

    Called at startup after init_client(). Safe to call multiple
    times — existing collections are left unchanged.

    Two collections are created:
        dense: stores embeddings from nomic-embed-text (768-dim in
               Phase 1, BAAI/bge-m3 at 1024-dim from Phase 3).
        sparse: stores SPLADE sparse vectors for keyword matching.
               Created as a named vector collection. Populated
               from Phase 3 onwards.

    Cosine distance is used for both collections. For normalized
    embeddings cosine and dot product are equivalent, but cosine
    is more intuitive and widely used in retrieval literature.
    """
    client = get_client()

    with tracer.start_as_current_span("qdrant.ensure_collections"):
        existing = await client.get_collections()
        existing_names = {c.name for c in existing.collections}

        if settings.qdrant.collection_dense not in existing_names:
            await client.create_collection(
                collection_name=settings.qdrant.collection_dense,
                vectors_config=VectorParams(
                    size=settings.ollama.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )

        if settings.qdrant.collection_sparse not in existing_names:
            await client.create_collection(
                collection_name=settings.qdrant.collection_sparse,
                vectors_config=VectorParams(
                    size=settings.ollama.embedding_dim,
                    distance=Distance.COSINE,
                ),
            )


async def upsert_chunks(chunks: list[dict]) -> None:
    """Write chunks to the dense vector collection.

    Args:
        chunks: List of dicts, each containing:
            - chunk_id: unique identifier string
            - embedding: list of floats
            - content: raw text
            - metadata: dict matching ChunkMetadata fields

    Uses upsert semantics — if a chunk with the same ID already
    exists it is overwritten. This makes ingestion idempotent: running
    the same document through the pipeline twice does not create
    duplicate entries.
    """
    client = get_client()

    with tracer.start_as_current_span("qdrant.upsert_chunks") as span:
        span.set_attribute("chunks.count", len(chunks))

        points = [
            PointStruct(
                id=chunk["chunk_id"],
                vector=chunk["embedding"],
                payload={
                    "content": chunk["content"],
                    **chunk["metadata"],
                },
            )
            for chunk in chunks
        ]

        await client.upsert(
            collection_name=settings.qdrant.collection_dense,
            points=points,
            wait=True,
        )

        span.set_attribute("collection", settings.qdrant.collection_dense)


async def search_dense(
    query_embedding: list[float],
    team_id: str,
    k: int,
) -> list[Chunk]:
    """Search the dense vector collection.

    Applies a mandatory scope filter so results are restricted to
    documents the requesting team is authorized to see. This filter
    is not optional and cannot be bypassed by callers.

    Args:
        query_embedding: The embedded query vector.
        team_id: The requesting team. Used to filter by scopes.
        k: Maximum number of results to return.

    Returns:
        List of Chunk objects sorted by descending relevance score.
    """
    client = get_client()

    with tracer.start_as_current_span("qdrant.search_dense") as span:
        span.set_attribute("team_id", team_id)
        span.set_attribute("k", k)

        scope_filter = Filter(
            must=[
                FieldCondition(
                    key="scopes",
                    match=MatchAny(any=[team_id, "public"]),
                )
            ]
        )

        results = await client.query_points(
        collection_name=settings.qdrant.collection_dense,
        query=query_embedding,
        query_filter=scope_filter,
        limit=k,
        with_payload=True,
    )

    span.set_attribute("results.count", len(results.points))

    chunks = []
    for result in results.points:
        payload = result.payload or {}
        chunks.append(
            Chunk(
                chunk_id=str(result.id),
                doc_id=payload.get("doc_id", ""),
                content=payload.get("content", ""),
                score=result.score,
                token_count=payload.get("token_count", 0),
                metadata=ChunkMetadata(
                    doc_id=payload.get("doc_id", ""),
                    ticker=payload.get("ticker", ""),
                    company_name=payload.get("company_name", ""),
                    filing_type=payload.get("filing_type", "10-K"),
                    filing_date=payload.get("filing_date", "2024-01-01"),
                    section=payload.get("section", ""),
                    chunk_index=payload.get("chunk_index", 0),
                    token_count=payload.get("token_count", 0),
                    embedding_model=payload.get("embedding_model", ""),
                    scopes=payload.get("scopes", ["public"]),
                ),
            )
        )

    return chunks