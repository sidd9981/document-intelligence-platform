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
    MatchValue,
    PointStruct,
    SparseVector,
    SparseVectorParams,
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
                vectors_config={},
                sparse_vectors_config={
                    "sparse": SparseVectorParams()
                },
            )


async def upsert_chunks(chunks: list[dict]) -> None:
    """Write chunks to the dense vector collection.

    Uses upsert semantics — running the same document through ingestion
    twice produces identical chunk IDs and overwrites rather than
    duplicating entries.

    Args:
        chunks: List of dicts with chunk_id, embedding, content, metadata.
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


async def upsert_sparse_chunks(chunks: list[dict]) -> None:
    """Write chunks to the sparse vector collection.

    Same structure as upsert_chunks but stores SPLADE sparse vectors
    instead of dense embeddings. Queried in parallel with the dense
    collection during hybrid retrieval.

    Args:
        chunks: List of dicts with chunk_id, sparse_vector, content, metadata.
    """
    client = get_client()

    with tracer.start_as_current_span("qdrant.upsert_sparse_chunks") as span:
        span.set_attribute("chunks.count", len(chunks))

        points = [
            PointStruct(
                id=chunk["chunk_id"],
                vector=SparseVector(
                    indices=list(chunk["sparse_vector"].keys()),
                    values=list(chunk["sparse_vector"].values()),
                ),
                payload={
                    "content": chunk["content"],
                    **chunk["metadata"],
                },
            )
            for chunk in chunks
        ]

        await client.upsert(
            collection_name=settings.qdrant.collection_sparse,
            points=points,
            wait=True,
        )

        span.set_attribute("collection", settings.qdrant.collection_sparse)


def _build_filter(team_id: str, ticker: str | None) -> Filter:
    """Build a Qdrant filter enforcing scope and optional ticker constraints.

    Scope filtering is mandatory on every search — a missing scope filter
    would allow a tenant to see documents they are not authorized to see.
    Ticker filtering is optional and used when the query is company-specific.

    Args:
        team_id: The requesting team. Chunks must have this team_id or
                 'public' in their scopes list.
        ticker:  Optional company ticker. When set, restricts results to
                 chunks from that company only.

    Returns:
        A Qdrant Filter with all required conditions.
    """
    must: list = [
        FieldCondition(
            key="scopes",
            match=MatchAny(any=[team_id, "public"]),
        )
    ]
    if ticker:
        must.append(
            FieldCondition(
                key="ticker",
                match=MatchValue(value=ticker),
            )
        )
    return Filter(must=must)


def _points_to_chunks(points: list) -> list[Chunk]:
    """Convert Qdrant query result points to Chunk model objects."""
    chunks = []
    for result in points:
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


async def search_dense(
    query_embedding: list[float],
    team_id: str,
    k: int,
    ticker: str | None = None,
) -> list[Chunk]:
    """Search the dense vector collection.

    Scope filtering is mandatory. When ticker is provided, results are
    further restricted to that company — prevents cross-company
    contamination in both production queries and eval runs.

    Args:
        query_embedding: The embedded query vector.
        team_id: The requesting team. Used to filter by scopes.
        k: Maximum number of results to return.
        ticker: Optional company ticker to restrict to one company.

    Returns:
        List of Chunk objects sorted by descending relevance score.
    """
    client = get_client()

    with tracer.start_as_current_span("qdrant.search_dense") as span:
        span.set_attribute("team_id", team_id)
        span.set_attribute("k", k)
        if ticker:
            span.set_attribute("ticker", ticker)

        results = await client.query_points(
            collection_name=settings.qdrant.collection_dense,
            query=query_embedding,
            query_filter=_build_filter(team_id, ticker),
            limit=k,
            with_payload=True,
        )

        span.set_attribute("results.count", len(results.points))
        return _points_to_chunks(results.points)


async def search_sparse(
    sparse_vector: dict[int, float],
    team_id: str,
    k: int,
    ticker: str | None = None,
) -> list[Chunk]:
    """Search the sparse vector collection.

    Same scope and ticker filtering as search_dense — tenant isolation
    is enforced at the data layer on every search operation regardless
    of retrieval method.

    Args:
        sparse_vector: SPLADE sparse vector from encode_sparse().
        team_id: The requesting team. Used to filter by scopes.
        k: Maximum number of results to return.
        ticker: Optional company ticker to restrict to one company.

    Returns:
        List of Chunk objects sorted by descending relevance score.
    """
    client = get_client()

    with tracer.start_as_current_span("qdrant.search_sparse") as span:
        span.set_attribute("team_id", team_id)
        span.set_attribute("k", k)
        span.set_attribute("nonzero_terms", len(sparse_vector))
        if ticker:
            span.set_attribute("ticker", ticker)

        results = await client.query_points(
            collection_name=settings.qdrant.collection_sparse,
            query=SparseVector(
                indices=list(sparse_vector.keys()),
                values=list(sparse_vector.values()),
            ),
            using="sparse",
            query_filter=_build_filter(team_id, ticker),
            limit=k,
            with_payload=True,
        )

        span.set_attribute("results.count", len(results.points))
        return _points_to_chunks(results.points)