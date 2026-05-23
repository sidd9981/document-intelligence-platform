"""
FastAPI application and query endpoint.

Phase 1 exposes a single endpoint: POST /query. It accepts a natural
language question and a team_id, retrieves relevant chunks from
Qdrant, and returns a grounded answer from the LLM.

This is the thin end-to-end slice that proves the core loop works.
Subsequent phases add:
    - OAuth2 token validation (Phase 5)
    - Multi-agent orchestration via LangGraph (Phase 4)
    - Token budget enforcement (Phase 5)
    - WebSocket streaming (Phase 5)
    - MCP tool servers (Phase 5)

The lifespan function handles startup and shutdown of all shared
resources. FastAPI guarantees lifespan runs before any request is
served and after the last request completes.
"""

import uuid
from contextlib import asynccontextmanager
from datetime import datetime
from typing import AsyncGenerator

import structlog
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel

from finsight.config.settings import settings
from finsight.gateway.db import close_pool, get_pool, init_pool
from finsight.models.synthesis import QueryResponse
from finsight.services.llm import close_client, complete, embed, init_client
from finsight.services.vector_store import (
    close_client as close_qdrant,
    ensure_collections_exist,
    init_client as init_qdrant,
    search_dense,
)
from finsight.telemetry.tracing import get_tracer, setup_tracing

log = structlog.get_logger()
tracer = get_tracer(__name__)


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    """Manage startup and shutdown of all shared resources.

    FastAPI calls the code before yield at startup and the code
    after yield at shutdown. This is the correct place to initialize
    connection pools and clients — not at module import time, which
    would run during testing and require live services.

    Order matters on startup:
        1. Tracing must be initialized first so all subsequent
           operations are traced including pool initialization.
        2. Database pool before any operation that writes to Postgres.
        3. Qdrant client before ensuring collections exist.
        4. LLM client last since it has no dependencies.

    Order matters on shutdown:
        Reverse of startup. Close dependents before dependencies.
    """
    setup_tracing()

    await init_pool()
    await init_qdrant()
    await ensure_collections_exist()
    await init_client()

    log.info(
        "startup complete",
        env=settings.app.env,
        ollama_model=settings.ollama.model,
        embedding_model=settings.ollama.embedding_model,
    )

    yield

    await close_client()
    await close_qdrant()
    await close_pool()

    log.info("shutdown complete")


app = FastAPI(
    title="FinSight",
    description="Enterprise document intelligence platform",
    version="0.1.0",
    lifespan=lifespan,
)


class QueryRequest(BaseModel):
    """Incoming query from a client.

    team_id identifies which tenant is making the request. In Phase 1
    this is passed directly in the request body. Phase 5 replaces this
    with OAuth2 token validation where team_id is extracted from the
    JWT claims — the client no longer declares their own identity.
    """

    query: str
    team_id: str


SYSTEM_PROMPT = """You are a financial document analyst. Answer the user's
question using only the context provided below. If the context does not
contain enough information to answer the question, say so clearly. Do not
use any knowledge outside of the provided context. Cite specific details
from the context in your answer."""


@app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest) -> QueryResponse:
    """Answer a natural language question about financial documents.

    Retrieves relevant chunks from Qdrant using dense vector search,
    constructs a prompt with those chunks as context, and returns the
    LLM's answer with metadata.

    In Phase 1 this endpoint does not enforce token budgets or validate
    OAuth tokens. Those are added in Phase 5. The structure here is
    intentionally compatible with Phase 5 — the request and response
    models will not change, only the middleware around this handler.

    Args:
        request: Contains the query text and team_id.

    Returns:
        QueryResponse with the answer, citations placeholder,
        faithfulness score placeholder, and request metadata.

    Raises:
        HTTPException 400: If the query is empty.
        HTTPException 404: If no relevant context is found.
        HTTPException 500: If the LLM call fails.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    trace_id = str(uuid.uuid4())

    with tracer.start_as_current_span("gateway.query") as span:
        span.set_attribute("trace_id", trace_id)
        span.set_attribute("team_id", request.team_id)
        span.set_attribute("query_length", len(request.query))

        query_embedding = await embed(request.query)

        tenant_config = await _load_tenant_config(request.team_id)
        k = tenant_config["retrieval_k"] if tenant_config else 5

        chunks = await search_dense(
            query_embedding=query_embedding,
            team_id=request.team_id,
            k=k,
        )

        span.set_attribute("chunks_retrieved", len(chunks))

        if not chunks:
            raise HTTPException(
                status_code=404,
                detail="no relevant context found for this query",
            )

        context = "\n\n---\n\n".join(
            f"[Source: {c.metadata.ticker} {c.metadata.filing_type} "
            f"{c.metadata.filing_date} | Section: {c.metadata.section}]\n"
            f"{c.content}"
            for c in chunks
        )

        prompt = f"Context:\n{context}\n\nQuestion: {request.query}"

        answer, prompt_tokens, completion_tokens = await complete(
            prompt=prompt,
            system=SYSTEM_PROMPT,
            max_tokens=tenant_config["max_output_tokens"] if tenant_config else 500,
        )

        span.set_attribute("prompt_tokens", prompt_tokens)
        span.set_attribute("completion_tokens", completion_tokens)

        await _log_query(
            trace_id=trace_id,
            team_id=request.team_id,
            query_text=request.query,
            chunks_retrieved=len(chunks),
            model_used=settings.ollama.model,
            total_tokens=prompt_tokens + completion_tokens,
        )

        log.info(
            "query completed",
            trace_id=trace_id,
            team_id=request.team_id,
            chunks_retrieved=len(chunks),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
        )

        return QueryResponse(
            trace_id=trace_id,
            answer=answer,
            citations=[],
            faithfulness_score=0.0,
            model_used=settings.ollama.model,
            latency_ms=0.0,
            cache_hit=False,
        )


async def _load_tenant_config(team_id: str) -> dict | None:
    """Load tenant configuration from Postgres.

    Returns None if the team_id is not found rather than raising
    so the caller can fall back to defaults. Phase 5 makes this
    mandatory — an unknown team_id will be rejected at the OAuth
    validation layer before reaching this function.

    Args:
        team_id: The team identifier to look up.

    Returns:
        Dict with tenant config fields, or None if not found.
    """
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM tenant_configs WHERE team_id = $1",
            team_id,
        )
        return dict(row) if row else None


async def _log_query(
    trace_id: str,
    team_id: str,
    query_text: str,
    chunks_retrieved: int,
    model_used: str,
    total_tokens: int,
) -> None:
    """Write a query record to the audit log.

    Fire-and-forget — logged after the response is constructed so
    logging latency does not affect the user-facing response time.
    Failures are caught and logged but do not raise to the caller.

    Args:
        trace_id: The request trace ID.
        team_id: The requesting team.
        query_text: The original query string.
        chunks_retrieved: Number of chunks returned by retrieval.
        model_used: The LLM model name used for this query.
        total_tokens: Total tokens consumed by this query.
    """
    try:
        import uuid as uuid_module
        pool = get_pool()
        async with pool.acquire() as conn:
            await conn.execute(
                """
                INSERT INTO query_audit_log (
                    trace_id, team_id, query_text,
                    retrieval_method, chunks_retrieved,
                    model_used, total_tokens, status
                ) VALUES ($1, $2, $3, $4, $5, $6, $7, $8)
                """,
                uuid_module.UUID(trace_id),
                team_id,
                query_text,
                "dense",
                chunks_retrieved,
                model_used,
                total_tokens,
                "success",
            )
    except Exception as exc:
        log.error("failed to write audit log", error=str(exc), trace_id=trace_id)


@app.get("/health")
async def health() -> dict:
    """Health check endpoint.

    Returns 200 if the application is running. Does not check
    downstream service health — that is the responsibility of
    the Docker health checks on each service container.
    """
    return {"status": "ok", "env": settings.app.env}