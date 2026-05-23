"""
FastAPI gateway.

Phase 5 upgrade: JWT validation, orchestrator wiring, token budget
enforcement. The Phase 1 direct dense search is replaced by the full
LangGraph orchestrator pipeline.

team_id is now extracted from the JWT claims rather than passed in
the request body. Clients no longer declare their own identity.
"""

from __future__ import annotations
from finsight.services.sparse_encoder import init_encoder, close_encoder
from finsight.services.reranker import init_reranker, close_reranker
from fastapi.responses import StreamingResponse
from neo4j import AsyncGraphDatabase
import redis.asyncio as aioredis
from finsight.services.circuit_breaker import CircuitBreaker

import uuid
from contextlib import asynccontextmanager
from typing import AsyncGenerator

import structlog
from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel
from fastapi import Form
from fastapi.responses import JSONResponse
import time
import jwt
from finsight.auth.scope_definitions import DEV_CLIENTS, TEAM_SCOPES
from finsight.auth.token_validator import JWT_SECRET, JWT_ALGORITHM

from finsight.agents.graph_agent import GraphAgent
from finsight.agents.orchestrator import Orchestrator
from finsight.agents.retrieval_agent import RetrievalAgent
from finsight.agents.synthesis_agent import SynthesisAgent
from finsight.auth.token_validator import decode_token, get_team_id
from finsight.config.settings import settings
from finsight.gateway.db import close_pool, get_pool, init_pool
from finsight.models.synthesis import QueryResponse
from finsight.models.tenant import TenantConfig
from finsight.services.llm import close_client, init_client
from finsight.services.vector_store import (
    close_client as close_qdrant,
    ensure_collections_exist,
    init_client as init_qdrant,
)
from finsight.telemetry.tracing import get_tracer, setup_tracing

log = structlog.get_logger()
tracer = get_tracer(__name__)
security = HTTPBearer()

app = FastAPI(
    title="FinSight",
    description="Enterprise document intelligence platform",
    version="0.1.0",
)

_orchestrator: Orchestrator | None = None


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_tracing()

    await init_pool()
    await init_qdrant()
    await ensure_collections_exist()
    await init_client()
    init_encoder()
    init_reranker()


    redis_client = aioredis.from_url(
        f"redis://localhost:6379/{settings.redis.cache_db}",
        decode_responses=False,
    )

    neo4j_driver = AsyncGraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "changeme"),
    )

    global _orchestrator
    _orchestrator = Orchestrator(
        retrieval_agent=RetrievalAgent(
            redis_client=redis_client,
            breaker=CircuitBreaker(name="qdrant", failure_threshold=5, recovery_timeout=30.0),
        ),
        graph_agent=GraphAgent(
            driver=neo4j_driver,
            breaker=CircuitBreaker(name="neo4j", failure_threshold=5, recovery_timeout=30.0),
        ),
        synthesis_agent=SynthesisAgent(),
    )

    log.info("startup complete", env=settings.app.env, model=settings.ollama.model)
    yield

    await close_client()
    await close_qdrant()
    await close_pool()
    close_encoder()
    close_reranker()
    await neo4j_driver.close()
    log.info("shutdown complete")


app.router.lifespan_context = lifespan


app.router.lifespan_context = lifespan


class QueryRequest(BaseModel):
    query: str


async def _get_tenant_config(team_id: str) -> TenantConfig:
    """Load TenantConfig from Postgres. Raises 404 if team not found."""
    pool = get_pool()
    async with pool.acquire() as conn:
        row = await conn.fetchrow(
            "SELECT * FROM tenant_configs WHERE team_id = $1",
            team_id,
        )
    if not row:
        raise HTTPException(status_code=404, detail=f"team {team_id} not configured")

    return TenantConfig(
        team_id=row["team_id"],
        daily_token_budget=row["daily_token_budget"],
        max_context_tokens=row["max_context_tokens"],
        max_output_tokens=row["max_output_tokens"],
        requests_per_minute=row["requests_per_minute"],
        priority=row["priority"],
        allowed_models=row["allowed_models"],
        retrieval_k=row["retrieval_k"],
        data_scopes=row["data_scopes"],
    )


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> QueryResponse:
    """Answer a natural language question about financial documents.

    Validates the JWT, loads tenant config, and runs the full
    orchestrator pipeline. Returns a cited, auditable answer.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    payload = decode_token(credentials.credentials)
    team_id = get_team_id(payload)
    tenant_config = await _get_tenant_config(team_id)

    with tracer.start_as_current_span("gateway.query") as span:
        span.set_attribute("team_id", team_id)
        span.set_attribute("query_length", len(request.query))

        if _orchestrator is None:
            raise HTTPException(status_code=503, detail="orchestrator not initialized")

        result = await _orchestrator.run(
            query=request.query,
            tenant_config=tenant_config,
        )

        await _log_query(
            trace_id=result.trace_id,
            team_id=team_id,
            query_text=request.query,
            chunks_retrieved=len(result.citations),
            model_used=result.model_used,
            total_tokens=0,
        )

        return result
    
@app.post("/query/stream")
async def query_stream(
    request: QueryRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> StreamingResponse:
    """Stream the answer token by token as server-sent events.

    The client receives a text/event-stream response. Each token
    arrives as 'data: <token>\n\n'. The stream ends with
    'data: [DONE]\n\n'. This is standard SSE format — works in
    any browser EventSource or curl with no special client library.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    payload = decode_token(credentials.credentials)
    team_id = get_team_id(payload)
    tenant_config = await _get_tenant_config(team_id)

    async def event_stream():
        from finsight.services.llm import stream_complete

        system = "You are a financial analyst. Answer using only the provided context. Be concise and cite sources."

        with tracer.start_as_current_span("gateway.query_stream") as span:
            span.set_attribute("team_id", team_id)
            span.set_attribute("query_length", len(request.query))

            try:
                async for token in stream_complete(
                    prompt=request.query,
                    system=system,
                    max_tokens=tenant_config.max_output_tokens,
                ):
                    yield f"data: {token}\n\n"
            except Exception as e:
                log.error("stream error", error=str(e))
                yield f"data: [ERROR]\n\n"
            finally:
                yield "data: [DONE]\n\n"

    return StreamingResponse(event_stream(), media_type="text/event-stream")


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "env": settings.app.env}


async def _log_query(
    trace_id: str,
    team_id: str,
    query_text: str,
    chunks_retrieved: int,
    model_used: str,
    total_tokens: int,
) -> None:
    try:
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
                uuid.UUID(trace_id),
                team_id,
                query_text,
                "hybrid",
                chunks_retrieved,
                model_used,
                total_tokens,
                "success",
            )
    except Exception as exc:
        log.error("failed to write audit log", error=str(exc))

@app.post("/oauth/token")
async def issue_token(
    grant_type: str = Form(...),
    client_id: str = Form(...),
    client_secret: str = Form(...),
) -> JSONResponse:
    if grant_type != "client_credentials":
        raise HTTPException(status_code=400, detail="unsupported_grant_type")

    expected_secret = DEV_CLIENTS.get(client_id)
    if not expected_secret or expected_secret != client_secret:
        raise HTTPException(status_code=401, detail="invalid_client")

    scopes = TEAM_SCOPES.get(client_id, [])
    now = int(time.time())
    payload = {
        "sub": f"team_{client_id}",
        "team_id": client_id,
        "scopes": scopes,
        "iat": now,
        "exp": now + 3600,
    }
    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    return JSONResponse({
        "access_token": token,
        "token_type": "bearer",
        "expires_in": 3600,
        "scope": " ".join(scopes),
    })