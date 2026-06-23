"""
FastAPI gateway.

JWT validation, orchestrator wiring, token budget enforcement, and
streaming. team_id is extracted from the JWT claims rather than passed in
the request body, so clients cannot declare their own identity.

Conversation history supplied by the client is untrusted. It is length
bounded, injection scanned, and assembled into a synthesis-only context.
The raw current query is what gets guardrailed, retrieved on, and audited.
"""

from __future__ import annotations

import time
import time as _time
import uuid
from contextlib import asynccontextmanager
from datetime import date
from typing import AsyncGenerator

import jwt
import redis.asyncio as aioredis
import structlog
from fastapi import Depends, FastAPI, Form, HTTPException
from fastapi.responses import JSONResponse, StreamingResponse
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from neo4j import AsyncGraphDatabase
from prometheus_client import make_asgi_app
from pydantic import BaseModel, Field

from finsight.agents.graph_agent import GraphAgent
from finsight.agents.orchestrator import Orchestrator
from finsight.agents.retrieval_agent import RetrievalAgent
from finsight.agents.synthesis_agent import SynthesisAgent
from finsight.auth.scope_definitions import DEV_CLIENTS, TEAM_SCOPES
from finsight.auth.token_validator import (
    JWT_ALGORITHM,
    JWT_SECRET,
    decode_token,
    get_team_id,
)
from finsight.config.settings import settings
from finsight.gateway.db import close_pool, get_pool, init_pool
from finsight.harness.input_harness import INJECTION_PATTERNS
from finsight.models.synthesis import QueryResponse
from finsight.models.tenant import TenantConfig
from finsight.services.circuit_breaker import CircuitBreaker
from finsight.services.guardrails import check_query
from finsight.services.llm import close_client, init_client
from finsight.services.metrics import (
    cache_hits_total,
    cache_misses_total,
    query_latency,
)
from finsight.services.reranker import close_reranker, init_reranker
from finsight.services.sparse_encoder import close_encoder, init_encoder
from finsight.services.vector_store import (
    close_client as close_qdrant,
    ensure_collections_exist,
    init_client as init_qdrant,
)
from finsight.telemetry.tracing import get_tracer, setup_tracing

log = structlog.get_logger()
tracer = get_tracer(__name__)
security = HTTPBearer()

MAX_QUERY_CHARS = 4000
MAX_TURN_CHARS = 4000
MAX_HISTORY_TURNS = 3
MAX_HISTORY_ITEMS = 20

app = FastAPI(
    title="FinSight",
    description="Enterprise document intelligence platform",
    version="0.1.0",
)

metrics_app = make_asgi_app()
app.mount("/metrics", metrics_app)

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


class ConversationTurn(BaseModel):
    """A single prior exchange supplied by the client for context.

    Both fields are client controlled and therefore untrusted. The length
    caps bound prompt size so a client cannot inflate token cost or push the
    synthesis prompt past a tenant's context window.
    """

    query: str = Field(max_length=MAX_TURN_CHARS)
    answer: str = Field(max_length=MAX_TURN_CHARS)


class QueryRequest(BaseModel):
    query: str = Field(max_length=MAX_QUERY_CHARS)
    history: list[ConversationTurn] = Field(default_factory=list, max_length=MAX_HISTORY_ITEMS)


def _build_conversation_context(query: str, history: list[ConversationTurn]) -> str:
    """Assemble the synthesis context from validated conversation history.

    History is client supplied and untrusted. The current query is guardrailed
    separately by the caller; this function applies the same injection scan the
    input harness runs on retrieved chunks to every historical turn before any
    of it can reach the LLM. We fail closed: a single injection hit rejects the
    whole request.

    Only the most recent MAX_HISTORY_TURNS turns are included. Returns the bare
    query when there is no history so a single-turn request pays no wrapping cost.
    """
    if not history:
        return query

    recent = history[-MAX_HISTORY_TURNS:]

    for turn in recent:
        for text in (turn.query, turn.answer):
            for pattern in INJECTION_PATTERNS:
                if pattern.search(text):
                    raise HTTPException(
                        status_code=422,
                        detail={
                            "message": "conversation history contains disallowed content",
                            "violation_type": "history_injection",
                        },
                    )

    history_text = "\n".join(
        f"User: {turn.query}\nAssistant: {turn.answer}" for turn in recent
    )
    return f"Previous conversation:\n{history_text}\n\nCurrent question: {query}"


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


async def _meter_budget(team_id: str) -> None:
    """Increment the team's daily token counter in Redis.

    Best effort. A metering failure must never fail the user's request, so
    every exception is logged and swallowed. The counter expires at the end
    of its day so budgets reset automatically.
    """
    try:
        r_budget = aioredis.from_url(
            f"redis://localhost:6379/{settings.redis.budget_db}",
            decode_responses=True,
        )
        key = f"budget:{team_id}:{date.today().isoformat()}"
        await r_budget.incrby(key, 1000)
        await r_budget.expire(key, 86400)
        await r_budget.aclose()
    except Exception as e:
        log.warning("budget metering failed", error=str(e))


@app.post("/query", response_model=QueryResponse)
async def query(
    request: QueryRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> QueryResponse:
    """Answer a question, optionally with prior conversation context.

    The raw query is guardrailed and drives retrieval and audit. History is
    validated, injection scanned, and assembled into a synthesis-only context.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    guardrail = check_query(request.query)
    if not guardrail.allowed:
        raise HTTPException(
            status_code=422,
            detail={
                "message": guardrail.reason,
                "violation_type": guardrail.violation_type,
            },
        )

    payload = decode_token(credentials.credentials)
    team_id = get_team_id(payload)
    tenant_config = await _get_tenant_config(team_id)

    conversation_context = _build_conversation_context(request.query, request.history)

    start = _time.perf_counter()
    status = "success"
    result: QueryResponse | None = None

    with tracer.start_as_current_span("gateway.query") as span:
        span.set_attribute("team_id", team_id)
        span.set_attribute("query_length", len(request.query))

        if _orchestrator is None:
            raise HTTPException(status_code=503, detail="orchestrator not initialized")

        try:
            result = await _orchestrator.run(
                query=request.query,
                tenant_config=tenant_config,
                conversation_context=conversation_context,
            )
            await _meter_budget(team_id)
        except Exception:
            status = "error"
            raise
        finally:
            latency = _time.perf_counter() - start
            query_latency.labels(team_id=team_id, intent="unknown", status=status).observe(latency)
            if result and result.cache_hit:
                cache_hits_total.labels(team_id=team_id).inc()
            else:
                cache_misses_total.labels(team_id=team_id).inc()

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
    """Stream a grounded answer as server-sent events.

    This routes through the full orchestrator exactly like /query, so the
    answer is retrieved, harnessed, and faithfulness scored. The completed
    answer is then streamed back word by word as SSE so the client gets a
    progressive render. Each chunk arrives as 'data: <word> \\n\\n' and the
    stream ends with 'data: [DONE]\\n\\n'.

    Streaming raw LLM tokens directly would bypass retrieval and the harnesses,
    producing ungrounded output, so we do not do that here.
    """
    if not request.query.strip():
        raise HTTPException(status_code=400, detail="query must not be empty")

    guardrail = check_query(request.query)
    if not guardrail.allowed:
        raise HTTPException(
            status_code=422,
            detail={
                "message": guardrail.reason,
                "violation_type": guardrail.violation_type,
            },
        )

    payload = decode_token(credentials.credentials)
    team_id = get_team_id(payload)
    tenant_config = await _get_tenant_config(team_id)

    conversation_context = _build_conversation_context(request.query, request.history)

    if _orchestrator is None:
        raise HTTPException(status_code=503, detail="orchestrator not initialized")

    async def event_stream() -> AsyncGenerator[str, None]:
        start = _time.perf_counter()
        status = "success"
        result: QueryResponse | None = None

        with tracer.start_as_current_span("gateway.query_stream") as span:
            span.set_attribute("team_id", team_id)
            span.set_attribute("query_length", len(request.query))

            try:
                result = await _orchestrator.run(
                    query=request.query,
                    tenant_config=tenant_config,
                    conversation_context=conversation_context,
                )
                for word in result.answer.split(" "):
                    yield f"data: {word} \n\n"
                await _meter_budget(team_id)
            except Exception as e:
                status = "error"
                log.error("stream error", error=str(e))
                yield "data: [ERROR]\n\n"
            finally:
                latency = _time.perf_counter() - start
                query_latency.labels(
                    team_id=team_id, intent="unknown", status=status
                ).observe(latency)
                if result and result.cache_hit:
                    cache_hits_total.labels(team_id=team_id).inc()
                else:
                    cache_misses_total.labels(team_id=team_id).inc()
                if result:
                    await _log_query(
                        trace_id=result.trace_id,
                        team_id=team_id,
                        query_text=request.query,
                        chunks_retrieved=len(result.citations),
                        model_used=result.model_used,
                        total_tokens=0,
                    )
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


@app.get("/budget/{team_id}")
async def get_budget(
    team_id: str,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> dict:
    payload = decode_token(credentials.credentials)
    if get_team_id(payload) != team_id:
        raise HTTPException(status_code=403, detail="forbidden")

    tenant_config = await _get_tenant_config(team_id)

    r = aioredis.from_url(
        f"redis://localhost:6379/{settings.redis.budget_db}",
        decode_responses=True,
    )
    key = f"budget:{team_id}:{date.today().isoformat()}"
    used = int(await r.get(key) or 0)
    await r.aclose()

    return {
        "team_id": team_id,
        "used": used,
        "limit": tenant_config.daily_token_budget,
        "pct": round(used / tenant_config.daily_token_budget * 100, 1),
    }