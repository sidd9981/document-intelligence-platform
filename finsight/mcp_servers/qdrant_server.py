"""
MCP Qdrant server.

Exposes vector_search and sparse_search as MCP-invokable tools.
Validates the JWT on every call — this is the second data isolation
layer after the gateway scope check.

The retrieval agent calls this instead of hitting Qdrant directly.
Keeping Qdrant access behind an MCP server means you can swap the
vector store without touching agent code.

Port 8102.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from finsight.auth.token_validator import decode_token, require_scope
from finsight.models.base import Chunk
from finsight.services.llm import close_client, embed, init_client
from finsight.services.sparse_encoder import close_encoder, encode_sparse, init_encoder
from finsight.services.vector_store import (
    close_client as close_qdrant,
    ensure_collections_exist,
    init_client as init_qdrant,
    search_dense,
    search_sparse,
)
from finsight.telemetry.tracing import get_tracer, setup_tracing

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_tracing()
    await init_qdrant()
    await ensure_collections_exist()
    await init_client()
    init_encoder()
    logger.info("qdrant mcp server ready")
    yield
    close_encoder()
    await close_client()
    await close_qdrant()
    logger.info("qdrant mcp server shutdown")


app = FastAPI(title="FinSight MCP Qdrant", version="0.1.0", lifespan=lifespan)


class VectorSearchRequest(BaseModel):
    query: str
    team_id: str
    k: int = 20


class SparseSearchRequest(BaseModel):
    query: str
    team_id: str
    k: int = 20


class SearchResponse(BaseModel):
    chunks: list[Chunk]


def _check_filing_scope(credentials: HTTPAuthorizationCredentials) -> dict:
    try:
        payload = decode_token(credentials.credentials)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"invalid token: {e}")
    scopes = set(payload.get("scopes", []))
    if "read:public_filings" not in scopes and "read:all_filings" not in scopes:
        raise HTTPException(status_code=403, detail="missing required filing scope")
    return payload


@app.post("/invoke/vector_search", response_model=SearchResponse)
async def vector_search(
    request: VectorSearchRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> SearchResponse:
    """Embed the query and run dense vector search against Qdrant."""
    _check_filing_scope(credentials)

    with tracer.start_as_current_span("mcp_qdrant.vector_search") as span:
        span.set_attribute("team_id", request.team_id)
        span.set_attribute("k", request.k)

        embedding = await embed(request.query)
        chunks = await search_dense(embedding, request.team_id, request.k)

        span.set_attribute("chunks_returned", len(chunks))
        return SearchResponse(chunks=chunks)


@app.post("/invoke/sparse_search", response_model=SearchResponse)
async def sparse_search(
    request: SparseSearchRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> SearchResponse:
    """Encode the query as a SPLADE sparse vector and run keyword search."""
    _check_filing_scope(credentials)

    with tracer.start_as_current_span("mcp_qdrant.sparse_search") as span:
        span.set_attribute("team_id", request.team_id)
        span.set_attribute("k", request.k)

        sparse_vector = encode_sparse(request.query)
        chunks = await search_sparse(sparse_vector, request.team_id, request.k)

        span.set_attribute("chunks_returned", len(chunks))
        return SearchResponse(chunks=chunks)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}