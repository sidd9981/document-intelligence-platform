"""
MCP reranker server.

Exposes the rerank tool. Takes a query and a list of chunks from RRF
fusion and rescores them using a cross-encoder. The cross-encoder
attends to query and chunk content jointly so it's more accurate than
the retrieval scores alone.

Only called on the top 50 RRF candidates — too slow for full corpus
search. Returns top_k chunks sorted by cross-encoder score.

Port 8105.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from finsight.auth.token_validator import decode_token
from finsight.models.base import Chunk
from finsight.services.reranker import close_reranker, init_reranker, rerank
from finsight.telemetry.tracing import get_tracer, setup_tracing

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)
security = HTTPBearer()


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_tracing()
    init_reranker()
    logger.info("reranker mcp server ready")
    yield
    close_reranker()
    logger.info("reranker mcp server shutdown")


app = FastAPI(title="FinSight MCP Reranker", version="0.1.0", lifespan=lifespan)


class RerankRequest(BaseModel):
    query: str
    chunks: list[Chunk]
    top_k: int = 10


class RerankResponse(BaseModel):
    chunks: list[Chunk]


def _require_filing_scope(credentials: HTTPAuthorizationCredentials) -> dict:
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


@app.post("/invoke/rerank", response_model=RerankResponse)
async def rerank_chunks(
    request: RerankRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> RerankResponse:
    """Rescore RRF candidates by joint query-chunk relevance."""
    _require_filing_scope(credentials)

    with tracer.start_as_current_span("mcp_reranker.rerank") as span:
        span.set_attribute("candidates", len(request.chunks))
        span.set_attribute("top_k", request.top_k)

        reranked = rerank(request.query, request.chunks, request.top_k)

        span.set_attribute("returned", len(reranked))
        return RerankResponse(chunks=reranked)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}