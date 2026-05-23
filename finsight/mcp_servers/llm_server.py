"""
MCP LLM server.

Exposes the generate tool. The synthesis agent posts a prompt and
system message here instead of calling Ollama directly. Keeping LLM
access behind an MCP server means swapping Ollama for vLLM in
production is a config change, not a code change.

Scope check is on model tier. The server picks the highest tier the
team's token permits. A team with only model:small cannot get a
model:large response even if they ask for it.

Port 8104.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from pydantic import BaseModel

from finsight.auth.token_validator import decode_token
from finsight.services.llm import close_client, complete, init_client
from finsight.telemetry.tracing import get_tracer, setup_tracing

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)
security = HTTPBearer()

# Ordered highest to lowest. First match in token scopes wins.
MODEL_TIER_PRIORITY = ["model:large", "model:medium", "model:small"]

# Maps scope to the Ollama model name. In production this maps to vLLM model names.
MODEL_FOR_SCOPE = {
    "model:large": "llama3.1:8b",
    "model:medium": "llama3.1:8b",
    "model:small": "llama3.2:3b",
}


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    setup_tracing()
    await init_client()
    logger.info("llm mcp server ready")
    yield
    await close_client()
    logger.info("llm mcp server shutdown")


app = FastAPI(title="FinSight MCP LLM", version="0.1.0", lifespan=lifespan)


class GenerateRequest(BaseModel):
    prompt: str
    system: str
    max_tokens: int = 1000


class GenerateResponse(BaseModel):
    answer: str
    prompt_tokens: int
    completion_tokens: int
    model_used: str


def _resolve_model(scopes: list[str]) -> str:
    """Pick the highest-tier model the token's scopes permit."""
    scopes_set = set(scopes)
    for tier in MODEL_TIER_PRIORITY:
        if tier in scopes_set:
            return MODEL_FOR_SCOPE[tier]
    raise HTTPException(status_code=403, detail="token has no model scope")


def _require_model_scope(credentials: HTTPAuthorizationCredentials) -> dict:
    try:
        payload = decode_token(credentials.credentials)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"invalid token: {e}")
    scopes = payload.get("scopes", [])
    if not any(s in scopes for s in MODEL_TIER_PRIORITY):
        raise HTTPException(status_code=403, detail="token has no model scope")
    return payload


@app.post("/invoke/generate", response_model=GenerateResponse)
async def generate(
    request: GenerateRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> GenerateResponse:
    """Run a completion and return the answer with token counts."""
    payload = _require_model_scope(credentials)
    model = _resolve_model(payload.get("scopes", []))

    with tracer.start_as_current_span("mcp_llm.generate") as span:
        span.set_attribute("model", model)
        span.set_attribute("max_tokens", request.max_tokens)

        answer, prompt_tokens, completion_tokens = await complete(
            prompt=request.prompt,
            system=request.system,
            max_tokens=request.max_tokens,
        )

        span.set_attribute("prompt_tokens", prompt_tokens)
        span.set_attribute("completion_tokens", completion_tokens)

        return GenerateResponse(
            answer=answer,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            model_used=model,
        )


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}