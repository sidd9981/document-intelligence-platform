"""
Tests for the streaming completion function and streaming gateway endpoint.

The llm streaming test mocks the OpenAI async stream. The gateway
streaming test mocks stream_complete directly.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from finsight.services import llm


@pytest.fixture(autouse=True)
def reset_llm():
    llm._client = None
    llm._tokenizer = None
    yield
    llm._client = None
    llm._tokenizer = None


def test_stream_complete_raises_before_init():
    """stream_complete must fail immediately if client not initialized."""
    import asyncio

    async def collect():
        tokens = []
        async for token in llm.stream_complete("hello", "system", 100):
            tokens.append(token)
        return tokens

    with pytest.raises(RuntimeError, match="not initialized"):
        asyncio.run(collect())


@pytest.mark.asyncio
async def test_stream_complete_yields_tokens():
    """stream_complete must yield each non-empty token from the stream."""
    import tiktoken
    llm._tokenizer = tiktoken.get_encoding("cl100k_base")

    async def fake_stream():
        for word in ["Apple", " revenue", " was", " $383B", "."]:
            chunk = MagicMock()
            chunk.choices[0].delta.content = word
            yield chunk

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=fake_stream())
    llm._client = mock_client

    tokens = []
    async for token in llm.stream_complete("What is Apple revenue?", "You are an analyst.", 100):
        tokens.append(token)

    assert tokens == ["Apple", " revenue", " was", " $383B", "."]


@pytest.mark.asyncio
async def test_stream_complete_skips_empty_tokens():
    """Empty token strings from the stream must not be yielded."""
    import tiktoken
    llm._tokenizer = tiktoken.get_encoding("cl100k_base")

    async def fake_stream():
        for word in ["Hello", "", None, " world"]:
            chunk = MagicMock()
            chunk.choices[0].delta.content = word
            yield chunk

    mock_client = AsyncMock()
    mock_client.chat.completions.create = AsyncMock(return_value=fake_stream())
    llm._client = mock_client

    tokens = []
    async for token in llm.stream_complete("hi", "system", 100):
        tokens.append(token)

    assert tokens == ["Hello", " world"]


def test_gateway_stream_endpoint_missing_token():
    """Streaming endpoint must reject requests with no auth token."""
    with (
        patch("finsight.gateway.api.init_pool", new_callable=AsyncMock),
        patch("finsight.gateway.api.init_qdrant", new_callable=AsyncMock),
        patch("finsight.gateway.api.ensure_collections_exist", new_callable=AsyncMock),
        patch("finsight.gateway.api.init_client", new_callable=AsyncMock),
        patch("finsight.gateway.api.init_encoder"),
        patch("finsight.gateway.api.init_reranker"),
        patch("finsight.gateway.api.close_client", new_callable=AsyncMock),
        patch("finsight.gateway.api.close_qdrant", new_callable=AsyncMock),
        patch("finsight.gateway.api.close_pool", new_callable=AsyncMock),
        patch("finsight.gateway.api.close_encoder"),
        patch("finsight.gateway.api.close_reranker"),
        patch("finsight.gateway.api.AsyncGraphDatabase.driver", return_value=AsyncMock()),
    ):
        from fastapi.testclient import TestClient
        from finsight.gateway.api import app
        with TestClient(app) as c:
            resp = c.post("/query/stream", json={"query": "What is Apple revenue?"})
            assert resp.status_code in (401, 403)


def test_gateway_stream_endpoint_empty_query():
    """Streaming endpoint must reject empty queries."""
    import time
    import jwt

    token = jwt.encode(
        {
            "sub": "team_ops",
            "team_id": "ops",
            "scopes": ["read:public_filings", "model:small"],
            "iat": int(time.time()),
            "exp": int(time.time()) + 3600,
        },
        "changeme-use-vault-in-prod",
        algorithm="HS256",
    )

    with (
        patch("finsight.gateway.api.init_pool", new_callable=AsyncMock),
        patch("finsight.gateway.api.init_qdrant", new_callable=AsyncMock),
        patch("finsight.gateway.api.ensure_collections_exist", new_callable=AsyncMock),
        patch("finsight.gateway.api.init_client", new_callable=AsyncMock),
        patch("finsight.gateway.api.init_encoder"),
        patch("finsight.gateway.api.init_reranker"),
        patch("finsight.gateway.api.close_client", new_callable=AsyncMock),
        patch("finsight.gateway.api.close_qdrant", new_callable=AsyncMock),
        patch("finsight.gateway.api.close_pool", new_callable=AsyncMock),
        patch("finsight.gateway.api.close_encoder"),
        patch("finsight.gateway.api.close_reranker"),
        patch("finsight.gateway.api.AsyncGraphDatabase.driver", return_value=AsyncMock()),
    ):
        from fastapi.testclient import TestClient
        from finsight.gateway.api import app
        with TestClient(app) as c:
            resp = c.post(
                "/query/stream",
                json={"query": "   "},
                headers={"Authorization": f"Bearer {token}"},
            )
            assert resp.status_code == 400