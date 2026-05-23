"""
Unit tests for the LLM MCP server.

No Ollama needed. Mocks complete() so we test auth, scope
enforcement, model tier selection, and response shaping only.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, patch

import pytest
from fastapi.testclient import TestClient

LARGE_SCOPES = ["read:all_filings", "query:graph", "model:large", "model:medium"]
SMALL_SCOPES = ["read:public_filings", "model:small"]
NO_MODEL_SCOPES = ["read:public_filings"]


def _auth_header() -> dict:
    return {"Authorization": "Bearer valid.jwt.token"}


@pytest.fixture
def client():
    with (
        patch("finsight.mcp_servers.llm_server.init_client", new_callable=AsyncMock),
        patch("finsight.mcp_servers.llm_server.close_client", new_callable=AsyncMock),
        patch("finsight.mcp_servers.llm_server.setup_tracing"),
    ):
        from finsight.mcp_servers.llm_server import app
        with TestClient(app) as c:
            yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_generate_missing_token(client):
    resp = client.post(
        "/invoke/generate",
        json={"prompt": "What is Apple revenue?", "system": "You are an analyst."},
    )
    assert resp.status_code in (401, 403)


def test_generate_no_model_scope(client):
    with patch(
        "finsight.mcp_servers.llm_server.decode_token",
        return_value={"team_id": "ops", "scopes": NO_MODEL_SCOPES},
    ):
        resp = client.post(
            "/invoke/generate",
            json={"prompt": "What is Apple revenue?", "system": "You are an analyst."},
            headers=_auth_header(),
        )
        assert resp.status_code == 403


def test_generate_returns_answer(client):
    with (
        patch(
            "finsight.mcp_servers.llm_server.decode_token",
            return_value={"team_id": "ops", "scopes": SMALL_SCOPES},
        ),
        patch(
            "finsight.mcp_servers.llm_server.complete",
            new_callable=AsyncMock,
            return_value=("Apple revenue was $383B.", 120, 30),
        ),
    ):
        resp = client.post(
            "/invoke/generate",
            json={"prompt": "What is Apple revenue?", "system": "You are an analyst."},
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["answer"] == "Apple revenue was $383B."
        assert data["prompt_tokens"] == 120
        assert data["completion_tokens"] == 30


def test_generate_picks_highest_tier(client):
    """A token with model:large should use the large model, not small."""
    captured = {}

    async def fake_complete(prompt, system, max_tokens):
        return ("answer", 10, 5)

    with (
        patch(
            "finsight.mcp_servers.llm_server.decode_token",
            return_value={"team_id": "analysis", "scopes": LARGE_SCOPES},
        ),
        patch("finsight.mcp_servers.llm_server.complete", side_effect=fake_complete),
    ):
        resp = client.post(
            "/invoke/generate",
            json={"prompt": "Summarise filings.", "system": "You are an analyst."},
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        assert resp.json()["model_used"] == "llama3.1:8b"


def test_generate_small_scope_uses_small_model(client):
    with (
        patch(
            "finsight.mcp_servers.llm_server.decode_token",
            return_value={"team_id": "ops", "scopes": SMALL_SCOPES},
        ),
        patch(
            "finsight.mcp_servers.llm_server.complete",
            new_callable=AsyncMock,
            return_value=("answer", 10, 5),
        ),
    ):
        resp = client.post(
            "/invoke/generate",
            json={"prompt": "Quick question.", "system": "You are an analyst."},
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        assert resp.json()["model_used"] == "llama3.2:3b"