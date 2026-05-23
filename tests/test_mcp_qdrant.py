"""
Unit tests for the Qdrant MCP server.

No running services. Mocks out embed, search_dense, search_sparse,
and encode_sparse so we're testing the server's auth and routing
logic only.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

from finsight.models.base import Chunk, ChunkMetadata

FAKE_CHUNK = Chunk(
    chunk_id="abc123",
    doc_id="doc001",
    content="Apple depends on TSMC for chip manufacturing.",
    score=0.91,
    token_count=10,
    metadata=ChunkMetadata(
        doc_id="doc001",
        ticker="AAPL",
        company_name="Apple Inc.",
        filing_type="10-K",
        filing_date=date(2023, 10, 27),
        section="Risk Factors",
        chunk_index=0,
        token_count=10,
        embedding_model="nomic-embed-text",
        scopes=["public"],
    ),
)

VALID_TOKEN = "valid.jwt.token"
ANALYSIS_SCOPES = ["read:all_filings", "query:graph", "model:large"]
OPS_SCOPES = ["read:public_filings", "model:small"]
NO_FILING_SCOPES = ["model:small"]


def _make_app():
    """Import the app inside the mock context so lifespan doesn't fire."""
    from finsight.mcp_servers.qdrant_server import app
    return app


@pytest.fixture
def client():
    with (
        patch("finsight.mcp_servers.qdrant_server.init_qdrant", new_callable=AsyncMock),
        patch("finsight.mcp_servers.qdrant_server.ensure_collections_exist", new_callable=AsyncMock),
        patch("finsight.mcp_servers.qdrant_server.init_client", new_callable=AsyncMock),
        patch("finsight.mcp_servers.qdrant_server.init_encoder"),
        patch("finsight.mcp_servers.qdrant_server.close_encoder"),
        patch("finsight.mcp_servers.qdrant_server.close_client", new_callable=AsyncMock),
        patch("finsight.mcp_servers.qdrant_server.close_qdrant", new_callable=AsyncMock),
    ):
        from finsight.mcp_servers.qdrant_server import app
        with TestClient(app) as c:
            yield c


def _auth_header(token: str = VALID_TOKEN) -> dict:
    return {"Authorization": f"Bearer {token}"}


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"


def test_vector_search_missing_token(client):
    resp = client.post("/invoke/vector_search", json={"query": "TSMC risk", "team_id": "ops"})
    assert resp.status_code in (401, 403)

def test_vector_search_invalid_token(client):
    with patch(
        "finsight.mcp_servers.qdrant_server.decode_token",
        side_effect=Exception("bad token"),
    ):
        resp = client.post(
            "/invoke/vector_search",
            json={"query": "TSMC risk", "team_id": "ops"},
            headers=_auth_header("bad"),
        )
        assert resp.status_code in (401, 403, 500)


def test_vector_search_missing_filing_scope(client):
    with patch(
        "finsight.mcp_servers.qdrant_server.decode_token",
        return_value={"team_id": "ops", "scopes": NO_FILING_SCOPES},
    ):
        resp = client.post(
            "/invoke/vector_search",
            json={"query": "TSMC risk", "team_id": "ops"},
            headers=_auth_header(),
        )
        assert resp.status_code == 403


def test_vector_search_returns_chunks(client):
    with (
        patch(
            "finsight.mcp_servers.qdrant_server.decode_token",
            return_value={"team_id": "ops", "scopes": OPS_SCOPES},
        ),
        patch(
            "finsight.mcp_servers.qdrant_server.embed",
            new_callable=AsyncMock,
            return_value=[0.1] * 768,
        ),
        patch(
            "finsight.mcp_servers.qdrant_server.search_dense",
            new_callable=AsyncMock,
            return_value=[FAKE_CHUNK],
        ),
    ):
        resp = client.post(
            "/invoke/vector_search",
            json={"query": "TSMC risk", "team_id": "ops"},
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["chunks"]) == 1
        assert data["chunks"][0]["chunk_id"] == "abc123"


def test_sparse_search_returns_chunks(client):
    with (
        patch(
            "finsight.mcp_servers.qdrant_server.decode_token",
            return_value={"team_id": "ops", "scopes": OPS_SCOPES},
        ),
        patch(
            "finsight.mcp_servers.qdrant_server.encode_sparse",
            return_value={42: 0.9, 77: 0.4},
        ),
        patch(
            "finsight.mcp_servers.qdrant_server.search_sparse",
            new_callable=AsyncMock,
            return_value=[FAKE_CHUNK],
        ),
    ):
        resp = client.post(
            "/invoke/sparse_search",
            json={"query": "TSMC risk", "team_id": "ops"},
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        assert len(resp.json()["chunks"]) == 1


def test_all_filing_scopes_accepted(client):
    """Both read:public_filings and read:all_filings should be accepted."""
    for scope_list in (OPS_SCOPES, ANALYSIS_SCOPES):
        with (
            patch(
                "finsight.mcp_servers.qdrant_server.decode_token",
                return_value={"team_id": "ops", "scopes": scope_list},
            ),
            patch(
                "finsight.mcp_servers.qdrant_server.embed",
                new_callable=AsyncMock,
                return_value=[0.1] * 768,
            ),
            patch(
                "finsight.mcp_servers.qdrant_server.search_dense",
                new_callable=AsyncMock,
                return_value=[],
            ),
        ):
            resp = client.post(
                "/invoke/vector_search",
                json={"query": "test", "team_id": "ops"},
                headers=_auth_header(),
            )
            assert resp.status_code == 200, f"scope {scope_list} should be accepted"