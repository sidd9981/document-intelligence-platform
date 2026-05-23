"""
Unit tests for the reranker MCP server.

No model loading needed. Mocks rerank() so we test auth,
scope enforcement, and response shaping only.
"""

from __future__ import annotations

from datetime import date
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from finsight.models.base import Chunk, ChunkMetadata

FILING_SCOPES = ["read:public_filings", "model:small"]
ALL_SCOPES = ["read:all_filings", "query:graph", "model:large"]
NO_FILING_SCOPES = ["model:small"]

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


def _auth_header() -> dict:
    return {"Authorization": "Bearer valid.jwt.token"}


@pytest.fixture
def client():
    with (
        patch("finsight.mcp_servers.reranker_server.init_reranker"),
        patch("finsight.mcp_servers.reranker_server.close_reranker"),
        patch("finsight.mcp_servers.reranker_server.setup_tracing"),
    ):
        from finsight.mcp_servers.reranker_server import app
        with TestClient(app) as c:
            yield c


def test_health(client):
    resp = client.get("/health")
    assert resp.status_code == 200


def test_rerank_missing_token(client):
    resp = client.post(
        "/invoke/rerank",
        json={"query": "TSMC risk", "chunks": []},
    )
    assert resp.status_code in (401, 403)


def test_rerank_missing_filing_scope(client):
    with patch(
        "finsight.mcp_servers.reranker_server.decode_token",
        return_value={"team_id": "ops", "scopes": NO_FILING_SCOPES},
    ):
        resp = client.post(
            "/invoke/rerank",
            json={"query": "TSMC risk", "chunks": []},
            headers=_auth_header(),
        )
        assert resp.status_code == 403


def test_rerank_empty_chunks_returns_empty(client):
    with (
        patch(
            "finsight.mcp_servers.reranker_server.decode_token",
            return_value={"team_id": "ops", "scopes": FILING_SCOPES},
        ),
        patch("finsight.mcp_servers.reranker_server.rerank", return_value=[]),
    ):
        resp = client.post(
            "/invoke/rerank",
            json={"query": "TSMC risk", "chunks": []},
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        assert resp.json()["chunks"] == []


def test_rerank_returns_rescored_chunks(client):
    reranked_chunk = FAKE_CHUNK.model_copy(update={"score": 0.97})
    with (
        patch(
            "finsight.mcp_servers.reranker_server.decode_token",
            return_value={"team_id": "ops", "scopes": FILING_SCOPES},
        ),
        patch(
            "finsight.mcp_servers.reranker_server.rerank",
            return_value=[reranked_chunk],
        ),
    ):
        resp = client.post(
            "/invoke/rerank",
            json={
                "query": "TSMC risk",
                "chunks": [FAKE_CHUNK.model_dump(mode="json")],
                "top_k": 5,
            },
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["chunks"]) == 1
        assert data["chunks"][0]["score"] == 0.97


def test_both_filing_scopes_accepted(client):
    """read:public_filings and read:all_filings must both be accepted."""
    for scopes in (FILING_SCOPES, ALL_SCOPES):
        with (
            patch(
                "finsight.mcp_servers.reranker_server.decode_token",
                return_value={"team_id": "ops", "scopes": scopes},
            ),
            patch("finsight.mcp_servers.reranker_server.rerank", return_value=[]),
        ):
            resp = client.post(
                "/invoke/rerank",
                json={"query": "test", "chunks": []},
                headers=_auth_header(),
            )
            assert resp.status_code == 200, f"scope {scopes} should be accepted"