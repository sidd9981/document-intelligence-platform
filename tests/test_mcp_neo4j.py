"""
Unit tests for the Neo4j MCP server.

No running Neo4j needed. Mocks the driver so we test auth,
scope enforcement, and response shaping only.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from fastapi.testclient import TestClient

GRAPH_SCOPES = ["read:all_filings", "query:graph", "model:large"]
NO_GRAPH_SCOPES = ["read:public_filings", "model:small"]

FAKE_ENTITY_ROW = {
    "id": "element:1",
    "name": "Apple Inc.",
    "entity_type": "Company",
    "props": {"ticker": "AAPL", "scopes": ["public"]},
}


def _auth_header() -> dict:
    return {"Authorization": "Bearer valid.jwt.token"}


def _mock_session(rows: list[dict]):
    """Build a mock Neo4j session that returns the given rows."""
    mock_result = AsyncMock()
    mock_result.data = AsyncMock(return_value=rows)

    mock_session = AsyncMock()
    mock_session.run = AsyncMock(return_value=mock_result)
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    mock_driver = MagicMock()
    mock_driver.session = MagicMock(return_value=mock_session)
    mock_driver.close = AsyncMock()
    return mock_driver


@pytest.fixture
def client():
    mock_driver = _mock_session([])
    with (
        patch("finsight.mcp_servers.neo4j_server.AsyncGraphDatabase") as mock_gdb,
        patch("finsight.mcp_servers.neo4j_server.setup_tracing"),
    ):
        mock_gdb.driver.return_value = mock_driver
        from finsight.mcp_servers.neo4j_server import app
        with TestClient(app) as c:
            yield c, mock_driver


def test_health(client):
    c, _ = client
    resp = c.get("/health")
    assert resp.status_code == 200


def test_entity_lookup_missing_token(client):
    c, _ = client
    resp = c.post("/invoke/entity_lookup", json={"name": "Apple", "team_id": "analysis"})
    assert resp.status_code in (401, 403)


def test_entity_lookup_missing_graph_scope(client):
    c, _ = client
    with patch(
        "finsight.mcp_servers.neo4j_server.decode_token",
        return_value={"team_id": "ops", "scopes": NO_GRAPH_SCOPES},
    ):
        resp = c.post(
            "/invoke/entity_lookup",
            json={"name": "Apple", "team_id": "ops"},
            headers=_auth_header(),
        )
        assert resp.status_code == 403


def test_entity_lookup_returns_entities(client):
    c, mock_driver = client
    mock_neo4j = _mock_session([FAKE_ENTITY_ROW])
    with (
        patch(
            "finsight.mcp_servers.neo4j_server.decode_token",
            return_value={"team_id": "analysis", "scopes": GRAPH_SCOPES},
        ),
        patch("finsight.mcp_servers.neo4j_server.get_driver", return_value=mock_neo4j),
    ):
        resp = c.post(
            "/invoke/entity_lookup",
            json={"name": "Apple", "team_id": "analysis"},
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        data = resp.json()
        assert len(data["entities"]) == 1
        assert data["entities"][0]["name"] == "Apple Inc."


def test_cypher_query_injects_team_id(client):
    """team_id must always be injected into params even if caller omits it."""
    c, _ = client
    captured_params = {}

    async def fake_run(cypher, **params):
        captured_params.update(params)
        mock_result = AsyncMock()
        mock_result.data = AsyncMock(return_value=[])
        return mock_result

    mock_session = AsyncMock()
    mock_session.run = fake_run
    mock_session.__aenter__ = AsyncMock(return_value=mock_session)
    mock_session.__aexit__ = AsyncMock(return_value=None)

    mock_drv = MagicMock()
    mock_drv.session = MagicMock(return_value=mock_session)

    with (
        patch(
            "finsight.mcp_servers.neo4j_server.decode_token",
            return_value={"team_id": "analysis", "scopes": GRAPH_SCOPES},
        ),
        patch("finsight.mcp_servers.neo4j_server.get_driver", return_value=mock_drv),
    ):
        resp = c.post(
            "/invoke/cypher_query",
            json={"cypher": "MATCH (c:Company) RETURN c", "team_id": "analysis"},
            headers=_auth_header(),
        )
        assert resp.status_code == 200
        assert captured_params.get("team_id") == "analysis"


def test_cypher_query_neo4j_down_returns_503(client):
    c, _ = client
    mock_drv = MagicMock()
    mock_drv.session.side_effect = Exception("connection refused")

    with (
        patch(
            "finsight.mcp_servers.neo4j_server.decode_token",
            return_value={"team_id": "analysis", "scopes": GRAPH_SCOPES},
        ),
        patch("finsight.mcp_servers.neo4j_server.get_driver", return_value=mock_drv),
    ):
        resp = c.post(
            "/invoke/cypher_query",
            json={"cypher": "MATCH (c:Company) RETURN c", "team_id": "analysis"},
            headers=_auth_header(),
        )
        assert resp.status_code == 503