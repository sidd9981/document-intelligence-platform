"""
Unit tests for the MCP registry.

No running services needed. Uses FastAPI's TestClient.
"""

import pytest
from fastapi.testclient import TestClient

from finsight.mcp_servers.registry import app, TOOL_REGISTRY

client = TestClient(app)


def test_health():
    resp = client.get("/health")
    assert resp.status_code == 200
    assert resp.json()["status"] == "ok"
    assert resp.json()["tools_registered"] == len(TOOL_REGISTRY)


def test_list_tools_unknown_team_returns_404():
    resp = client.get("/registry/tools?team_id=ghost")
    assert resp.status_code == 404


def test_ops_cannot_see_graph_tools():
    """Ops has no query:graph scope so Neo4j tools must be absent."""
    resp = client.get("/registry/tools?team_id=ops")
    assert resp.status_code == 200
    names = [t["name"] for t in resp.json()]
    assert "entity_lookup" not in names
    assert "cypher_query" not in names


def test_analysis_can_see_graph_tools():
    resp = client.get("/registry/tools?team_id=analysis")
    assert resp.status_code == 200
    names = [t["name"] for t in resp.json()]
    assert "entity_lookup" in names
    assert "cypher_query" in names


def test_all_teams_can_see_vector_search():
    for team in ("analysis", "risk", "ops"):
        resp = client.get(f"/registry/tools?team_id={team}")
        assert resp.status_code == 200
        names = [t["name"] for t in resp.json()]
        assert "vector_search" in names, f"{team} should see vector_search"


def test_get_tool_by_name_returns_correct_definition():
    resp = client.get("/registry/tool/vector_search")
    assert resp.status_code == 200
    data = resp.json()
    assert data["name"] == "vector_search"
    assert "invoke_url" in data
    assert "input_schema" in data


def test_get_tool_unknown_name_returns_404():
    resp = client.get("/registry/tool/does_not_exist")
    assert resp.status_code == 404


def test_list_tools_returns_invoke_urls():
    """Every tool must have an invoke_url so agents know where to call."""
    resp = client.get("/registry/tools?team_id=analysis")
    for tool in resp.json():
        assert tool["invoke_url"].startswith("http"), f"{tool['name']} missing invoke_url"