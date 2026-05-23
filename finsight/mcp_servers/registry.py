"""
MCP tool registry.

Single source of truth for which tools exist and where they live.
Agents call GET /registry/tools?team_id=<id> at startup to get the
filtered manifest for their team. The registry uses the same TEAM_SCOPES
map as the OAuth server — a team without query:graph never sees Neo4j
tools, so the graph agent skips gracefully rather than getting a 403
mid-request.

Adding a new data source means adding entries to TOOL_REGISTRY and
spinning up a new server. No agent code changes.

Port 8101. No auth on the registry itself — individual MCP servers
validate the JWT on every invoke call.
"""

from __future__ import annotations

import logging
from typing import Any

from fastapi import FastAPI, HTTPException, Query
from pydantic import BaseModel

from finsight.auth.scope_definitions import TEAM_SCOPES

logger = logging.getLogger(__name__)

app = FastAPI(title="FinSight MCP Registry", version="0.1.0")


class ToolDefinition(BaseModel):
    name: str
    description: str
    invoke_url: str
    allowed_scopes: list[str]
    input_schema: dict[str, Any]


TOOL_REGISTRY: list[ToolDefinition] = [
    ToolDefinition(
        name="vector_search",
        description="Semantic search over financial document chunks using dense embeddings.",
        invoke_url="http://mcp-qdrant:8102/invoke/vector_search",
        allowed_scopes=["read:public_filings", "read:all_filings"],
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "team_id": {"type": "string"},
                "k": {"type": "integer", "default": 20},
            },
            "required": ["query", "team_id"],
        },
    ),
    ToolDefinition(
        name="sparse_search",
        description="Keyword-aware search using SPLADE sparse vectors.",
        invoke_url="http://mcp-qdrant:8102/invoke/sparse_search",
        allowed_scopes=["read:public_filings", "read:all_filings"],
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "team_id": {"type": "string"},
                "k": {"type": "integer", "default": 20},
            },
            "required": ["query", "team_id"],
        },
    ),
    ToolDefinition(
        name="rerank",
        description="Cross-encoder reranking of RRF candidates by joint query-chunk relevance.",
        invoke_url="http://mcp-reranker:8105/invoke/rerank",
        allowed_scopes=["read:public_filings", "read:all_filings"],
        input_schema={
            "type": "object",
            "properties": {
                "query": {"type": "string"},
                "chunks": {"type": "array"},
                "top_k": {"type": "integer", "default": 10},
            },
            "required": ["query", "chunks"],
        },
    ),
    ToolDefinition(
        name="entity_lookup",
        description="Look up a canonical entity node in the knowledge graph by name.",
        invoke_url="http://mcp-neo4j:8103/invoke/entity_lookup",
        allowed_scopes=["query:graph"],
        input_schema={
            "type": "object",
            "properties": {
                "name": {"type": "string"},
                "entity_type": {"type": "string", "default": "Company"},
                "team_id": {"type": "string"},
            },
            "required": ["name", "team_id"],
        },
    ),
    ToolDefinition(
        name="cypher_query",
        description="Execute a parameterised Cypher query against Neo4j.",
        invoke_url="http://mcp-neo4j:8103/invoke/cypher_query",
        allowed_scopes=["query:graph"],
        input_schema={
            "type": "object",
            "properties": {
                "cypher": {"type": "string"},
                "params": {"type": "object"},
                "team_id": {"type": "string"},
            },
            "required": ["cypher", "team_id"],
        },
    ),
    ToolDefinition(
        name="generate",
        description="LLM completion. Produces a grounded answer from context and query.",
        invoke_url="http://mcp-llm:8104/invoke/generate",
        allowed_scopes=["model:small", "model:medium", "model:large"],
        input_schema={
            "type": "object",
            "properties": {
                "prompt": {"type": "string"},
                "system": {"type": "string"},
                "max_tokens": {"type": "integer", "default": 1000},
            },
            "required": ["prompt", "system"],
        },
    ),
]

_BY_NAME: dict[str, ToolDefinition] = {t.name: t for t in TOOL_REGISTRY}


@app.get("/registry/tools", response_model=list[ToolDefinition])
async def list_tools(team_id: str = Query(...)) -> list[ToolDefinition]:
    """Return only the tools this team's scopes permit."""
    team_scopes = TEAM_SCOPES.get(team_id)
    if team_scopes is None:
        raise HTTPException(status_code=404, detail=f"unknown team: {team_id}")

    scopes_set = set(team_scopes)
    visible = [t for t in TOOL_REGISTRY if any(s in scopes_set for s in t.allowed_scopes)]
    logger.info("registry list_tools team=%s visible=%d", team_id, len(visible))
    return visible


@app.get("/registry/tool/{tool_name}", response_model=ToolDefinition)
async def get_tool(tool_name: str) -> ToolDefinition:
    """Return a single tool definition by name."""
    tool = _BY_NAME.get(tool_name)
    if tool is None:
        raise HTTPException(status_code=404, detail=f"unknown tool: {tool_name}")
    return tool


@app.get("/health")
async def health() -> dict:
    return {"status": "ok", "tools_registered": len(TOOL_REGISTRY)}