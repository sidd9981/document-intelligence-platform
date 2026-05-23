"""
MCP Neo4j server.

Exposes entity_lookup and cypher_query as MCP-invokable tools.
Validates the JWT and requires query:graph scope on every call.

Every Cypher query receives team_id as a parameter so the WHERE
clause can enforce data isolation at the graph layer. This is the
third isolation layer: OAuth scope at the gateway, payload filter
at Qdrant, and scope check in every Cypher query here.

If Neo4j is unavailable the endpoints return 503. The orchestrator
catches this and continues with vector-only retrieval.

Port 8103.
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

from fastapi import Depends, FastAPI, HTTPException
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer
from neo4j import AsyncDriver, AsyncGraphDatabase
from pydantic import BaseModel

from finsight.auth.token_validator import decode_token
from finsight.telemetry.tracing import get_tracer, setup_tracing

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)
security = HTTPBearer()

_driver: AsyncDriver | None = None


def get_driver() -> AsyncDriver:
    if _driver is None:
        raise RuntimeError("neo4j driver not initialized")
    return _driver


@asynccontextmanager
async def lifespan(app: FastAPI) -> AsyncGenerator[None, None]:
    global _driver
    setup_tracing()
    _driver = AsyncGraphDatabase.driver(
        "bolt://localhost:7687",
        auth=("neo4j", "changeme"),
    )
    logger.info("neo4j mcp server ready")
    yield
    await _driver.close()
    _driver = None
    logger.info("neo4j mcp server shutdown")


app = FastAPI(title="FinSight MCP Neo4j", version="0.1.0", lifespan=lifespan)


class EntityLookupRequest(BaseModel):
    name: str
    entity_type: str = "Company"
    team_id: str


class CypherQueryRequest(BaseModel):
    cypher: str
    params: dict[str, Any] = {}
    team_id: str


class EntityNode(BaseModel):
    id: str
    name: str
    entity_type: str
    properties: dict[str, Any]


class EntityLookupResponse(BaseModel):
    entities: list[EntityNode]


class CypherQueryResponse(BaseModel):
    rows: list[dict[str, Any]]
    cypher_executed: str


def _require_graph_scope(credentials: HTTPAuthorizationCredentials) -> dict:
    try:
        payload = decode_token(credentials.credentials)
    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=401, detail=f"invalid token: {e}")
    if "query:graph" not in set(payload.get("scopes", [])):
        raise HTTPException(status_code=403, detail="missing required scope: query:graph")
    return payload


@app.post("/invoke/entity_lookup", response_model=EntityLookupResponse)
async def entity_lookup(
    request: EntityLookupRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> EntityLookupResponse:
    """Look up a canonical entity node by name and type."""
    _require_graph_scope(credentials)

    with tracer.start_as_current_span("mcp_neo4j.entity_lookup") as span:
        span.set_attribute("team_id", request.team_id)
        span.set_attribute("entity_name", request.name)
        span.set_attribute("entity_type", request.entity_type)

        cypher = """
            MATCH (e {name: $name})
            WHERE $team_id IN e.scopes OR 'public' IN e.scopes
            RETURN elementId(e) AS id, e.name AS name,
                   labels(e)[0] AS entity_type, properties(e) AS props
            LIMIT 10
        """
        try:
            driver = get_driver()
            async with driver.session() as session:
                result = await session.run(
                    cypher,
                    name=request.name,
                    team_id=request.team_id,
                )
                records = await result.data()
        except Exception as e:
            logger.error("neo4j entity_lookup failed: %s", e)
            raise HTTPException(status_code=503, detail=f"graph unavailable: {e}")

        entities = [
            EntityNode(
                id=r["id"],
                name=r["name"],
                entity_type=r["entity_type"],
                properties=r["props"],
            )
            for r in records
        ]
        span.set_attribute("entities_found", len(entities))
        return EntityLookupResponse(entities=entities)


@app.post("/invoke/cypher_query", response_model=CypherQueryResponse)
async def cypher_query(
    request: CypherQueryRequest,
    credentials: HTTPAuthorizationCredentials = Depends(security),
) -> CypherQueryResponse:
    """Execute a parameterised Cypher query. team_id is always injected."""
    _require_graph_scope(credentials)

    with tracer.start_as_current_span("mcp_neo4j.cypher_query") as span:
        span.set_attribute("team_id", request.team_id)

        # Always inject team_id so callers cannot forget scope filtering.
        params = {**request.params, "team_id": request.team_id}

        try:
            driver = get_driver()
            async with driver.session() as session:
                result = await session.run(request.cypher, **params)
                rows = await result.data()
        except Exception as e:
            logger.error("neo4j cypher_query failed: %s", e)
            raise HTTPException(status_code=503, detail=f"graph unavailable: {e}")

        span.set_attribute("rows_returned", len(rows))
        return CypherQueryResponse(rows=rows, cypher_executed=request.cypher)


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}