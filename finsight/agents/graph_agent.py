"""
Graph agent.

Queries Neo4j for entity relationships relevant to the current query.
Runs in parallel with the retrieval agent. Returns related entities,
relationships, and document IDs that can supplement vector search results.

Never raises. Returns GraphResult with fallback=True if Neo4j is
unavailable so the orchestrator can continue with vector-only retrieval.
"""

from __future__ import annotations

import logging
import time

from neo4j import AsyncDriver

from finsight.models.base import AgentError
from finsight.models.graph import EntityNode, GraphResult, Relationship
from finsight.models.tenant import TenantConfig
from finsight.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)


class GraphAgent:
    """Queries Neo4j for entity relationships relevant to a query.

    Pass the Neo4j driver in so this is testable without a live instance.
    """

    def __init__(self, driver: AsyncDriver) -> None:
        self._driver = driver

    async def query(
        self,
        entities: list[str],
        tenant_config: TenantConfig,
        trace_id: str,
    ) -> GraphResult:
        """Find entities and relationships relevant to the query entities.

        Takes the entity list extracted by the orchestrator (ticker symbols,
        company names) and returns graph context: the canonical nodes,
        their relationships, and IDs of related filings.

        Args:
            entities: List of entity strings from the query, e.g. ['AAPL', 'TSMC'].
            tenant_config: Used for scope filtering on every Cypher query.
            trace_id: Propagated to spans.

        Returns:
            GraphResult with entities, relationships, and related doc IDs.
            Returns fallback=True with empty lists if Neo4j is unavailable.
        """
        start = time.perf_counter()

        with tracer.start_as_current_span("graph_agent.query") as span:
            span.set_attribute("team_id", tenant_config.team_id)
            span.set_attribute("trace_id", trace_id)
            span.set_attribute("entities.count", len(entities))

            if not entities:
                return GraphResult(latency_ms=0.0)

            try:
                async with self._driver.session() as session:
                    nodes, relationships = await session.execute_read(
                        _fetch_entity_subgraph,
                        entities=entities,
                        team_id=tenant_config.team_id,
                    )

                    related_doc_ids = await session.execute_read(
                        _fetch_related_doc_ids,
                        entities=entities,
                        team_id=tenant_config.team_id,
                    )

                cypher_executed = (
                    "MATCH (c:Company) WHERE c.ticker IN $entities "
                    "OR c.name IN $entities "
                    "MATCH (c)-[r]-(n) "
                    f"WHERE $team_id IN c.scopes OR 'public' IN c.scopes "
                    "RETURN c, r, n"
                )

                latency_ms = (time.perf_counter() - start) * 1000
                span.set_attribute("entities.found", len(nodes))
                span.set_attribute("relationships.found", len(relationships))
                span.set_attribute("doc_ids.found", len(related_doc_ids))

                return GraphResult(
                    entities=nodes,
                    relationships=relationships,
                    related_doc_ids=related_doc_ids,
                    cypher_executed=cypher_executed,
                    latency_ms=latency_ms,
                )

            except Exception as e:
                logger.error("graph agent error: %s", e)
                latency_ms = (time.perf_counter() - start) * 1000
                return GraphResult(
                    fallback=True,
                    latency_ms=latency_ms,
                    errors=[AgentError(
                        agent="graph_agent",
                        error_type="service_unavailable",
                        message=str(e),
                        fallback_used=True,
                    )],
                )


async def _fetch_entity_subgraph(
    tx,
    entities: list[str],
    team_id: str,
) -> tuple[list[EntityNode], list[Relationship]]:
    """Fetch company nodes and their direct relationships.

    Matches companies by ticker or name, filtered by team scope.
    Returns the nodes and all relationships one hop away.
    """
    result = await tx.run(
        """
        MATCH (c:Company)
        WHERE (c.ticker IN $entities OR c.name IN $entities)
          AND ($team_id IN c.scopes OR 'public' IN c.scopes)
        OPTIONAL MATCH (c)-[r]-(n:Company)
        WHERE $team_id IN n.scopes OR 'public' IN n.scopes
        RETURN c, r, n
        """,
        entities=entities,
        team_id=team_id,
    )

    nodes: dict[str, EntityNode] = {}
    relationships: list[Relationship] = []

    async for record in result:
        company = record["c"]
        if company and company.get("cik") not in nodes:
            nodes[company["cik"]] = EntityNode(
                id=company["cik"],
                cik=company.get("cik"),
                name=company.get("name", ""),
                entity_type="Company",
                properties=dict(company),
            )

        rel = record["r"]
        neighbor = record["n"]
        if rel and neighbor and neighbor.get("cik"):
            if neighbor["cik"] not in nodes:
                nodes[neighbor["cik"]] = EntityNode(
                    id=neighbor["cik"],
                    cik=neighbor.get("cik"),
                    name=neighbor.get("name", ""),
                    entity_type="Company",
                    properties=dict(neighbor),
                )
            relationships.append(Relationship(
                source_id=company["cik"],
                target_id=neighbor["cik"],
                relationship_type=rel.type,
                properties=dict(rel),
            ))

    return list(nodes.values()), relationships


async def _fetch_related_doc_ids(
    tx,
    entities: list[str],
    team_id: str,
) -> list[str]:
    """Fetch filing doc IDs connected to the query entities.

    These supplement vector search — filings directly linked to
    entities in the graph may contain relevant context that didn't
    rank highly in semantic search.
    """
    result = await tx.run(
        """
        MATCH (c:Company)-[:MENTIONED_IN]->(f:Filing)
        WHERE (c.ticker IN $entities OR c.name IN $entities)
          AND ($team_id IN f.scopes OR 'public' IN f.scopes)
        RETURN f.doc_id AS doc_id
        LIMIT 20
        """,
        entities=entities,
        team_id=team_id,
    )

    doc_ids = []
    async for record in result:
        if record["doc_id"]:
            doc_ids.append(record["doc_id"])

    return doc_ids