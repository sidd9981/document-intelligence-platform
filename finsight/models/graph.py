"""
Typed contract for the graph agent.

The graph agent returns entity nodes and relationships from Neo4j.
These supplement the retrieved chunks by providing structured
relationship context that vector search cannot find.

When fallback is True, the orchestrator knows graph retrieval
was unavailable and can note this in the response metadata.
The synthesis agent can still produce an answer using only the
chunks from the retrieval agent.
"""

from pydantic import BaseModel, Field

from finsight.models.base import AgentError


class EntityNode(BaseModel):
    """A company, person, or concept node from the knowledge graph.

    cik is the SEC Central Index Key — the canonical identifier for
    any registered company. Using cik as the primary identifier
    rather than name prevents ambiguity between entities with
    similar names.
    """

    id: str
    cik: str | None = None
    name: str
    entity_type: str
    properties: dict = Field(default_factory=dict)


class Relationship(BaseModel):
    """A directed relationship between two entity nodes.

    relationship_type uses uppercase snake case to match Neo4j
    convention, for example SUPPLIES_TO or COMPETITOR_OF.
    """

    source_id: str
    target_id: str
    relationship_type: str
    properties: dict = Field(default_factory=dict)


class GraphResult(BaseModel):
    """The complete output of the graph agent.

    related_doc_ids contains document IDs from the knowledge graph
    that are relevant to the query entities. These can be used to
    fetch additional chunks from Qdrant that the vector search
    might not have ranked highly.

    cypher_executed records the actual query run against Neo4j.
    Logged in every trace for debugging retrieval quality issues.
    """

    entities: list[EntityNode] = Field(default_factory=list)
    relationships: list[Relationship] = Field(default_factory=list)
    related_doc_ids: list[str] = Field(default_factory=list)
    cypher_executed: str = ""
    latency_ms: float = Field(default=0.0, ge=0.0)
    fallback: bool = False
    errors: list[AgentError] = Field(default_factory=list)