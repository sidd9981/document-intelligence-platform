"""
Shared base types used across all agent contracts.

These types are the vocabulary of the system. Every component that
moves data between agents uses these. Adding a field here affects
every component that uses that model, which is intentional — the
type system enforces consistency across the whole pipeline.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field


class ChunkMetadata(BaseModel):
    """Metadata attached to every chunk regardless of source document type.

    Every chunk in Qdrant carries this as its payload. It is also
    attached to Chunk objects returned by the retrieval agent so the
    synthesis agent and harness know exactly where each piece of
    context came from.

    The scopes field drives data isolation. Every Qdrant query filters
    on this field so a tenant only receives chunks their team is
    authorized to see.
    """

    doc_id: str
    ticker: str
    company_name: str
    filing_type: Literal["10-K", "10-Q", "8-K", "transcript", "news"]
    filing_date: date
    section: str
    page_number: int | None = None
    chunk_index: int
    token_count: int
    embedding_model: str
    scopes: list[str]


class Chunk(BaseModel):
    """A single retrieved chunk with its relevance score.

    Produced by the retrieval agent after RRF fusion and re-ranking.
    Consumed by the input harness for context building and by the
    output harness for citation extraction.
    """

    chunk_id: str
    doc_id: str
    content: str
    score: float 
    token_count: int = Field(gt=0)
    metadata: ChunkMetadata


class AgentError(BaseModel):
    """A structured error from any agent or MCP tool call.

    Errors are accumulated in AgentState rather than raised as
    exceptions. This allows the orchestrator to continue with
    degraded results rather than failing the entire query when
    one agent encounters a problem.
    """

    agent: str
    error_type: Literal[
        "timeout",
        "empty_result",
        "parse_failure",
        "budget_exceeded",
        "service_unavailable",
    ]
    message: str
    fallback_used: bool = False