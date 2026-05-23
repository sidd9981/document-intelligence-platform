"""
Typed contract for the retrieval agent.

The retrieval agent returns exactly one RetrievalResult. The
orchestrator reads this to decide whether to proceed, degrade,
or reject the query. The synthesis agent reads the chunks field
to build the LLM context.
"""

from typing import Literal

from pydantic import BaseModel, Field

from finsight.models.base import AgentError, Chunk


class RetrievalResult(BaseModel):
    """The complete output of the retrieval agent.

    chunks contains the final ranked list after RRF fusion and
    cross-encoder re-ranking. The orchestrator checks this list
    before invoking the synthesis agent. An empty list means the
    system has no grounded context and must not call the LLM.

    retrieval_method records which path produced the result. Used
    in observability to track which method is contributing most
    to successful retrievals over time.
    """

    chunks: list[Chunk]
    cache_hit: bool = False
    retrieval_method: Literal["dense", "sparse", "hybrid", "cached"]
    total_tokens: int = Field(ge=0)
    latency_ms: float = Field(ge=0.0)
    errors: list[AgentError] = Field(default_factory=list)