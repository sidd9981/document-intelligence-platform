"""
Typed contract for the synthesis agent and output harness.

SynthesisResult is the final structured output before the gateway
formats and streams the response to the client. Every field here
is either shown to the analyst or used by the observability layer.
"""

from pydantic import BaseModel, Field

from finsight.models.base import AgentError


class Citation(BaseModel):
    """A mapping from a claim in the answer to its source chunk.

    The output harness produces one Citation per verifiable claim.
    Claims that cannot be mapped to a source chunk are recorded in
    SynthesisResult.unsupported_claims rather than silently omitted.

    confidence reflects how strongly the claim text matches the
    source chunk content. A low confidence citation is still a
    citation but should be reviewed.
    """

    claim: str
    source_chunk_id: str
    source_doc_id: str
    confidence: float = Field(ge=0.0, le=1.0)


class SynthesisResult(BaseModel):
    """The complete output of the synthesis agent and output harness.

    faithfulness_score is produced by the LLM-as-judge pipeline in
    the output harness. A score below 0.85 triggers a warning in the
    response and is logged to the faithfulness_failures table.

    unsupported_claims contains specific claim text that the judge
    determined was not grounded in the retrieved context. Logged as
    structured data in Postgres, not just as a score.

    hallucination_flags contains numbers or statistics present in
    the answer that do not appear verbatim in any retrieved chunk.
    Financial figures that cannot be sourced are the highest-risk
    hallucination type in this domain.
    """

    answer: str
    citations: list[Citation] = Field(default_factory=list)
    faithfulness_score: float = Field(default=0.0, ge=0.0, le=1.0)
    unsupported_claims: list[str] = Field(default_factory=list)
    hallucination_flags: list[str] = Field(default_factory=list)
    tokens_used: int = Field(ge=0)
    model_used: str
    prompt_version: str
    latency_ms: float = Field(ge=0.0)
    errors: list[AgentError] = Field(default_factory=list)


class QueryResponse(BaseModel):
    """The final response returned to the client by the gateway.

    This is what an analyst receives. It contains the answer, its
    citations, and enough metadata for the analyst to verify and
    for the compliance team to audit.
    """

    trace_id: str
    answer: str
    citations: list[Citation]
    faithfulness_score: float
    model_used: str
    latency_ms: float
    cache_hit: bool
    warning: str | None = None