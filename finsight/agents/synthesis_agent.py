"""
Synthesis agent.

Formats the prepared context as numbered sources, calls the model for the
tenant's tier, and returns a structured SynthesisResult. Chunk selection
and ordering happen in the input harness, and validation happens in the
output harness.

The model cites sources as [1], [2]. Short numeric markers are far more
reliable for small models than copying long chunk ids, and each number
maps back to exactly one chunk. Square brackets are reserved for
citations, so the prompt forbids bracketed list numbering, which would
otherwise be misread as citations.

The prompt asks for a direct answer in the model's own words. Without
that instruction, smaller models tend to paste source passages verbatim,
which reads as noncommittal and inflates faithfulness scores.
"""

from __future__ import annotations

import logging
import re
import time

from finsight.models.base import Chunk
from finsight.models.graph import GraphResult
from finsight.models.synthesis import Citation, SynthesisResult
from finsight.models.tenant import TenantConfig
from finsight.services.llm import complete, count_tokens, resolve_model
from finsight.services.reranker import relevance_probability
from finsight.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

PROMPT_VERSION = "synthesis_v3"
CLAIM_WINDOW_CHARS = 200

SYSTEM_PROMPT = """You are a financial document analyst. Answer the question using only the numbered sources provided.

Write the answer in your own words. Start with one sentence that directly answers the question, then give the supporting details. Summarize what the sources say. Do not copy passages verbatim and do not quote long excerpts.

End each sentence that makes a factual claim with the number of the source that supports it, in square brackets, for example [3] or [2][5]. Use square brackets only for citations. For lists, start each item with a hyphen, never with a number in brackets.

Rules:
- Only use information from the provided sources
- If the sources do not contain enough information to answer, say so explicitly
- Never invent figures, dates, or names not present in the sources
- Be concise and precise"""

_CITATION_PATTERN = re.compile(r"\[(\d{1,3})\]")


class SynthesisAgent:
    """Calls the tenant's model with prepared context and returns a structured result.

    Stateless, safe to reuse across requests.
    """

    async def synthesize(
        self,
        query: str,
        chunks: list[Chunk],
        graph_result: GraphResult | None,
        tenant_config: TenantConfig,
        trace_id: str,
    ) -> SynthesisResult:
        """Build the prompt and call the model.

        Args:
            query: The question, or the history-wrapped prompt for multi-turn requests.
            chunks: Chunks from the input harness, already trimmed and ordered.
                    Source numbers in the prompt follow this order.
            graph_result: Entity and relationship context from the graph agent.
                          May be None if graph retrieval was skipped or failed.
            tenant_config: Supplies the model tier and max_output_tokens.
            trace_id: Propagated to spans.

        Returns:
            SynthesisResult with the answer, citations mapped to chunks, and
            token usage. Never raises, returns an empty answer on failure.
        """
        start = time.perf_counter()
        model = resolve_model(tenant_config.default_model)

        with tracer.start_as_current_span("synthesis_agent.synthesize") as span:
            span.set_attribute("team_id", tenant_config.team_id)
            span.set_attribute("trace_id", trace_id)
            span.set_attribute("chunks.count", len(chunks))
            span.set_attribute("model_tier", tenant_config.default_model)
            span.set_attribute("model", model)

            try:
                context = _build_context(chunks, graph_result)
                prompt = _build_prompt(query, context)
                span.set_attribute("prompt_tokens_estimate", count_tokens(SYSTEM_PROMPT + prompt))

                answer, prompt_tokens, completion_tokens = await complete(
                    prompt=prompt,
                    system=SYSTEM_PROMPT,
                    max_tokens=tenant_config.max_output_tokens,
                    model=model,
                )

                citations = _extract_citations(answer, chunks)
                latency_ms = (time.perf_counter() - start) * 1000

                span.set_attribute("completion_tokens", completion_tokens)
                span.set_attribute("citations.count", len(citations))
                span.set_attribute("latency_ms", round(latency_ms, 2))

                return SynthesisResult(
                    answer=answer,
                    citations=citations,
                    faithfulness_score=0.0,
                    tokens_used=prompt_tokens + completion_tokens,
                    model_used=model,
                    prompt_version=PROMPT_VERSION,
                    latency_ms=latency_ms,
                )

            except Exception as e:
                logger.error("synthesis agent error: %s", e)
                latency_ms = (time.perf_counter() - start) * 1000
                return SynthesisResult(
                    answer="",
                    tokens_used=0,
                    model_used=model,
                    prompt_version=PROMPT_VERSION,
                    latency_ms=latency_ms,
                )


def _build_context(chunks: list[Chunk], graph_result: GraphResult | None) -> str:
    """Format chunks as numbered sources, followed by graph context if any."""
    parts = [
        f"[{number}] {chunk.metadata.company_name} {chunk.metadata.filing_type} "
        f"{chunk.metadata.filing_date}, {chunk.metadata.section}\n{chunk.content}"
        for number, chunk in enumerate(chunks, start=1)
    ]

    if graph_result:
        entity_lines = [f"- {e.name} (CIK: {e.cik})" for e in graph_result.entities if e.name]
        rel_lines = [
            f"- {r.source_id} {r.relationship_type} {r.target_id}"
            for r in graph_result.relationships
        ]
        if entity_lines or rel_lines:
            graph_section = "Graph context:\n"
            if entity_lines:
                graph_section += "Entities:\n" + "\n".join(entity_lines) + "\n"
            if rel_lines:
                graph_section += "Relationships:\n" + "\n".join(rel_lines)
            parts.append(graph_section)

    return "\n\n".join(parts)


def _build_prompt(query: str, context: str) -> str:
    return f"Sources:\n{context}\n\nQuestion: {query}"


def _extract_citations(answer: str, chunks: list[Chunk]) -> list[Citation]:
    """Map [n] markers in the answer back to the nth chunk.

    Numbers outside the source range are ignored, and each source is
    cited at most once. The claim is the text just before the marker.
    """
    citations: list[Citation] = []
    seen: set[int] = set()

    for match in _CITATION_PATTERN.finditer(answer):
        index = int(match.group(1)) - 1
        if not 0 <= index < len(chunks) or index in seen:
            continue
        seen.add(index)
        chunk = chunks[index]
        claim = answer[max(0, match.start() - CLAIM_WINDOW_CHARS):match.start()].strip()
        citations.append(Citation(
            claim=claim,
            source_chunk_id=chunk.chunk_id,
            source_doc_id=chunk.doc_id,
            confidence=relevance_probability(chunk.score),
        ))

    return citations