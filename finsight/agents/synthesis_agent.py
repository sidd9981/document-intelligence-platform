"""
Synthesis agent.

Builds a prompt from retrieved context, calls the LLM, and returns
a structured SynthesisResult. The harness wraps this agent — input
validation and output validation happen outside it.

Kept intentionally thin. All logic that affects answer quality lives
in the harness where it is versioned, tested, and observable.
"""

from __future__ import annotations

import logging
import time

from finsight.models.base import Chunk
from finsight.models.graph import GraphResult
from finsight.models.synthesis import Citation, SynthesisResult
from finsight.models.tenant import TenantConfig
from finsight.services.llm import complete, count_tokens
from finsight.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

PROMPT_VERSION = "synthesis_v1"

SYSTEM_PROMPT = """You are a financial document analyst. Answer the question using only the provided context.

For every factual claim in your answer, cite the source chunk using this format: [chunk_id: <id>]

Rules:
- Only use information from the provided context
- If the context does not contain enough information to answer, say so explicitly
- Never invent figures, dates, or names not present in the context
- Be concise and precise"""


class SynthesisAgent:
    """Calls the LLM with retrieved context and returns a structured result.

    Stateless — safe to reuse across requests.
    """

    async def synthesize(
        self,
        query: str,
        chunks: list[Chunk],
        graph_result: GraphResult | None,
        tenant_config: TenantConfig,
        trace_id: str,
    ) -> SynthesisResult:
        """Build a prompt and call the LLM.

        Args:
            query: The original user query.
            chunks: Retrieved and reranked chunks from the retrieval agent.
                    Already in position-bias order from the input harness.
            graph_result: Entity and relationship context from the graph agent.
                          May be None if graph retrieval was skipped or failed.
            tenant_config: Controls max_output_tokens and model selection.
            trace_id: Propagated to spans.

        Returns:
            SynthesisResult with the answer, raw citations from the model,
            and token usage. The output harness validates and enriches this.
            Never raises — returns a SynthesisResult with an error on failure.
        """
        start = time.perf_counter()

        with tracer.start_as_current_span("synthesis_agent.synthesize") as span:
            span.set_attribute("team_id", tenant_config.team_id)
            span.set_attribute("trace_id", trace_id)
            span.set_attribute("chunks.count", len(chunks))

            try:
                ordered_chunks = _reorder_for_position_bias(chunks)
                context = _build_context(ordered_chunks, graph_result)
                prompt = _build_prompt(query, context)

                span.set_attribute("prompt_tokens_estimate", count_tokens(SYSTEM_PROMPT + prompt))

                answer, prompt_tokens, completion_tokens = await complete(
                    prompt=prompt,
                    system=SYSTEM_PROMPT,
                    max_tokens=tenant_config.max_output_tokens,
                )

                citations = _extract_citations(answer, chunks)
                latency_ms = (time.perf_counter() - start) * 1000

                span.set_attribute("completion_tokens", completion_tokens)
                span.set_attribute("latency_ms", round(latency_ms, 2))

                return SynthesisResult(
                    answer=answer,
                    citations=citations,
                    faithfulness_score=0.0,
                    tokens_used=prompt_tokens + completion_tokens,
                    model_used=_model_name(),
                    prompt_version=PROMPT_VERSION,
                    latency_ms=latency_ms,
                )

            except Exception as e:
                logger.error("synthesis agent error: %s", e)
                latency_ms = (time.perf_counter() - start) * 1000
                return SynthesisResult(
                    answer="",
                    tokens_used=0,
                    model_used=_model_name(),
                    prompt_version=PROMPT_VERSION,
                    latency_ms=latency_ms,
                )


def _reorder_for_position_bias(chunks: list[Chunk]) -> list[Chunk]:
    """Put best evidence where the LLM actually reads it.

    LLMs over-attend to the beginning and end of their context.
    Best chunk goes first, second-best goes last, rest fill the middle.
    Chunks are already sorted by descending score from the reranker.
    """
    if len(chunks) <= 2:
        return chunks
    return [chunks[0]] + chunks[2:] + [chunks[1]]


def _build_context(chunks: list[Chunk], graph_result: GraphResult | None) -> str:
    """Format chunks and graph entities into a context string for the prompt."""
    parts = []

    for chunk in chunks:
        parts.append(
            f"[chunk_id: {chunk.chunk_id}]\n"
            f"Source: {chunk.metadata.company_name} {chunk.metadata.filing_type} "
            f"{chunk.metadata.filing_date} — {chunk.metadata.section}\n"
            f"{chunk.content}"
        )

    if graph_result and (graph_result.entities or graph_result.relationships):
        entity_lines = [
            f"- {e.name} (CIK: {e.cik})" for e in graph_result.entities if e.name
        ]
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
    return f"Context:\n{context}\n\nQuestion: {query}"


def _extract_citations(answer: str, chunks: list[Chunk]) -> list[Citation]:
    """Pull chunk_id references out of the model's answer.

    The system prompt instructs the model to cite as [chunk_id: <id>].
    We scan for that pattern and map each cited ID to its chunk.
    Citations for IDs not in the chunk list are silently dropped —
    the output harness flags those as unsupported claims.
    """
    import re

    chunk_map = {c.chunk_id: c for c in chunks}
    citations = []
    seen = set()

    pattern = re.compile(r"\[chunk_id:\s*([a-f0-9]{32})\]")

    for match in pattern.finditer(answer):
        chunk_id = match.group(1)
        if chunk_id in chunk_map and chunk_id not in seen:
            seen.add(chunk_id)
            chunk = chunk_map[chunk_id]
            claim_start = max(0, match.start() - 100)
            claim_text = answer[claim_start:match.start()].strip()

            citations.append(Citation(
                claim=claim_text[-200:] if len(claim_text) > 200 else claim_text,
                source_chunk_id=chunk_id,
                source_doc_id=chunk.doc_id,
                confidence=chunk.score,
            ))

    return citations


def _model_name() -> str:
    return settings.ollama.model


from finsight.config.settings import settings