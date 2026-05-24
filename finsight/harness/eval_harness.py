"""
Eval harness.

Async evaluation on sampled live traffic. Computes lightweight
metrics and logs full traces to Langfuse. Does not block the user
response — called with asyncio.create_task() after the response
is sent.

Metrics logged per trace:
    faithfulness: from the output harness LLM-as-judge score
    answer_relevance: keyword overlap between query and answer
    context_recall: fraction of retrieved chunks with score > 0.5
    prompt_version: stamped on every trace for regression detection

Full RAGAS metrics (context precision, answer correctness) are
documented as a Phase 6.5 upgrade once baseline Langfuse logging
is confirmed working.
"""

from __future__ import annotations

import logging
import random
from typing import TYPE_CHECKING

from langfuse import Langfuse

from finsight.config.settings import settings
from finsight.models.base import Chunk
from finsight.models.synthesis import SynthesisResult
from finsight.telemetry.tracing import get_tracer

if TYPE_CHECKING:
    pass

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

EVAL_SAMPLE_RATE = 0.10

_langfuse: Langfuse | None = None


def get_langfuse() -> Langfuse:
    """Return the shared Langfuse client, initializing it if needed."""
    global _langfuse
    if _langfuse is None:
        _langfuse = Langfuse(
            public_key=settings.langfuse.public_key,
            secret_key=settings.langfuse.secret_key,
            host=settings.langfuse.host,
        )
    return _langfuse


def _score_answer_relevance(query: str, answer: str) -> float:
    """Lightweight proxy for answer relevance.

    Checks what fraction of meaningful query terms appear in the
    answer. Not as accurate as an LLM judge but zero additional
    latency and zero additional cost.

    Full RAGAS answer relevance (embedding-based) is a Phase 6.5
    upgrade.
    """
    if not query or not answer:
        return 0.0

    stopwords = {"what", "is", "the", "a", "an", "of", "in", "for", "and", "or", "how", "does"}
    query_terms = [w.lower() for w in query.split() if w.lower() not in stopwords]

    if not query_terms:
        return 1.0

    answer_lower = answer.lower()
    matched = sum(1 for term in query_terms if term in answer_lower)
    return matched / len(query_terms)


def _score_context_recall(chunks: list[Chunk]) -> float:
    """Fraction of retrieved chunks with retrieval score above 0.5.

    A proxy for whether the retrieval pipeline is returning relevant
    context. If most chunks are below 0.5 the retrieval is struggling.
    """
    if not chunks:
        return 0.0
    above_threshold = sum(1 for c in chunks if c.score > 0.5)
    return above_threshold / len(chunks)


async def maybe_run_eval(
    query: str,
    result: SynthesisResult,
    chunks: list[Chunk],
    trace_id: str,
    team_id: str,
) -> None:
    """Evaluate a query-result pair and log to Langfuse if sampled.

    Called after the response is sent so it never blocks the user.
    Samples at EVAL_SAMPLE_RATE (10%) to limit Langfuse usage.

    Args:
        query: The original user query.
        result: The synthesis result from the output harness.
        chunks: The retrieved chunks that were passed to the LLM.
        trace_id: The OTEL trace ID, used as Langfuse trace ID so
                  traces are correlatable across both systems.
        team_id: Used as a tag for filtering in Langfuse.
    """
    if random.random() > EVAL_SAMPLE_RATE:
        return

    with tracer.start_as_current_span("eval_harness.sample") as span:
        span.set_attribute("trace_id", trace_id)
        span.set_attribute("sampled", True)

        try:
            await _log_to_langfuse(
                query=query,
                result=result,
                chunks=chunks,
                trace_id=trace_id,
                team_id=team_id,
            )
        except Exception as e:
            logger.warning("eval harness langfuse logging failed: %s", e)


async def _log_to_langfuse(
    query: str,
    result: SynthesisResult,
    chunks: list[Chunk],
    trace_id: str,
    team_id: str,
) -> None:
    """Create a Langfuse trace with retrieval and generation observations."""
    lf = get_langfuse()

    trace = lf.trace(
        id=trace_id,
        name="finsight_query",
        input=query,
        output=result.answer,
        metadata={
            "team_id": team_id,
            "prompt_version": result.prompt_version,
            "model_used": result.model_used,
            "chunks_retrieved": len(chunks),
        },
        tags=[team_id, result.prompt_version],
    )

    trace.span(
        name="retrieval",
        input=query,
        output={
            "chunks": [
                {
                    "chunk_id": c.chunk_id,
                    "score": c.score,
                    "section": c.metadata.section,
                    "ticker": c.metadata.ticker,
                }
                for c in chunks
            ]
        },
        metadata={"chunk_count": len(chunks)},
    )

    trace.generation(
        name="synthesis",
        model=result.model_used,
        input=query,
        output=result.answer,
        metadata={
            "prompt_version": result.prompt_version,
            "unsupported_claims": result.unsupported_claims,
            "hallucination_flags": result.hallucination_flags,
        },
    )

    answer_relevance = _score_answer_relevance(query, result.answer)
    context_recall = _score_context_recall(chunks)

    trace.score(name="faithfulness", value=result.faithfulness_score)
    trace.score(name="answer_relevance", value=answer_relevance)
    trace.score(name="context_recall", value=context_recall)

    lf.flush()

    logger.info(
        "eval harness logged trace %s faithfulness=%.2f relevance=%.2f recall=%.2f",
        trace_id,
        result.faithfulness_score,
        answer_relevance,
        context_recall,
    )