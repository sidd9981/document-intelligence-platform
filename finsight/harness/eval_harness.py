"""
Eval harness.

Online evaluation on sampled live traffic. Computes lightweight metrics
and logs full traces to Langfuse. Samples EVAL_SAMPLE_RATE of requests
to limit Langfuse volume.

Metrics logged per sampled trace:
    faithfulness: the output harness LLM-judge score
    answer_relevance: share of non-stopword query terms found in the answer,
        a cheap lexical proxy, not the ragas metric
    context_recall: share of retrieved chunks with score above 0.5, a
        proxy only. Scores are cross-encoder logits, so this threshold is
        not calibrated and the name overstates what it measures
    prompt_version: stamped on every trace for regression detection

The offline eval in eval/offline_eval.py computes the real ragas metrics.
"""

from __future__ import annotations

import logging
import random

from langfuse import Langfuse

from finsight.config.settings import settings
from finsight.models.base import Chunk
from finsight.models.synthesis import SynthesisResult
from finsight.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

EVAL_SAMPLE_RATE = 0.1

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
    """Lexical proxy for answer relevance.

    Returns the fraction of non-stopword query terms that appear in the
    answer. Zero added latency and cost, but much weaker than an LLM judge.
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
    """Fraction of retrieved chunks with a score above 0.5.

    A rough proxy for retrieval health, not true recall, which needs
    ground-truth relevant chunks.
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
    """Evaluate a query-result pair and log to Langfuse if this request is sampled.

    Samples EVAL_SAMPLE_RATE of requests. Logging failures are caught and
    logged so they never affect the user's response.

    Args:
        query: The original user query.
        result: The synthesis result from the output harness.
        chunks: The retrieved chunks that were passed to the LLM.
        trace_id: Used as the Langfuse trace id so traces correlate with OTEL.
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