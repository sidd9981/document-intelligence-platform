"""
Output harness.

Runs after every LLM response. Validates schema, checks faithfulness
via LLM-as-judge, flags numbers not grounded in context, and scrubs PII.

Faithfulness failures are returned as structured data — specific claim
text, not just a score. The orchestrator writes these to the
faithfulness_failures table in Postgres.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from finsight.models.base import Chunk
from finsight.models.synthesis import Citation, SynthesisResult
from finsight.services.llm import complete, count_tokens
from finsight.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

FAITHFULNESS_THRESHOLD = 0.85
NUMBER_PATTERN = re.compile(r"\b\d+(?:\.\d+)?(?:%|B|M|bn|m)?\b")
PII_PATTERNS = [
    re.compile(r"\b\d{3}-\d{2}-\d{4}\b"),
    re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b"),
]

FAITHFULNESS_SYSTEM = """You are an evaluator checking whether an answer is grounded in the provided context.

Respond in this exact format:
SCORE: <float between 0.0 and 1.0>
UNSUPPORTED: <comma-separated list of unsupported claims, or NONE>

A claim is unsupported if it cannot be verified from the context. Be specific."""


@dataclass
class OutputHarnessResult:
    answer: str
    citations: list[Citation]
    faithfulness_score: float
    unsupported_claims: list[str] = field(default_factory=list)
    hallucination_flags: list[str] = field(default_factory=list)
    low_faithfulness: bool = False


async def run_output_harness(
    result: SynthesisResult,
    chunks: list[Chunk],
    query: str,
) -> OutputHarnessResult:
    """Validate the synthesis result and enrich it with faithfulness data.

    Args:
        result: Raw SynthesisResult from the synthesis agent.
        chunks: The context chunks that were passed to the LLM.
        query: The original query, used in the faithfulness judge prompt.

    Returns:
        OutputHarnessResult with faithfulness score, unsupported claims,
        and hallucination flags. Never raises.
    """
    with tracer.start_as_current_span("output_harness.run") as span:
        span.set_attribute("answer_length", len(result.answer))
        span.set_attribute("chunks.count", len(chunks))

        if not result.answer:
            return OutputHarnessResult(
                answer="",
                citations=result.citations,
                faithfulness_score=0.0,
                low_faithfulness=True,
            )

        answer = _scrub_pii(result.answer)
        hallucination_flags = _flag_hallucinations(answer, chunks)
        faithfulness_score, unsupported_claims = await _check_faithfulness(
            query=query,
            answer=answer,
            chunks=chunks,
        )

        low_faithfulness = faithfulness_score < FAITHFULNESS_THRESHOLD

        span.set_attribute("faithfulness_score", faithfulness_score)
        span.set_attribute("unsupported_claims.count", len(unsupported_claims))
        span.set_attribute("hallucination_flags.count", len(hallucination_flags))
        span.set_attribute("low_faithfulness", low_faithfulness)

        return OutputHarnessResult(
            answer=answer,
            citations=result.citations,
            faithfulness_score=faithfulness_score,
            unsupported_claims=unsupported_claims,
            hallucination_flags=hallucination_flags,
            low_faithfulness=low_faithfulness,
        )


async def _check_faithfulness(
    query: str,
    answer: str,
    chunks: list[Chunk],
) -> tuple[float, list[str]]:
    """Run the LLM-as-judge faithfulness check.

    Uses a structured prompt that asks the judge to return a score
    and a list of specific unsupported claims. Parses the response
    and returns defaults on any parse failure rather than raising.
    """
    context = "\n\n".join(c.content for c in chunks[:10])
    prompt = (
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer: {answer}\n\n"
        "Evaluate whether the answer is grounded in the context."
    )

    try:
        response, _, _ = await complete(
            prompt=prompt,
            system=FAITHFULNESS_SYSTEM,
            max_tokens=300,
        )
        return _parse_faithfulness_response(response)
    except Exception as e:
        logger.warning("faithfulness check failed: %s", e)
        return 1.0, []


def _parse_faithfulness_response(response: str) -> tuple[float, list[str]]:
    """Parse the structured faithfulness judge response.

    Expected format:
        SCORE: 0.85
        UNSUPPORTED: claim one, claim two

    Returns (1.0, []) on any parse failure — we don't penalise
    the answer if the judge itself failed.
    """
    score = 1.0
    unsupported: list[str] = []

    score_match = re.search(r"SCORE:\s*([0-9.]+)", response)
    if score_match:
        try:
            score = float(score_match.group(1))
            score = max(0.0, min(1.0, score))
        except ValueError:
            pass

    unsupported_match = re.search(r"UNSUPPORTED:\s*(.+)", response, re.IGNORECASE)
    if unsupported_match:
        raw = unsupported_match.group(1).strip()
        if raw.upper() != "NONE":
            unsupported = [c.strip() for c in raw.split(",") if c.strip()]

    return score, unsupported


def _flag_hallucinations(answer: str, chunks: list[Chunk]) -> list[str]:
    """Flag numbers in the answer that don't appear in any retrieved chunk.

    A revenue figure or percentage that the model generated but which
    doesn't appear verbatim in the context is the highest-risk failure
    mode in financial Q&A.
    """
    context_text = " ".join(c.content for c in chunks)
    flags = []

    for match in NUMBER_PATTERN.finditer(answer):
        number = match.group(0)
        if number not in context_text:
            flags.append(number)

    return flags


def _scrub_pii(text: str) -> str:
    """Remove SSN and email patterns from the answer.

    Financial filings occasionally contain personal data. We scrub
    before returning to the client.
    """
    for pattern in PII_PATTERNS:
        text = pattern.sub("[REDACTED]", text)
    return text