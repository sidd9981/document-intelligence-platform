"""
Input harness.

Runs before every LLM call. Validates context quality, enforces the
tenant's context window limit, and reorders chunks for position bias.

If context quality is below the minimum threshold the harness returns
a flag so the orchestrator can return a low-confidence warning rather
than calling the LLM with garbage input.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass

from finsight.models.base import Chunk
from finsight.services.llm import count_tokens
from finsight.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

MIN_RETRIEVAL_SCORE = 0.5
INJECTION_PATTERNS = [
    re.compile(r"ignore\s+previous\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+all\s+prior", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a", re.IGNORECASE),
    re.compile(r"system\s*:\s*you\s+are", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
]


@dataclass
class InputHarnessResult:
    chunks: list[Chunk]
    low_confidence: bool
    injection_detected: bool
    prompt_version: str
    tokens_in_context: int


def run_input_harness(
    chunks: list[Chunk],
    max_context_tokens: int,
    prompt_version: str,
) -> InputHarnessResult:
    """Validate and prepare context before passing it to the synthesis agent.

    Args:
        chunks: Reranked chunks from the retrieval agent.
        max_context_tokens: The tenant's context window limit. Chunks
                            are trimmed from the bottom of the list until
                            the total fits.
        prompt_version: Stamped onto the result so every trace records
                        which prompt template was running.

    Returns:
        InputHarnessResult with the processed chunks and quality flags.
    """
    with tracer.start_as_current_span("input_harness.run") as span:
        span.set_attribute("chunks.input", len(chunks))
        span.set_attribute("max_context_tokens", max_context_tokens)

        low_confidence = _check_confidence(chunks)
        injection_detected = _check_injection(chunks)
        trimmed = _trim_to_context_window(chunks, max_context_tokens)
        ordered = _reorder_for_position_bias(trimmed)
        tokens_in_context = sum(c.token_count for c in ordered)

        span.set_attribute("chunks.output", len(ordered))
        span.set_attribute("low_confidence", low_confidence)
        span.set_attribute("injection_detected", injection_detected)
        span.set_attribute("tokens_in_context", tokens_in_context)

        return InputHarnessResult(
            chunks=ordered,
            low_confidence=low_confidence,
            injection_detected=injection_detected,
            prompt_version=prompt_version,
            tokens_in_context=tokens_in_context,
        )


def _check_confidence(chunks: list[Chunk]) -> bool:
    """Return True if the top retrieval score is below the minimum threshold.

    An empty chunk list is also low confidence — we have no context at all.
    """
    if not chunks:
        return True
    return max(c.score for c in chunks) < MIN_RETRIEVAL_SCORE


def _check_injection(chunks: list[Chunk]) -> bool:
    """Scan chunk content for prompt injection patterns.

    Financial filings are public documents that adversarial actors
    could craft to contain injection attempts. We scan retrieved
    chunks before passing them to the LLM.
    """
    for chunk in chunks:
        for pattern in INJECTION_PATTERNS:
            if pattern.search(chunk.content):
                logger.warning("injection pattern detected in chunk %s", chunk.chunk_id)
                return True
    return False


def _trim_to_context_window(
    chunks: list[Chunk],
    max_context_tokens: int,
) -> list[Chunk]:
    """Drop lowest-scoring chunks until total tokens fit within the limit.

    Chunks are already sorted by descending score so we drop from the
    tail first — lowest-relevance content goes first.
    """
    total = 0
    kept = []
    for chunk in chunks:
        if total + chunk.token_count <= max_context_tokens:
            kept.append(chunk)
            total += chunk.token_count
        else:
            break
    return kept


def _reorder_for_position_bias(chunks: list[Chunk]) -> list[Chunk]:
    """Best chunk first, second-best last, rest in the middle.

    Matches the reorder logic in the synthesis agent. Defined here
    because the input harness is the canonical place for context
    preparation — the synthesis agent delegates to this.
    """
    if len(chunks) <= 2:
        return chunks
    return [chunks[0]] + chunks[2:] + [chunks[1]]