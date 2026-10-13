"""
Input harness.

Runs before every synthesis call. Drops chunks containing prompt-injection
patterns, flags low retrieval confidence, trims context to the tenant's
budget after reserving room for the rest of the prompt and the output,
and orders chunks for position bias.

This is the only place chunk order is decided. The synthesis agent uses
the chunks in the order returned here.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

from finsight.models.base import Chunk
from finsight.services.reranker import relevance_probability
from finsight.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

MIN_RELEVANCE_PROBABILITY = 0.5
DEFAULT_RESERVED_TOKENS = 1024
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
    dropped_chunk_ids: list[str] = field(default_factory=list)


def run_input_harness(
    chunks: list[Chunk],
    max_context_tokens: int,
    prompt_version: str,
    reserved_tokens: int = DEFAULT_RESERVED_TOKENS,
) -> InputHarnessResult:
    """Validate and prepare context before passing it to the synthesis agent.

    Args:
        chunks: Reranked chunks from the retrieval agent, best first.
        max_context_tokens: The tenant's context window limit.
        prompt_version: Stamped onto the result so every trace records
            which prompt template was running.
        reserved_tokens: Tokens held back for the system prompt, question,
            graph context, conversation history and the model's output.
            Chunks are packed into max_context_tokens minus this.

    Returns:
        InputHarnessResult with the processed chunks and quality flags.
    """
    with tracer.start_as_current_span("input_harness.run") as span:
        span.set_attribute("chunks.input", len(chunks))
        span.set_attribute("max_context_tokens", max_context_tokens)
        span.set_attribute("reserved_tokens", reserved_tokens)

        clean, dropped = _drop_injected(chunks)
        low_confidence = _check_confidence(clean)
        budget = max(0, max_context_tokens - reserved_tokens)
        trimmed = _trim_to_budget(clean, budget)
        ordered = _reorder_for_position_bias(trimmed)
        tokens_in_context = sum(c.token_count for c in ordered)

        span.set_attribute("chunks.output", len(ordered))
        span.set_attribute("chunks.dropped_injection", len(dropped))
        span.set_attribute("low_confidence", low_confidence)
        span.set_attribute("tokens_in_context", tokens_in_context)

        return InputHarnessResult(
            chunks=ordered,
            low_confidence=low_confidence,
            injection_detected=bool(dropped),
            prompt_version=prompt_version,
            tokens_in_context=tokens_in_context,
            dropped_chunk_ids=dropped,
        )


def _drop_injected(chunks: list[Chunk]) -> tuple[list[Chunk], list[str]]:
    """Remove chunks matching any injection pattern.

    Returns the clean chunks in their original order and the ids of the
    dropped ones, so the drop is visible in traces rather than silent.
    """
    clean: list[Chunk] = []
    dropped: list[str] = []
    for chunk in chunks:
        if any(pattern.search(chunk.content) for pattern in INJECTION_PATTERNS):
            logger.warning("dropping chunk %s, injection pattern detected", chunk.chunk_id)
            dropped.append(chunk.chunk_id)
        else:
            clean.append(chunk)
    return clean, dropped


def _check_confidence(chunks: list[Chunk]) -> bool:
    """Return True if the best chunk's relevance probability is below the threshold.

    An empty chunk list is also low confidence, there is no context at all.
    """
    if not chunks:
        return True
    best = max(relevance_probability(c.score) for c in chunks)
    return best < MIN_RELEVANCE_PROBABILITY


def _trim_to_budget(chunks: list[Chunk], budget: int) -> list[Chunk]:
    """Keep the highest-ranked chunks that fit within the token budget.

    Chunks arrive best first, so stopping at the first chunk that does not
    fit drops the least relevant content.
    """
    total = 0
    kept: list[Chunk] = []
    for chunk in chunks:
        if total + chunk.token_count > budget:
            break
        kept.append(chunk)
        total += chunk.token_count
    return kept


def _reorder_for_position_bias(chunks: list[Chunk]) -> list[Chunk]:
    """Best chunk first, second best last, the rest in between.

    Models attend most to the start and end of their context.
    """
    if len(chunks) <= 2:
        return chunks
    return [chunks[0]] + chunks[2:] + [chunks[1]]