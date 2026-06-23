"""
RAGAS-equivalent metrics implemented directly against the Ollama LLM.

Implements the four canonical RAG evaluation metrics without the RAGAS
library dependency. Each metric uses the same LLM-as-judge approach
that RAGAS uses internally — we just own the prompts and parsing
directly instead of routing through a broken dependency chain.

Metric definitions:
    context_precision  — of retrieved chunks, what fraction are actually
                         relevant to answering the question
    context_recall     — of the information needed to answer, what fraction
                         is present in the retrieved chunks
    faithfulness       — of claims made in the answer, what fraction are
                         directly supported by the retrieved context
    answer_relevancy   — does the answer actually address what was asked,
                         or does it drift, hedge, or answer a different question

All four use structured LLM judge prompts that return numeric scores.
Scores are floats in [0.0, 1.0]. Parse failures return None and are
excluded from averages rather than zeroing the metric.

In an interview: "We implemented the four RAGAS metrics directly as
LLM judge calls rather than taking the library dependency. Each metric
is a structured prompt against the same Ollama instance we use for
synthesis. This means the eval is self-contained, auditable, and we
understand exactly what each score measures."
"""

from __future__ import annotations

import logging
import re

logger = logging.getLogger(__name__)


# ---------------------------------------------------------------------------
# Prompt templates
# ---------------------------------------------------------------------------

_CONTEXT_PRECISION_PROMPT = """You are evaluating a retrieval system.

Question: {question}

Retrieved context chunk:
{chunk}

Is this context chunk relevant and useful for answering the question above?
A chunk is relevant if it contains information that directly helps answer the question.

Respond with exactly one line in this format:
RELEVANT: yes
or
RELEVANT: no

Do not add any explanation."""


_CONTEXT_RECALL_PROMPT = """You are evaluating a retrieval system.

Question: {question}

Reference answer (ground truth):
{ground_truth}

Retrieved context:
{context}

Does the retrieved context contain enough information to produce the reference answer?
Consider: are the key facts, claims, and details in the reference answer present in the context?

Respond with a score from 0.0 to 1.0 where:
1.0 = all key information needed for the reference answer is present in the context
0.5 = about half the key information is present
0.0 = none of the key information is present

Respond with exactly one line in this format:
SCORE: 0.8

Do not add any explanation."""


_FAITHFULNESS_PROMPT = """You are evaluating whether an answer is faithful to its source context.

Context:
{context}

Answer:
{answer}

List each distinct factual claim made in the answer. For each claim, determine if it is
directly supported by the context above.

Then provide an overall faithfulness score from 0.0 to 1.0 where:
1.0 = every claim in the answer is supported by the context
0.5 = about half the claims are supported
0.0 = no claims are supported by the context

Respond in exactly this format:
SCORE: 0.9
UNSUPPORTED: <comma-separated list of unsupported claims, or NONE>

Do not add any other text."""


_ANSWER_RELEVANCY_PROMPT = """You are evaluating whether an answer is relevant to the question asked.

Question: {question}

Answer: {answer}

Score how well the answer addresses the question from 0.0 to 1.0 where:
1.0 = the answer directly and completely addresses the question
0.5 = the answer partially addresses the question or includes significant irrelevant content
0.0 = the answer does not address the question at all

Respond with exactly one line in this format:
SCORE: 0.9

Do not add any explanation."""


# ---------------------------------------------------------------------------
# Score parsers
# ---------------------------------------------------------------------------

def _parse_score(response: str) -> float | None:
    """Extract a SCORE: float from a judge response.

    Returns None on any parse failure so the caller can exclude this
    item from the average rather than counting it as zero.
    """
    match = re.search(r"SCORE:\s*([0-9.]+)", response, re.IGNORECASE)
    if not match:
        return None
    try:
        val = float(match.group(1))
        return max(0.0, min(1.0, val))
    except ValueError:
        return None


def _parse_relevant(response: str) -> bool | None:
    """Extract a RELEVANT: yes/no from a judge response."""
    match = re.search(r"RELEVANT:\s*(yes|no)", response, re.IGNORECASE)
    if not match:
        return None
    return match.group(1).lower() == "yes"


def _parse_faithfulness(response: str) -> tuple[float | None, list[str]]:
    """Extract score and unsupported claims from faithfulness response."""
    score = _parse_score(response)
    unsupported: list[str] = []

    match = re.search(r"UNSUPPORTED:\s*(.+)", response, re.IGNORECASE)
    if match:
        raw = match.group(1).strip()
        if raw.upper() != "NONE":
            unsupported = [c.strip() for c in raw.split(",") if c.strip()]

    return score, unsupported


# ---------------------------------------------------------------------------
# Individual metric functions
# ---------------------------------------------------------------------------

async def score_context_precision(
    question: str,
    contexts: list[str],
    complete_fn,
) -> float | None:
    """Score what fraction of retrieved chunks are relevant to the question.

    Judges each chunk independently then averages. A retrieval pipeline
    with high precision returns mostly relevant chunks. Low precision means
    the ranker is including noise that dilutes the context window.

    Args:
        question: The original user query.
        contexts: List of retrieved chunk texts.
        complete_fn: The async complete() function from finsight.services.llm.

    Returns:
        Float in [0.0, 1.0] or None if all judge calls failed.
    """
    if not contexts:
        return 0.0

    relevant_count = 0
    scored_count = 0

    for chunk in contexts:
        prompt = _CONTEXT_PRECISION_PROMPT.format(
            question=question,
            chunk=chunk[:800],
        )
        try:
            response, _, _ = await complete_fn(
                prompt=prompt,
                system="You are a precise relevance judge. Follow the format exactly.",
                max_tokens=20,
            )
            result = _parse_relevant(response)
            if result is not None:
                scored_count += 1
                if result:
                    relevant_count += 1
        except Exception as e:
            logger.warning("context_precision judge call failed: %s", e)

    if scored_count == 0:
        return None
    return relevant_count / scored_count


async def score_context_recall(
    question: str,
    contexts: list[str],
    ground_truth: str,
    complete_fn,
) -> float | None:
    """Score what fraction of needed information is in the retrieved context.

    Uses the ground truth reference answer to determine what information
    was needed, then asks the judge whether the retrieved context contains
    that information. High recall means the retrieval pipeline found the
    right documents. Low recall means relevant documents were missed.

    Args:
        question: The original user query.
        contexts: List of retrieved chunk texts.
        ground_truth: The reference answer from the golden dataset.
        complete_fn: The async complete() function from finsight.services.llm.

    Returns:
        Float in [0.0, 1.0] or None if the judge call failed.
    """
    if not contexts:
        return 0.0

    combined_context = "\n\n---\n\n".join(c[:600] for c in contexts[:8])
    prompt = _CONTEXT_RECALL_PROMPT.format(
        question=question,
        ground_truth=ground_truth,
        context=combined_context,
    )

    try:
        response, _, _ = await complete_fn(
            prompt=prompt,
            system="You are a precise recall judge. Follow the format exactly.",
            max_tokens=20,
        )
        return _parse_score(response)
    except Exception as e:
        logger.warning("context_recall judge call failed: %s", e)
        return None


async def score_faithfulness(
    answer: str,
    contexts: list[str],
    complete_fn,
) -> tuple[float | None, list[str]]:
    """Score what fraction of answer claims are grounded in context.

    This is the highest-stakes metric for a financial Q&A system. An
    answer that introduces numbers or facts not present in the retrieved
    context is a hallucination. The judge returns both a score and a
    list of specific unsupported claims for structured logging.

    Args:
        answer: The generated answer text.
        contexts: List of retrieved chunk texts used to generate the answer.
        complete_fn: The async complete() function from finsight.services.llm.

    Returns:
        Tuple of (score, unsupported_claims). Score is float in [0.0, 1.0]
        or None if the judge call failed.
    """
    if not answer:
        return 0.0, []

    combined_context = "\n\n---\n\n".join(c[:600] for c in contexts[:8])
    prompt = _FAITHFULNESS_PROMPT.format(
        context=combined_context,
        answer=answer,
    )

    try:
        response, _, _ = await complete_fn(
            prompt=prompt,
            system="You are a precise faithfulness judge. Follow the format exactly.",
            max_tokens=200,
        )
        score, unsupported = _parse_faithfulness(response)
        return score, unsupported
    except Exception as e:
        logger.warning("faithfulness judge call failed: %s", e)
        return None, []


async def score_answer_relevancy(
    question: str,
    answer: str,
    complete_fn,
) -> float | None:
    """Score how well the answer addresses the question.

    Catches answers that are technically grounded but don't actually
    answer what was asked — a common failure mode when the synthesis
    prompt is too permissive or the context is off-topic.

    Args:
        question: The original user query.
        answer: The generated answer text.
        complete_fn: The async complete() function from finsight.services.llm.

    Returns:
        Float in [0.0, 1.0] or None if the judge call failed.
    """
    if not answer:
        return 0.0

    prompt = _ANSWER_RELEVANCY_PROMPT.format(
        question=question,
        answer=answer,
    )

    try:
        response, _, _ = await complete_fn(
            prompt=prompt,
            system="You are a precise relevancy judge. Follow the format exactly.",
            max_tokens=20,
        )
        return _parse_score(response)
    except Exception as e:
        logger.warning("answer_relevancy judge call failed: %s", e)
        return None


# ---------------------------------------------------------------------------
# Convenience: score all four metrics for one query-answer pair
# ---------------------------------------------------------------------------

async def score_all(
    question: str,
    answer: str,
    contexts: list[str],
    ground_truth: str,
    complete_fn,
) -> dict:
    """Run all four metrics for a single query-answer pair.

    Args:
        question: The original user query.
        answer: The generated answer.
        contexts: Retrieved chunk texts passed to the LLM.
        ground_truth: Reference answer from the golden dataset.
        complete_fn: The async complete() function.

    Returns:
        Dict with keys: context_precision, context_recall, faithfulness,
        answer_relevancy, unsupported_claims. Values are floats or None.
    """
    cp = await score_context_precision(question, contexts, complete_fn)
    cr = await score_context_recall(question, contexts, ground_truth, complete_fn)
    f, unsupported = await score_faithfulness(answer, contexts, complete_fn)
    ar = await score_answer_relevancy(question, answer, complete_fn)

    return {
        "context_precision": cp,
        "context_recall": cr,
        "faithfulness": f,
        "answer_relevancy": ar,
        "unsupported_claims": unsupported,
    }