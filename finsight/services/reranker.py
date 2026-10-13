"""
Cross-encoder reranker.

Takes the candidate chunks from RRF fusion and rescores them by encoding
query and chunk content jointly. More accurate than retrieval scores alone
because the model attends to both at once.

Runs after RRF fusion, before the input harness. Only called on the top 50
candidates because cross-encoding the full corpus would be far too slow.

Scores are raw logits from cross-encoder/ms-marco-MiniLM-L-6-v2, roughly
in the range -11 to +11. They order chunks correctly but are not
probabilities. Use relevance_probability wherever a 0 to 1 value is needed,
such as confidence thresholds or citation confidence.
"""

from __future__ import annotations

import logging
import math

import torch
from transformers import AutoModelForSequenceClassification, AutoTokenizer

from finsight.models.base import Chunk
from finsight.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

RERANKER_MODEL_NAME = "cross-encoder/ms-marco-MiniLM-L-6-v2"
MAX_LENGTH = 512

_tokenizer = None
_model = None


def get_tokenizer():
    if _tokenizer is None:
        raise RuntimeError(
            "reranker is not initialized. "
            "call init_reranker() during application startup."
        )
    return _tokenizer


def get_model():
    if _model is None:
        raise RuntimeError(
            "reranker is not initialized. "
            "call init_reranker() during application startup."
        )
    return _model


def init_reranker() -> None:
    """Load the cross-encoder model and tokenizer.

    Called once at startup. Safe to call multiple times.
    """
    global _tokenizer, _model

    if _tokenizer is not None:
        return

    with tracer.start_as_current_span("reranker.init"):
        logger.info("loading reranker model %s", RERANKER_MODEL_NAME)
        _tokenizer = AutoTokenizer.from_pretrained(RERANKER_MODEL_NAME)
        _model = AutoModelForSequenceClassification.from_pretrained(RERANKER_MODEL_NAME)
        _model.eval()
        logger.info("reranker model loaded")


def close_reranker() -> None:
    global _tokenizer, _model
    _tokenizer = None
    _model = None


def relevance_probability(score: float) -> float:
    """Convert a cross-encoder logit into a relevance probability.

    ms-marco cross-encoders are trained with a binary cross-entropy
    objective, so the sigmoid of the logit is the model's estimate that
    the chunk is relevant to the query.
    """
    return 1.0 / (1.0 + math.exp(-score))


def rerank(query: str, chunks: list[Chunk], top_k: int) -> list[Chunk]:
    """Rerank chunks by joint query-chunk relevance.

    Scores each (query, chunk) pair with the cross-encoder and returns
    the top_k chunks sorted by descending score. The original retrieval
    scores are replaced with cross-encoder logits.

    Args:
        query: The raw query text.
        chunks: Candidate chunks from RRF fusion.
        top_k: Number of chunks to return after reranking.

    Returns:
        Top top_k chunks sorted by cross-encoder score, highest first.
    """
    if not chunks:
        return []

    tokenizer = get_tokenizer()
    model = get_model()

    with tracer.start_as_current_span("reranker.rerank") as span:
        span.set_attribute("candidates", len(chunks))
        span.set_attribute("top_k", top_k)

        pairs = [(query, chunk.content) for chunk in chunks]

        inputs = tokenizer(
            pairs,
            padding=True,
            truncation=True,
            max_length=MAX_LENGTH,
            return_tensors="pt",
        )

        with torch.no_grad():
            scores = model(**inputs).logits.squeeze(-1)

        if scores.dim() == 0:
            scores = scores.unsqueeze(0)

        scored_chunks = [
            chunk.model_copy(update={"score": float(score)})
            for chunk, score in zip(chunks, scores.tolist())
        ]

        reranked = sorted(scored_chunks, key=lambda c: c.score, reverse=True)
        span.set_attribute("returned", min(top_k, len(reranked)))
        return reranked[:top_k]