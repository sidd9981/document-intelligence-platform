"""
SPLADE sparse encoder for hybrid retrieval.

Produces sparse vectors from text for keyword-aware retrieval.
Used alongside dense embeddings in the hybrid search pipeline.
Fused with RRF in the retrieval agent.

Model: naver/splade-cocondenser-selfdistil
Why: strong retrieval quality, reasonable size (~500MB), well
documented behaviour on financial and technical text.
"""

from __future__ import annotations

import logging

import torch
from transformers import AutoModelForMaskedLM, AutoTokenizer

from finsight.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

SPLADE_MODEL_NAME = "naver/splade-cocondenser-selfdistil"
MAX_LENGTH = 512

_tokenizer = None
_model = None


def get_tokenizer():
    if _tokenizer is None:
        raise RuntimeError(
            "sparse encoder is not initialized. "
            "call init_encoder() during application startup."
        )
    return _tokenizer


def get_model():
    if _model is None:
        raise RuntimeError(
            "sparse encoder is not initialized. "
            "call init_encoder() during application startup."
        )
    return _model


def init_encoder() -> None:
    """Load the SPLADE model and tokenizer into memory.

    Called once at startup. The model is ~500MB and takes a few
    seconds to load. Subsequent calls return immediately.

    Uses CPU by default. If a GPU is available it will be used
    automatically via the device map.
    """
    global _tokenizer, _model

    if _tokenizer is not None:
        return

    with tracer.start_as_current_span("sparse_encoder.init"):
        logger.info("loading SPLADE model %s", SPLADE_MODEL_NAME)
        _tokenizer = AutoTokenizer.from_pretrained(SPLADE_MODEL_NAME)
        _model = AutoModelForMaskedLM.from_pretrained(SPLADE_MODEL_NAME)
        _model.eval()
        logger.info("SPLADE model loaded")


def close_encoder() -> None:
    global _tokenizer, _model
    _tokenizer = None
    _model = None


def encode_sparse(text: str) -> dict[int, float]:
    """Encode text into a SPLADE sparse vector.

    Returns a dict of {token_id: weight} where token_id is the
    vocabulary index and weight is the importance score. Most
    weights are zero and are omitted — only non-zero terms are
    returned. This is the format Qdrant's sparse vector API expects.

    Args:
        text: The text to encode. Keep under 512 tokens — longer
              inputs are truncated at the tokenizer level.

    Returns:
        Sparse vector as a dict of non-zero {token_id: weight} pairs.
    """
    tokenizer = get_tokenizer()
    model = get_model()

    with tracer.start_as_current_span("sparse_encoder.encode") as span:
        span.set_attribute("text_length", len(text))

        inputs = tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=False,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        activations = torch.log(1 + torch.relu(logits))
        sparse_vec = torch.max(activations, dim=1).values.squeeze()

        indices = sparse_vec.nonzero(as_tuple=True)[0].tolist()
        weights = sparse_vec[indices].tolist()

        result = {int(idx): float(w) for idx, w in zip(indices, weights) if w > 0}
        span.set_attribute("nonzero_terms", len(result))
        return result


def encode_sparse_batch(texts: list[str]) -> list[dict[int, float]]:
    """Encode a list of texts into sparse vectors.

    More efficient than calling encode_sparse() in a loop because
    the tokenizer and model handle batches in a single forward pass.

    Args:
        texts: List of texts to encode.

    Returns:
        List of sparse vectors in the same order as the input texts.
    """
    tokenizer = get_tokenizer()
    model = get_model()

    with tracer.start_as_current_span("sparse_encoder.encode_batch") as span:
        span.set_attribute("batch_size", len(texts))

        inputs = tokenizer(
            texts,
            return_tensors="pt",
            truncation=True,
            max_length=MAX_LENGTH,
            padding=True,
        )

        with torch.no_grad():
            outputs = model(**inputs)

        logits = outputs.logits
        activations = torch.log(1 + torch.relu(logits))
        batch_vecs = torch.max(activations, dim=1).values

        results = []
        for vec in batch_vecs:
            indices = vec.nonzero(as_tuple=True)[0].tolist()
            weights = vec[indices].tolist()
            results.append({int(idx): float(w) for idx, w in zip(indices, weights) if w > 0})

        return results