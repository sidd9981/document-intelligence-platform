"""
Unit tests for the sparse encoder.

The actual SPLADE model is not loaded in unit tests. We mock the
tokenizer and model to verify the encoding logic and interface
without a 500MB model download.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
import torch

from finsight.services import sparse_encoder


@pytest.fixture(autouse=True)
def reset_encoder():
    sparse_encoder._tokenizer = None
    sparse_encoder._model = None
    yield
    sparse_encoder._tokenizer = None
    sparse_encoder._model = None


def make_mock_model_output(vocab_size: int = 30522, seq_len: int = 5) -> MagicMock:
    """Build a fake model output with known logits."""
    logits = torch.zeros(1, seq_len, vocab_size)
    logits[0, 0, 100] = 2.0
    logits[0, 1, 200] = 1.5
    logits[0, 2, 300] = 0.0

    output = MagicMock()
    output.logits = logits
    return output


def make_mock_tokenizer() -> MagicMock:
    tokenizer = MagicMock()
    tokenizer.return_value = {
        "input_ids": torch.tensor([[1, 2, 3, 4, 5]]),
        "attention_mask": torch.tensor([[1, 1, 1, 1, 1]]),
    }
    return tokenizer


def test_get_tokenizer_raises_before_init():
    with pytest.raises(RuntimeError, match="not initialized"):
        sparse_encoder.get_tokenizer()


def test_get_model_raises_before_init():
    with pytest.raises(RuntimeError, match="not initialized"):
        sparse_encoder.get_model()


def test_encode_sparse_raises_before_init():
    with pytest.raises(RuntimeError, match="not initialized"):
        sparse_encoder.encode_sparse("Apple supply chain risk")


def test_encode_sparse_returns_dict_of_nonzero_weights():
    mock_tokenizer = make_mock_tokenizer()
    mock_model = MagicMock()
    mock_model.return_value = make_mock_model_output()

    sparse_encoder._tokenizer = mock_tokenizer
    sparse_encoder._model = mock_model

    result = sparse_encoder.encode_sparse("Apple supply chain risk")

    assert isinstance(result, dict)
    assert len(result) > 0
    for token_id, weight in result.items():
        assert isinstance(token_id, int)
        assert isinstance(weight, float)
        assert weight > 0


def test_encode_sparse_omits_zero_weights():
    mock_tokenizer = make_mock_tokenizer()
    mock_model = MagicMock()

    logits = torch.zeros(1, 5, 30522)
    logits[0, 0, 100] = 2.0
    output = MagicMock()
    output.logits = logits

    mock_model.return_value = output
    sparse_encoder._tokenizer = mock_tokenizer
    sparse_encoder._model = mock_model

    result = sparse_encoder.encode_sparse("test text")

    for weight in result.values():
        assert weight > 0


def test_encode_sparse_batch_returns_one_vector_per_input():
    mock_tokenizer = make_mock_tokenizer()
    mock_model = MagicMock()

    batch_size = 3
    logits = torch.zeros(batch_size, 5, 30522)
    logits[0, 0, 100] = 2.0
    logits[1, 0, 200] = 1.5
    logits[2, 0, 300] = 1.0
    output = MagicMock()
    output.logits = logits

    mock_model.return_value = output
    sparse_encoder._tokenizer = mock_tokenizer
    sparse_encoder._model = mock_model

    texts = ["first text", "second text", "third text"]
    results = sparse_encoder.encode_sparse_batch(texts)

    assert len(results) == batch_size
    for result in results:
        assert isinstance(result, dict)


def test_close_encoder_resets_state():
    sparse_encoder._tokenizer = MagicMock()
    sparse_encoder._model = MagicMock()

    sparse_encoder.close_encoder()

    assert sparse_encoder._tokenizer is None
    assert sparse_encoder._model is None