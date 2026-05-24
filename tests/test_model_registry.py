"""
Unit tests for the MLflow model registry wrapper.

Mocks the MLflow client so no running MLflow instance is needed.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

from finsight.mlops.model_registry import (
    MIN_RECALL_IMPROVEMENT,
    ModelVersion,
    get_production_model,
    promote_model,
    register_model,
)


def _make_run(run_id: str, recall: float, precision: float, collection: str, is_production: bool):
    run = MagicMock()
    run.info.run_id = run_id
    run.data.params = {
        "model_name": "nomic-embed-text",
        "model_version": "1.0.0",
        "qdrant_collection": collection,
        "doc_count": "1000",
    }
    run.data.metrics = {
        "context_recall": recall,
        "context_precision": precision,
    }
    run.data.tags = {"is_production": str(is_production).lower()}
    return run


@pytest.fixture
def mock_mlflow():
    with (
        patch("finsight.mlops.model_registry.mlflow") as mock_mf,
        patch("finsight.mlops.model_registry.MlflowClient") as mock_client_cls,
    ):
        mock_client = MagicMock()
        mock_client_cls.return_value = mock_client
        mock_mf.start_run.return_value.__enter__ = MagicMock(
            return_value=MagicMock(info=MagicMock(run_id="run-001"))
        )
        mock_mf.start_run.return_value.__exit__ = MagicMock(return_value=False)
        yield mock_mf, mock_client


def test_register_model_returns_run_id(mock_mlflow):
    mock_mf, mock_client = mock_mlflow
    mock_client.get_experiment_by_name.return_value = MagicMock(experiment_id="exp-1")

    run_id = register_model(
        model_name="nomic-embed-text",
        model_version="1.0.0",
        qdrant_collection="filings_dense_v1",
        context_recall=0.72,
        context_precision=0.78,
        doc_count=1000,
    )
    assert run_id == "run-001"


def test_register_model_logs_metrics(mock_mlflow):
    mock_mf, mock_client = mock_mlflow
    mock_client.get_experiment_by_name.return_value = MagicMock(experiment_id="exp-1")

    register_model(
        model_name="nomic-embed-text",
        model_version="1.0.0",
        qdrant_collection="filings_dense_v1",
        context_recall=0.72,
        context_precision=0.78,
        doc_count=1000,
    )

    mock_mf.log_metrics.assert_called_once_with({
        "context_recall": 0.72,
        "context_precision": 0.78,
    })


def test_get_production_model_returns_none_when_no_runs(mock_mlflow):
    mock_mf, mock_client = mock_mlflow
    mock_client.get_experiment_by_name.return_value = MagicMock(experiment_id="exp-1")
    mock_client.search_runs.return_value = []

    result = get_production_model(mock_client)
    assert result is None


def test_get_production_model_returns_model_when_exists(mock_mlflow):
    mock_mf, mock_client = mock_mlflow
    mock_client.get_experiment_by_name.return_value = MagicMock(experiment_id="exp-1")
    mock_client.search_runs.return_value = [
        _make_run("run-001", 0.72, 0.78, "filings_dense_v1", True)
    ]

    result = get_production_model(mock_client)
    assert result is not None
    assert result.context_recall == 0.72
    assert result.qdrant_collection == "filings_dense_v1"
    assert result.is_production is True


def test_promote_model_promotes_when_no_current_production(mock_mlflow):
    mock_mf, mock_client = mock_mlflow
    mock_client.get_experiment_by_name.return_value = None
    mock_client.get_run.return_value = _make_run(
        "run-001", 0.72, 0.78, "filings_dense_v1", False
    )

    with patch("finsight.mlops.model_registry.get_production_model", return_value=None):
        result = promote_model("run-001")

    assert result is True
    mock_client.set_tag.assert_called_with("run-001", "is_production", "true")


def test_promote_model_rejects_insufficient_improvement(mock_mlflow):
    mock_mf, mock_client = mock_mlflow
    mock_client.get_run.return_value = _make_run(
        "run-002", 0.73, 0.78, "filings_dense_v2", False
    )

    current = ModelVersion(
        model_name="nomic-embed-text",
        model_version="1.0.0",
        qdrant_collection="filings_dense_v1",
        context_recall=0.72,
        context_precision=0.78,
        is_production=True,
        run_id="run-001",
    )

    with patch("finsight.mlops.model_registry.get_production_model", return_value=current):
        result = promote_model("run-002")

    assert result is False


def test_promote_model_promotes_with_sufficient_improvement(mock_mlflow):
    mock_mf, mock_client = mock_mlflow
    mock_client.get_run.return_value = _make_run(
        "run-002", 0.75, 0.80, "filings_dense_v2", False
    )

    current = ModelVersion(
        model_name="nomic-embed-text",
        model_version="1.0.0",
        qdrant_collection="filings_dense_v1",
        context_recall=0.72,
        context_precision=0.78,
        is_production=True,
        run_id="run-001",
    )

    with patch("finsight.mlops.model_registry.get_production_model", return_value=current):
        result = promote_model("run-002")

    assert result is True
    mock_client.set_tag.assert_any_call("run-001", "is_production", "false")
    mock_client.set_tag.assert_any_call("run-002", "is_production", "true")


def test_promote_model_demotes_current_on_promotion(mock_mlflow):
    mock_mf, mock_client = mock_mlflow
    mock_client.get_run.return_value = _make_run(
        "run-002", 0.75, 0.80, "filings_dense_v2", False
    )

    current = ModelVersion(
        model_name="nomic-embed-text",
        model_version="1.0.0",
        qdrant_collection="filings_dense_v1",
        context_recall=0.72,
        context_precision=0.78,
        is_production=True,
        run_id="run-001",
    )

    with patch("finsight.mlops.model_registry.get_production_model", return_value=current):
        promote_model("run-002")

    calls = [str(c) for c in mock_client.set_tag.call_args_list]
    assert any("run-001" in c and "false" in c for c in calls)