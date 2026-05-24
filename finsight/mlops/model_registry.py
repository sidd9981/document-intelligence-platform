"""
MLflow model registry wrapper.

Logs embedding model versions with their RAGAS scores and tracks
which version is currently in production. Every version maps to a
specific Qdrant collection so the blue/green pipeline knows where
to point the alias on promotion.

Promotion requires context_recall improvement >= MIN_RECALL_IMPROVEMENT
over the current production model. This prevents regressions from
silently entering production.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import mlflow
from mlflow.tracking import MlflowClient

from finsight.config.settings import settings
from finsight.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

EXPERIMENT_NAME = "finsight-embeddings"
MODEL_NAME = "finsight-embedding"
MIN_RECALL_IMPROVEMENT = 0.02


@dataclass
class ModelVersion:
    model_name: str
    model_version: str
    qdrant_collection: str
    context_recall: float
    context_precision: float
    is_production: bool
    run_id: str


def get_client() -> MlflowClient:
    mlflow.set_tracking_uri(settings.app.__dict__.get("mlflow_uri", "http://localhost:5001"))
    return MlflowClient()


def _get_or_create_experiment(client: MlflowClient) -> str:
    """Return the experiment ID, creating it if it doesn't exist."""
    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if experiment:
        return experiment.experiment_id
    return client.create_experiment(EXPERIMENT_NAME)


def register_model(
    model_name: str,
    model_version: str,
    qdrant_collection: str,
    context_recall: float,
    context_precision: float,
    doc_count: int,
) -> str:
    """Register a new embedding model version in MLflow.

    Logs the model with its eval metrics and collection name as
    params. Returns the MLflow run ID for reference.

    Args:
        model_name: HuggingFace model name, e.g. nomic-embed-text.
        model_version: Semantic version string, e.g. 1.0.0.
        qdrant_collection: The Qdrant collection this version owns.
        context_recall: RAGAS context recall score at registration.
        context_precision: RAGAS context precision score at registration.
        doc_count: Number of documents indexed in this collection.

    Returns:
        MLflow run ID.
    """
    with tracer.start_as_current_span("model_registry.register") as span:
        span.set_attribute("model_name", model_name)
        span.set_attribute("model_version", model_version)

        client = get_client()
        experiment_id = _get_or_create_experiment(client)

        with mlflow.start_run(experiment_id=experiment_id) as run:
            mlflow.log_params({
                "model_name": model_name,
                "model_version": model_version,
                "qdrant_collection": qdrant_collection,
                "doc_count": doc_count,
            })
            mlflow.log_metrics({
                "context_recall": context_recall,
                "context_precision": context_precision,
            })
            mlflow.set_tags({
                "model_name": model_name,
                "qdrant_collection": qdrant_collection,
                "is_production": "false",
            })

        logger.info(
            "registered model %s v%s recall=%.3f precision=%.3f run=%s",
            model_name,
            model_version,
            context_recall,
            context_precision,
            run.info.run_id,
        )
        return run.info.run_id


def get_production_model(client: MlflowClient | None = None) -> ModelVersion | None:
    """Return the current production model version, or None if none promoted yet."""
    if client is None:
        client = get_client()

    experiment = client.get_experiment_by_name(EXPERIMENT_NAME)
    if not experiment:
        return None

    runs = client.search_runs(
        experiment_ids=[experiment.experiment_id],
        filter_string="tags.is_production = 'true'",
        order_by=["start_time DESC"],
        max_results=1,
    )

    if not runs:
        return None

    run = runs[0]
    return ModelVersion(
        model_name=run.data.params.get("model_name", ""),
        model_version=run.data.params.get("model_version", ""),
        qdrant_collection=run.data.params.get("qdrant_collection", ""),
        context_recall=run.data.metrics.get("context_recall", 0.0),
        context_precision=run.data.metrics.get("context_precision", 0.0),
        is_production=True,
        run_id=run.info.run_id,
    )


def promote_model(run_id: str) -> bool:
    """Promote a model version to production if it meets the improvement threshold.

    Compares context_recall of the candidate run against the current
    production model. Promotion requires >= MIN_RECALL_IMPROVEMENT
    improvement. If no production model exists, promotes unconditionally.

    Args:
        run_id: MLflow run ID of the candidate model.

    Returns:
        True if promoted, False if threshold not met.
    """
    with tracer.start_as_current_span("model_registry.promote") as span:
        span.set_attribute("run_id", run_id)

        client = get_client()
        candidate_run = client.get_run(run_id)
        candidate_recall = candidate_run.data.metrics.get("context_recall", 0.0)

        current = get_production_model(client)

        if current is not None:
            improvement = candidate_recall - current.context_recall
            span.set_attribute("recall_improvement", improvement)

            if improvement < MIN_RECALL_IMPROVEMENT:
                logger.warning(
                    "promotion rejected: recall improvement %.3f below threshold %.3f",
                    improvement,
                    MIN_RECALL_IMPROVEMENT,
                )
                return False

            client.set_tag(current.run_id, "is_production", "false")

        client.set_tag(run_id, "is_production", "true")

        logger.info(
            "promoted run %s to production recall=%.3f",
            run_id,
            candidate_recall,
        )
        return True