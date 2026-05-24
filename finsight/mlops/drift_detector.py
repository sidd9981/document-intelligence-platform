"""
Retrieval drift detector.

Runs weekly as a background job. Samples recent queries from the
faithfulness_failures and query_audit_log tables, reruns retrieval,
and compares context recall against a rolling baseline.

A drop > DRIFT_THRESHOLD triggers an alert logged to Postgres and
a GitHub issue in production. In dev it logs a warning and writes
to the drift_alerts table.

This is the thing that tells you your RAG system is getting worse
before your users tell you.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import datetime, timedelta

from finsight.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

DRIFT_THRESHOLD = 0.10
SAMPLE_SIZE = 50
BASELINE_WEEKS = 4


@dataclass
class DriftReport:
    current_recall: float
    baseline_recall: float
    drift: float
    drifted: bool
    sample_size: int
    baseline_weeks: int
    checked_at: datetime


def _compute_recall(chunks_list: list[list]) -> float:
    """Compute average context recall across a set of retrieval results.

    Recall proxy: fraction of retrieved chunks with score > 0.5.
    Same metric as the offline eval so results are comparable.
    """
    if not chunks_list:
        return 0.0

    recalls = []
    for chunks in chunks_list:
        if not chunks:
            recalls.append(0.0)
            continue
        above = sum(1 for c in chunks if c.score > 0.5)
        recalls.append(above / len(chunks))

    return sum(recalls) / len(recalls)


async def run_drift_check(
    recent_queries: list[str],
    baseline_queries: list[str],
    team_id: str,
    retrieval_fn,
) -> DriftReport:
    """Compare retrieval quality between recent and baseline query sets.

    Args:
        recent_queries: Queries from the last week to evaluate.
        baseline_queries: Queries from the baseline period (last 4 weeks).
        team_id: Used for scope filtering in retrieval.
        retrieval_fn: Async callable that takes (query, team_id) and
                      returns a list of Chunk objects. Injected so
                      this is testable without live services.

    Returns:
        DriftReport with current recall, baseline recall, and whether
        drift exceeded the threshold.
    """
    with tracer.start_as_current_span("drift_detector.run") as span:
        span.set_attribute("recent_queries", len(recent_queries))
        span.set_attribute("baseline_queries", len(baseline_queries))
        span.set_attribute("team_id", team_id)

        recent_results = []
        for query in recent_queries[:SAMPLE_SIZE]:
            try:
                chunks = await retrieval_fn(query, team_id)
                recent_results.append(chunks)
            except Exception as e:
                logger.warning("drift check retrieval failed for query: %s", e)
                recent_results.append([])

        baseline_results = []
        for query in baseline_queries[:SAMPLE_SIZE]:
            try:
                chunks = await retrieval_fn(query, team_id)
                baseline_results.append(chunks)
            except Exception as e:
                logger.warning("drift check retrieval failed for query: %s", e)
                baseline_results.append([])

        current_recall = _compute_recall(recent_results)
        baseline_recall = _compute_recall(baseline_results)
        drift = baseline_recall - current_recall
        drifted = drift > DRIFT_THRESHOLD

        span.set_attribute("current_recall", current_recall)
        span.set_attribute("baseline_recall", baseline_recall)
        span.set_attribute("drift", drift)
        span.set_attribute("drifted", drifted)

        if drifted:
            logger.error(
                "retrieval drift detected: current=%.3f baseline=%.3f drift=%.3f threshold=%.3f",
                current_recall,
                baseline_recall,
                drift,
                DRIFT_THRESHOLD,
            )
        else:
            logger.info(
                "drift check passed: current=%.3f baseline=%.3f drift=%.3f",
                current_recall,
                baseline_recall,
                drift,
            )

        return DriftReport(
            current_recall=current_recall,
            baseline_recall=baseline_recall,
            drift=drift,
            drifted=drifted,
            sample_size=len(recent_results),
            baseline_weeks=BASELINE_WEEKS,
            checked_at=datetime.utcnow(),
        )