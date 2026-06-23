"""
RAGAS CI gate.

Runs the offline eval and fails if any real RAGAS metric is below
its threshold. This test requires live services (Qdrant, Postgres,
Ollama). Excluded from the normal unit test run.

Run explicitly:
    pytest tests/eval/test_ragas_gate.py -v

Or in CI after services are healthy:
    pytest tests/eval/test_ragas_gate.py -v -m eval
"""

import asyncio
import pytest

pytestmark = pytest.mark.eval

METRICS = ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]


def test_eval_metrics_above_threshold():
    """All four RAGAS metrics must exceed their configured thresholds.

    Fails with a specific message showing which metric failed,
    its actual score, and the threshold it missed.
    """
    from eval.offline_eval import run_eval

    report = asyncio.run(run_eval())
    summary = report["summary"]
    thresholds = report["thresholds"]

    failures = []
    for metric in METRICS:
        if metric not in thresholds:
            continue
        actual = summary.get(metric, 0.0)
        threshold = thresholds[metric]
        if actual < threshold:
            failures.append(
                f"{metric}: {actual:.3f} < {threshold:.2f} (threshold)"
            )

    assert not failures, (
        "RAGAS eval failed on the following metrics:\n"
        + "\n".join(f"  {f}" for f in failures)
    )