"""
RAGAS CI gate.

Runs the offline eval and fails if any metric is below threshold.
This test is excluded from the normal unit test run — it requires
live services. Run explicitly with:

    pytest tests/eval/test_ragas_gate.py -v

Or add to CI after the unit test step with services running.
"""

import asyncio
import pytest

pytestmark = pytest.mark.eval


def test_eval_metrics_above_threshold():
    from eval.offline_eval import run_eval
    report = asyncio.run(run_eval())

    summary = report["summary"]
    thresholds = report["thresholds"]

    assert summary["avg_keyword_match"] >= thresholds["answer_keyword_match"], (
        f"keyword match {summary['avg_keyword_match']:.2f} below threshold "
        f"{thresholds['answer_keyword_match']}"
    )

    assert summary["avg_section_recall"] >= thresholds["context_recall"], (
        f"section recall {summary['avg_section_recall']:.2f} below threshold "
        f"{thresholds['context_recall']}"
    )