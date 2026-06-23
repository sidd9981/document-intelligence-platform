"""
Offline eval runner — real RAGAS-equivalent metrics, no library dependency.

Computes all four canonical RAG evaluation metrics against the golden
dataset using our own LLM judge implementation in eval/ragas_metrics.py.
Results are logged to Langfuse and written to stdout.

Run manually (main venv, services running):
    python eval/offline_eval.py

Run as CI gate:
    pytest tests/eval/test_ragas_gate.py -v -m eval
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import yaml
from langfuse import Langfuse

from finsight.config.settings import settings
from finsight.gateway.db import close_pool, init_pool
from finsight.services.llm import close_client, complete, embed, init_client
from finsight.services.vector_store import (
    close_client as close_qdrant,
    init_client as init_qdrant,
    search_dense,
)
from finsight.telemetry.tracing import setup_tracing
from eval.ragas_metrics import score_all

logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset" / "queries.json"
THRESHOLDS_PATH = Path(__file__).parent / "thresholds.yaml"

lf = Langfuse(
    public_key=settings.langfuse.public_key,
    secret_key=settings.langfuse.secret_key,
    host=settings.langfuse.host,
)


def load_thresholds() -> dict:
    if THRESHOLDS_PATH.exists():
        with open(THRESHOLDS_PATH) as f:
            return yaml.safe_load(f)
    return {
        "context_precision": 0.60,
        "context_recall": 0.60,
        "faithfulness": 0.75,
        "answer_relevancy": 0.60,
    }


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH) as f:
        return json.load(f)


def _log_to_langfuse(
    item_id: str,
    query: str,
    answer: str,
    chunks: list,
    scores: dict,
    team_id: str,
) -> None:
    try:
        obs = lf.start_observation(
            name="finsight_query",
            as_type="evaluator",
            input=query,
            output=result.answer,
            metadata={
                "team_id": team_id,
                "prompt_version": result.prompt_version,
                "model_used": result.model_used,
                "chunks_retrieved": len(chunks),
            },
        )

        for metric_name, score in [
            ("faithfulness", result.faithfulness_score),
            ("answer_relevance", answer_relevance),
            ("context_recall", context_recall),
        ]:
            if score is not None:
                obs.score_trace(name=metric_name, value=float(score))

        obs.end()
        lf.flush()
    except Exception as e:
        logger.warning("langfuse logging skipped: %s", e)


async def _run_single_item(item: dict) -> dict:
    query = item["query"]
    ground_truth = item["ground_truth"]
    ticker = item.get("ticker")
    team_id = item["team"]
    is_adversarial = "adversarial" in item.get("id", "")

    embedding = await embed(query)
    chunks = await search_dense(
        query_embedding=embedding,
        team_id=team_id,
        k=10,
        ticker=ticker,
    )

    contexts = [c.content for c in chunks]

    if not contexts:
        logger.warning(
            "no chunks retrieved for %s (ticker=%s team=%s)",
            item["id"], ticker, team_id,
        )
        return {
            "id": item["id"],
            "team": team_id,
            "intent": item.get("intent", ""),
            "question": query,
            "answer": "No relevant context found.",
            "contexts": [],
            "ground_truth": ground_truth,
            "chunks": [],
            "top_score": 0.0,
            "context_precision": None,
            "context_recall": None,
            "faithfulness": None,
            "answer_relevancy": None,
            "unsupported_claims": [],
            "chunks_retrieved": 0,
            "is_adversarial": is_adversarial,
        }

    context_text = "\n\n---\n\n".join(contexts[:5])
    answer, _, _ = await complete(
        prompt=(
            "Using only the following context, answer the question. "
            "Be specific and cite relevant details.\n\n"
            f"Context:\n{context_text}\n\n"
            f"Question: {query}\n\nAnswer:"
        ),
        system=(
            "You are a financial analyst answering questions about SEC filings. "
            "Answer using only the provided context."
        ),
        max_tokens=400,
    )

    logger.info(
        "generated answer for %s: chunks=%d top_score=%.3f",
        item["id"],
        len(chunks),
        chunks[0].score if chunks else 0.0,
    )

    scores = await score_all(
        question=query,
        answer=answer,
        contexts=contexts,
        ground_truth=ground_truth,
        complete_fn=complete,
    )

    logger.info(
        "scored %s  cp=%s cr=%s f=%s ar=%s",
        item["id"],
        f"{scores['context_precision']:.2f}" if scores["context_precision"] is not None else "None",
        f"{scores['context_recall']:.2f}" if scores["context_recall"] is not None else "None",
        f"{scores['faithfulness']:.2f}" if scores["faithfulness"] is not None else "None",
        f"{scores['answer_relevancy']:.2f}" if scores["answer_relevancy"] is not None else "None",
    )

    _log_to_langfuse(
        item_id=item["id"],
        query=query,
        answer=answer,
        chunks=chunks,
        scores=scores,
        team_id=team_id,
    )

    return {
        "id": item["id"],
        "team": team_id,
        "intent": item.get("intent", ""),
        "question": query,
        "answer": answer,
        "contexts": contexts,
        "ground_truth": ground_truth,
        "chunks": chunks,
        "top_score": chunks[0].score if chunks else 0.0,
        "chunks_retrieved": len(chunks),
        "is_adversarial": is_adversarial,
        **scores,
    }


async def run_eval() -> dict:
    setup_tracing()
    await init_pool()
    await init_qdrant()
    await init_client()

    dataset_items = load_golden_dataset()
    thresholds = load_thresholds()
    results = []

    try:
        for item in dataset_items:
            result = await _run_single_item(item)
            results.append(result)
    finally:
        await close_client()
        await close_qdrant()
        await close_pool()

    non_adversarial = [r for r in results if not r.get("is_adversarial")]

    def _avg(metric: str, subset: list) -> float:
        vals = [r[metric] for r in subset if r.get(metric) is not None]
        return round(sum(vals) / len(vals), 4) if vals else 0.0

    summary = {
        "context_precision": _avg("context_precision", non_adversarial),
        "context_recall": _avg("context_recall", non_adversarial),
        "faithfulness": _avg("faithfulness", non_adversarial),
        "answer_relevancy": _avg("answer_relevancy", non_adversarial),
        "total_queries": len(non_adversarial),
        "adversarial_excluded": len(results) - len(non_adversarial),
    }

    passed = all(
        summary[metric] >= thresholds.get(metric, 0.0)
        for metric in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]
        if metric in thresholds
    )

    return {
        "results": results,
        "summary": summary,
        "thresholds": thresholds,
        "passed": passed,
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    report = asyncio.run(run_eval())

    print("\nRAGAS-Equivalent Eval Results")
    print("=" * 50)
    summary = report["summary"]
    thresholds = report["thresholds"]

    for metric in ["context_precision", "context_recall", "faithfulness", "answer_relevancy"]:
        val = summary[metric]
        threshold = thresholds.get(metric, 0.0)
        status = "PASS" if val >= threshold else "FAIL"
        print(f"  {status}  {metric:<22} {val:.3f}  (threshold: {threshold:.2f})")

    print(f"\n  total queries evaluated: {summary['total_queries']}")
    print(f"  adversarial excluded:    {summary['adversarial_excluded']}")
    print(f"  overall: {'PASSED' if report['passed'] else 'FAILED'}")

    print("\nPer-item breakdown:")
    for r in report["results"]:
        def fmt(v):
            return f"{v:.2f}" if v is not None else "N/A"
        adv = " [adversarial]" if r.get("is_adversarial") else ""
        print(
            f"  {r['id']:<22} "
            f"cp={fmt(r['context_precision'])} "
            f"cr={fmt(r['context_recall'])} "
            f"f={fmt(r['faithfulness'])} "
            f"ar={fmt(r['answer_relevancy'])} "
            f"chunks={r['chunks_retrieved']}{adv}"
        )