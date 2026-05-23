"""
Offline eval runner.

Loads the golden dataset, runs retrieval for each query, and computes
context precision and recall. Used as a CI gate — fails if any metric
drops below the threshold defined in eval/thresholds.yaml.

Run manually: python eval/offline_eval.py
Run in CI: pytest tests/eval/test_ragas_gate.py
"""

from __future__ import annotations

import asyncio
import json
import logging
from pathlib import Path

import yaml

from finsight.gateway.db import init_pool, close_pool
from finsight.services.llm import init_client, close_client, embed
from finsight.services.vector_store import (
    init_client as init_qdrant,
    close_client as close_qdrant,
    search_dense,
)
from finsight.telemetry.tracing import setup_tracing

logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset" / "queries.json"
THRESHOLDS_PATH = Path(__file__).parent / "thresholds.yaml"


def load_thresholds() -> dict:
    if THRESHOLDS_PATH.exists():
        with open(THRESHOLDS_PATH) as f:
            return yaml.safe_load(f)
    return {
        "context_recall": 0.70,
        "answer_keyword_match": 0.60,
    }


def load_golden_dataset() -> list[dict]:
    with open(GOLDEN_DATASET_PATH) as f:
        return json.load(f)


def score_keyword_match(answer: str, expected_keywords: list[str]) -> float:
    """Check what fraction of expected keywords appear in the answer.

    This is a lightweight proxy for answer quality when we don't have
    a full RAGAS setup running. Phase 6 replaces this with real RAGAS
    metrics via Langfuse.
    """
    if not expected_keywords:
        return 1.0
    answer_lower = answer.lower()
    matched = sum(1 for kw in expected_keywords if kw.lower() in answer_lower)
    return matched / len(expected_keywords)


def score_section_recall(chunks: list, expected_sections: list[str]) -> float:
    """Check what fraction of expected sections appear in retrieved chunks."""
    if not expected_sections:
        return 1.0
    retrieved_sections = {c.metadata.section for c in chunks}
    matched = sum(
        1 for s in expected_sections
        if any(s.lower() in rs.lower() for rs in retrieved_sections)
    )
    return matched / len(expected_sections)


async def run_eval() -> dict:
    """Run the full offline eval and return metric scores."""
    setup_tracing()
    await init_pool()
    await init_qdrant()
    await init_client()

    dataset = load_golden_dataset()
    thresholds = load_thresholds()

    results = []

    try:
        for item in dataset:
            query = item["query"]
            expected_keywords = item.get("expected_answer_contains", [])
            expected_sections = item.get("expected_sections", [])

            embedding = await embed(query)
            chunks = await search_dense(embedding, team_id=item["team"], k=5)

            from finsight.services.llm import complete
            context = "\n\n".join(c.content[:300] for c in chunks)
            prompt = f"Context:\n{context}\n\nQuestion: {query}"
            answer, _, _ = await complete(
                prompt=prompt,
                system="You are a financial analyst. Answer using only the provided context.",
                max_tokens=300,
            )

            keyword_score = score_keyword_match(answer, expected_keywords)
            section_score = score_section_recall(chunks, expected_sections)

            results.append({
                "id": item["id"],
                "team": item["team"],
                "intent": item["intent"],
                "keyword_match": keyword_score,
                "section_recall": section_score,
                "chunks_retrieved": len(chunks),
                "top_score": chunks[0].score if chunks else 0.0,
            })

            logger.info(
                "eval %s keyword=%.2f section=%.2f chunks=%d",
                item["id"],
                keyword_score,
                section_score,
                len(chunks),
            )

    finally:
        await close_client()
        await close_qdrant()
        await close_pool()

    avg_keyword = sum(r["keyword_match"] for r in results) / len(results)
    avg_section = sum(r["section_recall"] for r in results) / len(results)

    return {
        "results": results,
        "summary": {
            "avg_keyword_match": avg_keyword,
            "avg_section_recall": avg_section,
            "total_queries": len(results),
        },
        "thresholds": thresholds,
        "passed": (
            avg_keyword >= thresholds["answer_keyword_match"]
            and avg_section >= thresholds["context_recall"]
        ),
    }


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    report = asyncio.run(run_eval())
    print(json.dumps(report["summary"], indent=2))
    print(f"\npassed: {report['passed']}")
    for r in report["results"]:
        status = "ok" if r["keyword_match"] >= 0.5 else "FAIL"
        print(f"  {status} {r['id']} keyword={r['keyword_match']:.2f} section={r['section_recall']:.2f}")