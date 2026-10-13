"""
Offline evaluation of the full FinSight pipeline.

Runs every golden query through the production orchestrator, then scores
the final answer against the exact chunks synthesis received with the
ragas Faithfulness, LLMContextRecall, LLMContextPrecisionWithReference
and ResponseRelevancy metrics. A separate judge model at temperature 0
grades the answers so the generator never scores itself. Adversarial
queries are scored separately as an abstention rate.

Full run, for numbers you report:
    python -m eval.offline_eval --runs 3

Quick run on a subset, for checking a fix. Marked partial in the report:
    python -m eval.offline_eval --runs 1 --limit 3
    python -m eval.offline_eval --runs 1 --ids aapl-risk-001 msft-risk-001
"""

from __future__ import annotations

import argparse
import asyncio
import functools
import json
import logging
import re
import statistics
import time
from datetime import datetime
from pathlib import Path

import yaml
from neo4j import AsyncGraphDatabase

from eval.ragas_metrics import RagasScorer
from finsight.agents.graph_agent import GraphAgent
from finsight.agents.orchestrator import Orchestrator
from finsight.agents.retrieval_agent import RetrievalAgent
from finsight.agents.synthesis_agent import SynthesisAgent
from finsight.gateway.db import close_pool, get_pool, init_pool
from finsight.models.tenant import TenantConfig
from finsight.services.circuit_breaker import CircuitBreaker
from finsight.services.llm import close_client, complete, init_client
from finsight.services.reranker import close_reranker, init_reranker
from finsight.services.sparse_encoder import close_encoder, init_encoder
from finsight.services.vector_store import close_client as close_qdrant
from finsight.services.vector_store import init_client as init_qdrant
from finsight.telemetry.tracing import setup_tracing

logger = logging.getLogger(__name__)

GOLDEN_DATASET_PATH = Path(__file__).parent / "golden_dataset" / "queries.json"
THRESHOLDS_PATH = Path(__file__).parent / "thresholds.yaml"
RESULTS_DIR = Path(__file__).parent / "results"

METRICS = ("context_precision", "context_recall", "faithfulness", "answer_relevancy")
DEFAULT_JUDGE_MODEL = "qwen2.5:7b"
DEFAULT_THRESHOLDS = {
    "context_precision": 0.60,
    "context_recall": 0.60,
    "faithfulness": 0.75,
    "answer_relevancy": 0.60,
}
TENANT_FIELDS = (
    "team_id",
    "daily_token_budget",
    "max_context_tokens",
    "max_output_tokens",
    "requests_per_minute",
    "priority",
    "allowed_models",
    "retrieval_k",
    "data_scopes",
)
NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "changeme")

_ABSTENTION_PROMPT = """The question below asks for information that is not in the system's documents.

Question: {question}

Answer: {answer}

Did the answer decline or state that the information is not available, without inventing specific facts, figures, or events?

Respond with exactly one line in this format:
ABSTAINED: <yes or no>"""

_ABSTAINED_LINE = re.compile(r"ABSTAINED:\s*(yes|no)\b", re.IGNORECASE)


class _NoCache:
    """Stand-in Redis client that never hits, so every run retrieves fresh."""

    async def get(self, key: str) -> None:
        return None

    async def set(self, key: str, value: bytes, ex: int | None = None) -> None:
        return None


def load_thresholds() -> dict:
    """Load gate thresholds from thresholds.yaml, falling back to defaults."""
    if THRESHOLDS_PATH.exists():
        with open(THRESHOLDS_PATH) as f:
            return yaml.safe_load(f)
    return dict(DEFAULT_THRESHOLDS)


def load_golden_dataset() -> list[dict]:
    """Load the golden query set."""
    with open(GOLDEN_DATASET_PATH) as f:
        return json.load(f)


def select_items(items: list[dict], limit: int | None, ids: list[str] | None) -> list[dict]:
    """Narrow the golden set to the requested ids, then to the first limit items.

    Raises:
        ValueError: If a requested id is not in the golden set, or nothing is left.
    """
    selected = items
    if ids:
        known = {item["id"] for item in items}
        unknown = [i for i in ids if i not in known]
        if unknown:
            raise ValueError(f"unknown golden ids: {unknown}")
        selected = [item for item in items if item["id"] in ids]
    if limit is not None:
        selected = selected[:limit]
    if not selected:
        raise ValueError("no golden items selected")
    return selected


async def load_tenant_configs(team_ids: set[str]) -> dict[str, TenantConfig]:
    """Load each team's TenantConfig from Postgres, failing loudly on a missing team."""
    configs: dict[str, TenantConfig] = {}
    async with get_pool().acquire() as conn:
        for team_id in team_ids:
            row = await conn.fetchrow(
                "SELECT * FROM tenant_configs WHERE team_id = $1",
                team_id,
            )
            if row is None:
                raise RuntimeError(
                    f"team {team_id} not in tenant_configs, run scripts/seed_tenants.py"
                )
            configs[team_id] = TenantConfig(**{name: row[name] for name in TENANT_FIELDS})
    return configs


def build_orchestrator(neo4j_driver) -> Orchestrator:
    """Wire the orchestrator the same way the gateway lifespan does, minus the cache."""
    return Orchestrator(
        retrieval_agent=RetrievalAgent(
            redis_client=_NoCache(),
            breaker=CircuitBreaker(name="qdrant", failure_threshold=5, recovery_timeout=30.0),
        ),
        graph_agent=GraphAgent(
            driver=neo4j_driver,
            breaker=CircuitBreaker(name="neo4j", failure_threshold=5, recovery_timeout=30.0),
        ),
        synthesis_agent=SynthesisAgent(),
    )


async def judge_abstention(question: str, answer: str, judge) -> bool | None:
    """Ask the judge whether the answer declined instead of inventing facts."""
    try:
        response, _, _ = await judge(
            prompt=_ABSTENTION_PROMPT.format(question=question, answer=answer),
            system="You are a strict evaluator. Follow the format exactly.",
            max_tokens=10,
        )
    except Exception as e:
        logger.warning("abstention judge call failed: %s", e)
        return None
    match = _ABSTAINED_LINE.search(response.replace("*", ""))
    return match.group(1).lower() == "yes" if match else None


async def evaluate_item(
    item: dict,
    orchestrator: Orchestrator,
    tenant_config: TenantConfig,
    judge,
    scorer: RagasScorer,
) -> dict:
    """Run one golden query through the full pipeline and score the result."""
    start = time.perf_counter()
    state = await orchestrator.run_state(query=item["query"], tenant_config=tenant_config)
    latency_ms = (time.perf_counter() - start) * 1000

    retrieval = state.get("retrieval_result")
    synthesis = state.get("synthesis_result")
    final = state.get("final_response")
    chunks = retrieval.chunks if retrieval else []

    if synthesis and synthesis.answer:
        answer = synthesis.answer
    else:
        answer = final.answer if final else ""

    expected_ticker = item.get("ticker")
    ticker_match_rate = (
        sum(1 for c in chunks if c.metadata.ticker == expected_ticker) / len(chunks)
        if chunks and expected_ticker
        else None
    )

    record = {
        "id": item["id"],
        "team": item["team"],
        "intent": item.get("intent", ""),
        "question": item["query"],
        "answer": answer,
        "chunk_ids": [c.chunk_id for c in chunks],
        "chunk_tickers": [c.metadata.ticker for c in chunks],
        "chunks_used": len(chunks),
        "ticker_match_rate": ticker_match_rate,
        "harness_faithfulness": synthesis.faithfulness_score if synthesis else None,
        "errors": [e.message for e in state.get("errors", [])],
        "latency_ms": round(latency_ms, 1),
        "is_adversarial": item.get("intent") == "adversarial",
    }

    if record["is_adversarial"]:
        record["abstained"] = await judge_abstention(item["query"], answer, judge)
        return record

    ranked = sorted(chunks, key=lambda c: c.score, reverse=True)
    scores = await scorer.score(
        question=item["query"],
        answer=answer,
        contexts=[c.content for c in ranked],
        ground_truth=item["ground_truth"],
    )
    record.update(scores)
    return record


def summarize_run(records: list[dict]) -> dict:
    """Average each metric over the non-adversarial items the judge actually scored."""
    scored = [r for r in records if not r["is_adversarial"]]
    adversarial = [r for r in records if r["is_adversarial"]]
    summary: dict = {"items": len(scored)}

    for metric in METRICS:
        values = [r[metric] for r in scored if r.get(metric) is not None]
        summary[metric] = sum(values) / len(values) if values else None
        summary[f"{metric}_n"] = len(values)

    judged = [r["abstained"] for r in adversarial if r.get("abstained") is not None]
    summary["abstention_rate"] = sum(judged) / len(judged) if judged else None
    summary["abstention_n"] = len(judged)

    matches = [r["ticker_match_rate"] for r in scored if r["ticker_match_rate"] is not None]
    summary["ticker_match_rate"] = sum(matches) / len(matches) if matches else None
    return summary


def aggregate(run_summaries: list[dict]) -> dict:
    """Mean and spread of each run-level metric across runs."""
    out: dict = {"runs": len(run_summaries), "items": run_summaries[0]["items"]}
    for key in (*METRICS, "abstention_rate", "ticker_match_rate"):
        values = [s[key] for s in run_summaries if s[key] is not None]
        out[key] = round(statistics.mean(values), 4) if values else None
        out[f"{key}_std"] = round(statistics.stdev(values), 4) if len(values) > 1 else 0.0
    for metric in METRICS:
        out[f"{metric}_n"] = [s[f"{metric}_n"] for s in run_summaries]
    return out


def _fmt(value) -> str:
    return f"{value:.2f}" if isinstance(value, (int, float)) else "N/A"


async def run_eval(
    runs: int = 3,
    judge_model: str = DEFAULT_JUDGE_MODEL,
    limit: int | None = None,
    ids: list[str] | None = None,
) -> dict:
    """Evaluate the golden set, or a subset of it, and gate on the mean scores.

    Args:
        runs: Number of passes over the selected items.
        judge_model: Ollama model used by ragas and the abstention check.
        limit: Evaluate only the first limit items after id filtering.
        ids: Evaluate only these golden ids.

    Returns:
        Report dict. partial is True when limit or ids narrowed the set,
        in which case the scores are not comparable to a full run.
    """
    all_items = load_golden_dataset()
    items = select_items(all_items, limit, ids)
    partial = len(items) < len(all_items)

    setup_tracing()
    await init_pool()
    await init_qdrant()
    await init_client()
    init_encoder()
    init_reranker()
    neo4j_driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    thresholds = load_thresholds()
    judge = functools.partial(complete, model=judge_model, temperature=0.0)
    scorer = RagasScorer(judge_model)
    all_records: list[dict] = []
    run_summaries: list[dict] = []

    try:
        configs = await load_tenant_configs({item["team"] for item in items})
        orchestrator = build_orchestrator(neo4j_driver)

        for run in range(runs):
            records = []
            for item in items:
                record = await evaluate_item(item, orchestrator, configs[item["team"]], judge, scorer)
                record["run"] = run
                records.append(record)
                logger.info(
                    "run=%d %s cp=%s cr=%s f=%s ar=%s abst=%s tickers=%s chunks=%d",
                    run,
                    record["id"],
                    _fmt(record.get("context_precision")),
                    _fmt(record.get("context_recall")),
                    _fmt(record.get("faithfulness")),
                    _fmt(record.get("answer_relevancy")),
                    record.get("abstained", "-"),
                    _fmt(record["ticker_match_rate"]),
                    record["chunks_used"],
                )
            run_summaries.append(summarize_run(records))
            all_records.extend(records)
    finally:
        await neo4j_driver.close()
        close_reranker()
        close_encoder()
        await close_client()
        await close_qdrant()
        await close_pool()

    summary = aggregate(run_summaries)
    passed = all(
        summary[metric] is not None and summary[metric] >= thresholds.get(metric, 0.0)
        for metric in METRICS
        if metric in thresholds
    )

    return {
        "judge_model": judge_model,
        "partial": partial,
        "item_ids": [item["id"] for item in items],
        "summary": summary,
        "run_summaries": run_summaries,
        "thresholds": thresholds,
        "passed": passed,
        "results": all_records,
    }


def main() -> None:
    """Run the eval from the command line, print a summary, and save the full report."""
    parser = argparse.ArgumentParser(description="FinSight offline eval")
    parser.add_argument("--runs", type=int, default=3)
    parser.add_argument("--judge-model", default=DEFAULT_JUDGE_MODEL)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--ids", nargs="+", default=None)
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    report = asyncio.run(
        run_eval(runs=args.runs, judge_model=args.judge_model, limit=args.limit, ids=args.ids)
    )

    RESULTS_DIR.mkdir(exist_ok=True)
    prefix = "partial" if report["partial"] else "eval"
    out_path = RESULTS_DIR / f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    with open(out_path, "w") as f:
        json.dump(report, f, indent=2, default=str)

    summary = report["summary"]
    print(f"\nFull pipeline eval, judge={report['judge_model']}, runs={summary['runs']}, items={summary['items']}")
    if report["partial"]:
        print(f"  PARTIAL RUN on {report['item_ids']}, not comparable to a full run")
    for metric in METRICS:
        value = summary[metric]
        threshold = report["thresholds"].get(metric, 0.0)
        status = "PASS" if value is not None and value >= threshold else "FAIL"
        print(
            f"  {status}  {metric:<20} {_fmt(value)} +/- {_fmt(summary[f'{metric}_std'])}"
            f"  n per run={summary[f'{metric}_n']}  threshold={threshold:.2f}"
        )
    print(f"  abstention rate (adversarial)  {_fmt(summary['abstention_rate'])}")
    print(f"  ticker match rate              {_fmt(summary['ticker_match_rate'])}")
    print(f"  overall: {'PASSED' if report['passed'] else 'FAILED'}")
    print(f"  full report: {out_path}")


if __name__ == "__main__":
    main()