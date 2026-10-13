"""
RAGAS metrics computed with the ragas package, judged by a local Ollama model.

Uses the ragas Faithfulness, LLMContextRecall,
LLMContextPrecisionWithReference and ResponseRelevancy metrics. The judge
and the embedder are reached through Ollama's OpenAI-compatible API via
LangChain, so the eval runs locally at no cost.

ragas expects the judge to return JSON matching its schemas. Local models
often wrap JSON in prose or get cut off mid-object, so the judge runs in
Ollama's JSON mode with an explicit output budget large enough for the
longest ragas responses.

The four metrics are independent, so they run concurrently for each query.
Set OLLAMA_NUM_PARALLEL above 1 for the concurrency to reach the model.

When ragas still cannot parse the judge output, the metric comes back as
NaN or raises. Both become None, so the caller excludes the item from
averages and reports how many items were actually scored.
"""

from __future__ import annotations

import asyncio
import logging
import math

from langchain_openai import ChatOpenAI, OpenAIEmbeddings
from ragas.dataset_schema import SingleTurnSample
from ragas.embeddings import LangchainEmbeddingsWrapper
from ragas.llms import LangchainLLMWrapper
from ragas.metrics import (
    Faithfulness,
    LLMContextPrecisionWithReference,
    LLMContextRecall,
    ResponseRelevancy,
)
from ragas.run_config import RunConfig

from finsight.config.settings import settings

logger = logging.getLogger(__name__)

METRIC_TIMEOUT_SECONDS = 600
METRIC_MAX_RETRIES = 2
JUDGE_MAX_OUTPUT_TOKENS = 2048


def _to_score(value) -> float | None:
    """Convert a ragas result to a float, mapping None and NaN to None."""
    if value is None:
        return None
    value = float(value)
    return None if math.isnan(value) else value


class RagasScorer:
    """Scores one query at a time with the four ragas RAG metrics.

    Build once per eval run. The judge runs at temperature 0 so repeated
    runs differ only through the pipeline being evaluated.
    """

    def __init__(self, judge_model: str) -> None:
        run_config = RunConfig(timeout=METRIC_TIMEOUT_SECONDS, max_retries=METRIC_MAX_RETRIES)

        llm = LangchainLLMWrapper(
            ChatOpenAI(
                model=judge_model,
                base_url=settings.ollama.base_url,
                api_key="ollama",
                temperature=0.0,
                max_tokens=JUDGE_MAX_OUTPUT_TOKENS,
                model_kwargs={"response_format": {"type": "json_object"}},
            ),
            run_config=run_config,
        )
        embeddings = LangchainEmbeddingsWrapper(
            OpenAIEmbeddings(
                model=settings.ollama.embedding_model,
                base_url=settings.ollama.base_url,
                api_key="ollama",
                check_embedding_ctx_length=False,
            ),
            run_config=run_config,
        )

        self._metrics = {
            "context_precision": LLMContextPrecisionWithReference(llm=llm),
            "context_recall": LLMContextRecall(llm=llm),
            "faithfulness": Faithfulness(llm=llm),
            "answer_relevancy": ResponseRelevancy(llm=llm, embeddings=embeddings),
        }

    async def _run_metric(self, name: str, metric, sample: SingleTurnSample) -> tuple[str, float | None]:
        """Score one metric, returning None instead of raising on failure."""
        try:
            value = await metric.single_turn_ascore(sample)
        except Exception as e:
            logger.warning("ragas %s failed: %s", name, e)
            value = None
        return name, _to_score(value)

    async def score(
        self,
        question: str,
        answer: str,
        contexts: list[str],
        ground_truth: str,
    ) -> dict:
        """Run all four metrics on one query concurrently.

        Args:
            question: The user query.
            answer: The generated answer.
            contexts: Chunk texts the generator received, in rank order, best first.
            ground_truth: Reference answer from the golden dataset.

        Returns:
            Dict of metric name to score, None where ragas failed or returned NaN.
        """
        sample = SingleTurnSample(
            user_input=question,
            response=answer,
            retrieved_contexts=contexts,
            reference=ground_truth,
        )
        results = await asyncio.gather(
            *(self._run_metric(name, metric, sample) for name, metric in self._metrics.items())
        )
        return dict(results)