"""
Retrieval utilities: RRF fusion and hybrid search orchestration.

The retrieval agent calls run_hybrid_search(), which fires dense and
sparse search in parallel, fuses the results with RRF, and returns a
single ranked list ready for the reranker. An optional ticker restricts
both searches to one company's chunks.
"""

from __future__ import annotations

import asyncio
import logging

from finsight.models.base import Chunk
from finsight.services.llm import embed
from finsight.services.sparse_encoder import encode_sparse
from finsight.services.vector_store import search_dense, search_sparse
from finsight.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

RRF_K = 60


def reciprocal_rank_fusion(result_lists: list[list[Chunk]]) -> list[Chunk]:
    """Merge multiple ranked chunk lists into one using RRF.

    Each list contributes 1/(rank + RRF_K) to a chunk's total score, so
    chunks appearing in several lists accumulate score from each. Only
    rank order matters, the original scores are ignored, which is what
    lets dense and sparse scores on different scales fuse cleanly.

    Args:
        result_lists: One list per retrieval method, each sorted by
                      descending relevance.

    Returns:
        Single deduplicated list sorted by fused score, highest first,
        with each chunk's score set to its fused score.
    """
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, Chunk] = {}

    with tracer.start_as_current_span("retrieval.rrf_fusion") as span:
        span.set_attribute("num_lists", len(result_lists))

        for result_list in result_lists:
            for rank, chunk in enumerate(result_list):
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + 1.0 / (rank + RRF_K)
                chunks_by_id[chunk.chunk_id] = chunk

        fused = [
            chunks_by_id[chunk_id].model_copy(update={"score": score})
            for chunk_id, score in sorted(scores.items(), key=lambda item: item[1], reverse=True)
        ]

        span.set_attribute("chunks.input", sum(len(lst) for lst in result_lists))
        span.set_attribute("chunks.output", len(fused))
        return fused


async def run_hybrid_search(
    query: str,
    team_id: str,
    k: int,
    ticker: str | None = None,
) -> list[Chunk]:
    """Run dense and sparse search in parallel and fuse with RRF.

    Falls back to dense-only if sparse encoding or search raises, and
    logs the fallback so it is visible.

    Args:
        query: The raw query text.
        team_id: Used for scope filtering on both searches.
        k: Number of results to request from each method.
        ticker: Optional company ticker restricting both searches.

    Returns:
        Fused and ranked list of chunks, at most k.
    """
    with tracer.start_as_current_span("retrieval.hybrid_search") as span:
        span.set_attribute("team_id", team_id)
        span.set_attribute("k", k)
        if ticker:
            span.set_attribute("ticker", ticker)

        dense_embedding = await embed(query)

        try:
            sparse_vector = await asyncio.to_thread(encode_sparse, query)
            dense_results, sparse_results = await asyncio.gather(
                search_dense(dense_embedding, team_id, k, ticker=ticker),
                search_sparse(sparse_vector, team_id, k, ticker=ticker),
            )
            span.set_attribute("dense.results", len(dense_results))
            span.set_attribute("sparse.results", len(sparse_results))
            fused = reciprocal_rank_fusion([dense_results, sparse_results])
        except Exception as e:
            logger.warning("sparse search failed, falling back to dense-only: %s", e)
            span.set_attribute("fallback", "dense_only")
            fused = await search_dense(dense_embedding, team_id, k, ticker=ticker)

        return fused[:k]