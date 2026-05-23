"""
Retrieval utilities: RRF fusion and hybrid search orchestration.

The retrieval agent calls run_hybrid_search() which fires dense and
sparse search in parallel, fuses the results with RRF, and returns
a single ranked list ready for the reranker.
"""

from __future__ import annotations

from finsight.models.base import Chunk
from finsight.services.llm import embed
from finsight.services.sparse_encoder import encode_sparse
from finsight.services.vector_store import search_dense, search_sparse
from finsight.telemetry.tracing import get_tracer

tracer = get_tracer(__name__)

RRF_K = 60


def reciprocal_rank_fusion(
    result_lists: list[list[Chunk]],
) -> list[Chunk]:
    """Merge multiple ranked chunk lists into one using RRF.

    Each list contributes 1/(rank + RRF_K) to a chunk's total score.
    Chunks appearing in multiple lists accumulate score from each.
    The final list is sorted by descending fused score.

    Args:
        result_lists: One list per retrieval method, each sorted by
                      descending relevance. Order within each list
                      is what matters — the original scores are ignored.

    Returns:
        Single deduplicated list sorted by fused score descending.
    """
    scores: dict[str, float] = {}
    chunks_by_id: dict[str, Chunk] = {}

    with tracer.start_as_current_span("retrieval.rrf_fusion") as span:
        span.set_attribute("num_lists", len(result_lists))

        for result_list in result_lists:
            for rank, chunk in enumerate(result_list):
                score = 1.0 / (rank + RRF_K)
                scores[chunk.chunk_id] = scores.get(chunk.chunk_id, 0.0) + score
                chunks_by_id[chunk.chunk_id] = chunk

        fused = sorted(
            chunks_by_id.values(),
            key=lambda c: scores[c.chunk_id],
            reverse=True,
        )

        for chunk in fused:
            fused = [
                chunk.model_copy(update={"score": scores[chunk.chunk_id]})
                for chunk in fused
            ]

        span.set_attribute("chunks.input", sum(len(l) for l in result_lists))
        span.set_attribute("chunks.output", len(fused))

        return fused


async def run_hybrid_search(
    query: str,
    team_id: str,
    k: int,
) -> list[Chunk]:
    """Run dense and sparse search in parallel and fuse with RRF.

    Args:
        query: The raw query text.
        team_id: Used for scope filtering on both searches.
        k: Number of results to request from each method.
           The fused list will have at most k unique chunks
           but may have fewer if there's significant overlap.

    Returns:
        Fused and ranked list of chunks, up to k results.
    """
    import asyncio

    with tracer.start_as_current_span("retrieval.hybrid_search") as span:
        span.set_attribute("team_id", team_id)
        span.set_attribute("k", k)

        dense_embedding, sparse_vector = await asyncio.gather(
            embed(query),
            asyncio.get_event_loop().run_in_executor(None, encode_sparse, query),
        )

        dense_results, sparse_results = await asyncio.gather(
            search_dense(dense_embedding, team_id, k),
            search_sparse(sparse_vector, team_id, k),
        )

        span.set_attribute("dense.results", len(dense_results))
        span.set_attribute("sparse.results", len(sparse_results))

        fused = reciprocal_rank_fusion([dense_results, sparse_results])
        return fused[:k]