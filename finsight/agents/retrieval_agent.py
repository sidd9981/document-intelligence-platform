"""
Retrieval agent.

Owns the retrieval pipeline for a single query: exact-match cache lookup,
hybrid search (dense plus sparse, fused with RRF) behind a circuit
breaker, and cross-encoder reranking to the tenant's retrieval_k. An
optional ticker restricts retrieval to one company.

Never raises. Errors are returned in RetrievalResult.errors so the
orchestrator can decide how to handle degraded results.
"""

from __future__ import annotations

import asyncio
import hashlib
import json
import logging
import time

import redis.asyncio as aioredis

from finsight.models.base import AgentError, Chunk
from finsight.models.retrieval import RetrievalResult
from finsight.models.tenant import TenantConfig
from finsight.services.circuit_breaker import CircuitBreaker
from finsight.services.reranker import rerank
from finsight.services.retrieval import run_hybrid_search
from finsight.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

FUSION_CANDIDATES = 50
CACHE_TTL_SECONDS = 3600


def _cache_key(query: str, team_id: str, ticker: str | None) -> str:
    """Exact-match cache key scoped to team and ticker filter."""
    digest = hashlib.sha256(query.encode()).hexdigest()[:16]
    return f"cache:{team_id}:{ticker or 'all'}:{digest}"


class RetrievalAgent:
    """Runs the retrieval pipeline for a single query.

    Pass the Redis client in so caching is testable without a live
    Redis instance. Built once and reused across requests.
    """

    def __init__(self, redis_client: aioredis.Redis, breaker: CircuitBreaker | None = None) -> None:
        self._redis = redis_client
        self._breaker = breaker or CircuitBreaker(name="qdrant", failure_threshold=5, recovery_timeout=30.0)

    async def retrieve(
        self,
        query: str,
        tenant_config: TenantConfig,
        trace_id: str,
        ticker: str | None = None,
    ) -> RetrievalResult:
        """Run the full retrieval pipeline for a query.

        Args:
            query: The raw query text from the user.
            tenant_config: Supplies retrieval_k and the team id for scope filtering.
            trace_id: Propagated to spans for distributed tracing.
            ticker: Optional company ticker restricting retrieval to one company.

        Returns:
            RetrievalResult with chunks, cache status, method and any errors.
            Never raises.
        """
        start = time.perf_counter()

        with tracer.start_as_current_span("retrieval_agent.retrieve") as span:
            span.set_attribute("team_id", tenant_config.team_id)
            span.set_attribute("trace_id", trace_id)
            span.set_attribute("retrieval_k", tenant_config.retrieval_k)
            if ticker:
                span.set_attribute("ticker", ticker)

            try:
                cached = await self._check_cache(query, tenant_config.team_id, ticker)
                if cached:
                    span.set_attribute("cache_hit", True)
                    return RetrievalResult(
                        chunks=cached,
                        cache_hit=True,
                        retrieval_method="cached",
                        total_tokens=sum(c.token_count for c in cached),
                        latency_ms=(time.perf_counter() - start) * 1000,
                    )

                span.set_attribute("cache_hit", False)

                fused_chunks = await self._breaker.call(
                    run_hybrid_search,
                    query=query,
                    team_id=tenant_config.team_id,
                    k=FUSION_CANDIDATES,
                    ticker=ticker,
                )

                if not fused_chunks:
                    return RetrievalResult(
                        chunks=[],
                        cache_hit=False,
                        retrieval_method="hybrid",
                        total_tokens=0,
                        latency_ms=(time.perf_counter() - start) * 1000,
                        errors=[AgentError(
                            agent="retrieval_agent",
                            error_type="empty_result",
                            message="hybrid search returned no results",
                        )],
                    )

                reranked = await asyncio.to_thread(
                    rerank,
                    query=query,
                    chunks=fused_chunks,
                    top_k=tenant_config.retrieval_k,
                )

                span.set_attribute("chunks.returned", len(reranked))
                return RetrievalResult(
                    chunks=reranked,
                    cache_hit=False,
                    retrieval_method="hybrid",
                    total_tokens=sum(c.token_count for c in reranked),
                    latency_ms=(time.perf_counter() - start) * 1000,
                )

            except Exception as e:
                logger.error("retrieval agent error: %s", e)
                return RetrievalResult(
                    chunks=[],
                    cache_hit=False,
                    retrieval_method="hybrid",
                    total_tokens=0,
                    latency_ms=(time.perf_counter() - start) * 1000,
                    errors=[AgentError(
                        agent="retrieval_agent",
                        error_type="service_unavailable",
                        message=str(e),
                    )],
                )

    async def _check_cache(self, query: str, team_id: str, ticker: str | None) -> list[Chunk] | None:
        """Return cached chunks for an identical query, or None on a miss or error."""
        try:
            cached = await self._redis.get(_cache_key(query, team_id, ticker))
            if cached:
                return [Chunk(**c) for c in json.loads(cached)]
        except Exception as e:
            logger.warning("cache lookup failed: %s", e)
        return None

    async def write_cache(
        self,
        query: str,
        team_id: str,
        chunks: list[Chunk],
        ticker: str | None = None,
        ttl_seconds: int = CACHE_TTL_SECONDS,
    ) -> None:
        """Cache the retrieval result for this exact query, team and ticker filter.

        Called by the orchestrator after a successful retrieval so an
        identical query can skip the retrieval pipeline.
        """
        try:
            serialized = json.dumps([c.model_dump(mode="json") for c in chunks])
            await self._redis.set(_cache_key(query, team_id, ticker), serialized, ex=ttl_seconds)
        except Exception as e:
            logger.warning("cache write failed: %s", e)