"""
Retrieval agent.

Owns the full retrieval pipeline for a single query: cache lookup,
hybrid search (dense + sparse + RRF), cross-encoder reranking, and
returning a typed RetrievalResult to the orchestrator.

Never raises. Errors are returned in RetrievalResult.errors so the
orchestrator can decide how to handle degraded results.
"""

from __future__ import annotations

import logging

import redis.asyncio as aioredis

from finsight.config.settings import settings
from finsight.models.base import AgentError, Chunk
from finsight.models.retrieval import RetrievalResult
from finsight.models.tenant import TenantConfig
from finsight.services.reranker import rerank
from finsight.services.retrieval import run_hybrid_search
from finsight.telemetry.tracing import get_tracer
import time
import json
import hashlib

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

CACHE_SIMILARITY_THRESHOLD = 0.95

from finsight.services.circuit_breaker import CircuitBreaker, CircuitOpenError

class RetrievalAgent:
    """Runs the retrieval pipeline for a single query.

    Pass the Redis client in so caching is testable without a live
    Redis instance. In production the orchestrator builds this once
    and reuses it across requests.
    """

    def __init__(self, redis_client: aioredis.Redis, breaker: CircuitBreaker | None = None) -> None:
        self._redis = redis_client
        self._breaker = breaker or CircuitBreaker(name="qdrant", failure_threshold=5, recovery_timeout=30.0)

    async def retrieve(
        self,
        query: str,
        tenant_config: TenantConfig,
        trace_id: str,
    ) -> RetrievalResult:
        """Run the full retrieval pipeline for a query.

        Args:
            query: The raw query text from the user.
            tenant_config: Controls retrieval_k, data_scopes, and
                           which collections are accessible.
            trace_id: Propagated to all spans for distributed tracing.

        Returns:
            RetrievalResult with chunks, cache status, method, and
            any errors that occurred. Never raises.
        """
        start = time.perf_counter()

        with tracer.start_as_current_span("retrieval_agent.retrieve") as span:
            span.set_attribute("team_id", tenant_config.team_id)
            span.set_attribute("trace_id", trace_id)
            span.set_attribute("retrieval_k", tenant_config.retrieval_k)

            try:
                cached = await self._check_cache(query, tenant_config.team_id)
                if cached:
                    span.set_attribute("cache_hit", True)
                    latency_ms = (time.perf_counter() - start) * 1000
                    return RetrievalResult(
                        chunks=cached,
                        cache_hit=True,
                        retrieval_method="cached",
                        total_tokens=sum(c.token_count for c in cached),
                        latency_ms=latency_ms,
                    )

                span.set_attribute("cache_hit", False)

                fused_chunks = await self._breaker.call(
                    run_hybrid_search,
                    query=query,
                    team_id=tenant_config.team_id,
                    k=50,
                )

                if not fused_chunks:
                    latency_ms = (time.perf_counter() - start) * 1000
                    return RetrievalResult(
                        chunks=[],
                        cache_hit=False,
                        retrieval_method="hybrid",
                        total_tokens=0,
                        latency_ms=latency_ms,
                        errors=[AgentError(
                            agent="retrieval_agent",
                            error_type="empty_result",
                            message="hybrid search returned no results",
                        )],
                    )

                reranked = rerank(
                    query=query,
                    chunks=fused_chunks,
                    top_k=tenant_config.retrieval_k,
                )

                span.set_attribute("chunks.returned", len(reranked))
                latency_ms = (time.perf_counter() - start) * 1000

                return RetrievalResult(
                    chunks=reranked,
                    cache_hit=False,
                    retrieval_method="hybrid",
                    total_tokens=sum(c.token_count for c in reranked),
                    latency_ms=latency_ms,
                )

            except Exception as e:
                logger.error("retrieval agent error: %s", e)
                latency_ms = (time.perf_counter() - start) * 1000
                return RetrievalResult(
                    chunks=[],
                    cache_hit=False,
                    retrieval_method="hybrid",
                    total_tokens=0,
                    latency_ms=latency_ms,
                    errors=[AgentError(
                        agent="retrieval_agent",
                        error_type="service_unavailable",
                        message=str(e),
                    )],
                )

    async def _check_cache(self, query: str, team_id: str) -> list[Chunk] | None:

        cache_key = f"cache:{team_id}:{hashlib.sha256(query.encode()).hexdigest()[:16]}"

        try:
            cached = await self._redis.get(cache_key)
            if cached:
                data = json.loads(cached)
                return [Chunk(**c) for c in data]
        except Exception as e:
            logger.warning("cache lookup failed: %s", e)

        return None

    async def write_cache(
        self,
        query: str,
        team_id: str,
        chunks: list[Chunk],
        ttl_seconds: int = 3600,
    ) -> None:
        """Cache the retrieval result for this query.

        Called by the orchestrator after a successful retrieval so
        subsequent similar queries can skip the retrieval pipeline.
        """
        import json
        import hashlib

        cache_key = f"cache:{team_id}:{hashlib.sha256(query.encode()).hexdigest()[:16]}"

        try:
            serialized = json.dumps([c.model_dump(mode="json") for c in chunks])
            await self._redis.set(cache_key, serialized, ex=ttl_seconds)
        except Exception as e:
            logger.warning("cache write failed: %s", e)