"""
Prometheus metrics registry.

All metrics are defined here and imported wherever they need to be
recorded. Nothing outside this file calls prometheus_client directly.

Metrics are initialized at import time. The /metrics endpoint in the
gateway exposes them to Prometheus for scraping.

Naming convention: finsight_<component>_<measurement>_<unit>
"""

from __future__ import annotations

from prometheus_client import Counter, Gauge, Histogram, CollectorRegistry, REGISTRY

LATENCY_BUCKETS = (0.1, 0.25, 0.5, 1.0, 2.0, 4.0, 8.0, 16.0, 30.0, 60.0)

query_latency = Histogram(
    "finsight_query_latency_seconds",
    "End-to-end query latency from gateway receipt to response sent.",
    labelnames=["team_id", "intent", "status"],
    buckets=LATENCY_BUCKETS,
)

tokens_used_total = Counter(
    "finsight_tokens_used_total",
    "Total LLM tokens consumed.",
    labelnames=["team_id", "model"],
)

cache_hits_total = Counter(
    "finsight_cache_hits_total",
    "Number of queries served from semantic cache.",
    labelnames=["team_id"],
)

cache_misses_total = Counter(
    "finsight_cache_misses_total",
    "Number of queries that missed the semantic cache.",
    labelnames=["team_id"],
)

retrieval_score = Histogram(
    "finsight_retrieval_score",
    "Top retrieval score per query. Low scores indicate poor context quality.",
    labelnames=["team_id", "method"],
    buckets=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
)

faithfulness_score = Gauge(
    "finsight_faithfulness_score",
    "Rolling faithfulness score from the LLM-as-judge pipeline.",
    labelnames=["team_id"],
)

budget_utilization_pct = Gauge(
    "finsight_budget_utilization_pct",
    "Percentage of daily token budget consumed. Resets at midnight.",
    labelnames=["team_id"],
)

ingestion_lag_seconds = Gauge(
    "finsight_ingestion_lag_seconds",
    "Seconds between document publication and ingestion completion.",
    labelnames=["source"],
)

dlq_depth = Gauge(
    "finsight_dlq_depth",
    "Number of messages in the dead letter queue awaiting review.",
    labelnames=["stream"],
)

mcp_latency = Histogram(
    "finsight_mcp_latency_seconds",
    "Latency of MCP tool invocations.",
    labelnames=["server", "tool"],
    buckets=LATENCY_BUCKETS,
)

circuit_breaker_opens_total = Counter(
    "finsight_circuit_breaker_opens_total",
    "Number of times a circuit breaker has opened.",
    labelnames=["name"],
)

entity_resolution_confidence = Histogram(
    "finsight_entity_resolution_confidence",
    "Confidence score from the fuzzy entity matcher.",
    buckets=(0.5, 0.6, 0.7, 0.75, 0.8, 0.85, 0.9, 0.95, 1.0),
)

provisional_entities_unresolved = Gauge(
    "finsight_provisional_entities_unresolved",
    "Number of provisional entities awaiting manual resolution.",
)