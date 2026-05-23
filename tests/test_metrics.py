"""
Tests for the metrics module.

Verifies that all metrics are registered and have the correct
types and label names. No running services needed.
"""

from prometheus_client import REGISTRY, CollectorRegistry
from prometheus_client import Counter, Gauge, Histogram

import finsight.services.metrics as m


def _get_metric(name: str):
    """Fetch a metric from the default registry by name."""
    for collector in REGISTRY._names_to_collectors.values():
        if hasattr(collector, "_name") and collector._name == name:
            return collector
    return None


def test_query_latency_is_histogram():
    assert isinstance(m.query_latency, Histogram)


def test_tokens_used_total_is_counter():
    assert isinstance(m.tokens_used_total, Counter)


def test_cache_hits_total_is_counter():
    assert isinstance(m.cache_hits_total, Counter)


def test_cache_misses_total_is_counter():
    assert isinstance(m.cache_misses_total, Counter)


def test_faithfulness_score_is_gauge():
    assert isinstance(m.faithfulness_score, Gauge)


def test_budget_utilization_is_gauge():
    assert isinstance(m.budget_utilization_pct, Gauge)


def test_circuit_breaker_opens_is_counter():
    assert isinstance(m.circuit_breaker_opens_total, Counter)


def test_mcp_latency_is_histogram():
    assert isinstance(m.mcp_latency, Histogram)


def test_query_latency_has_team_id_label():
    assert "team_id" in m.query_latency._labelnames


def test_tokens_used_has_model_label():
    assert "model" in m.tokens_used_total._labelnames


def test_cache_counters_have_team_id_label():
    assert "team_id" in m.cache_hits_total._labelnames
    assert "team_id" in m.cache_misses_total._labelnames


def test_mcp_latency_has_server_and_tool_labels():
    assert "server" in m.mcp_latency._labelnames
    assert "tool" in m.mcp_latency._labelnames


def test_circuit_breaker_opens_increments():
    before = m.circuit_breaker_opens_total.labels(name="test_cb")._value.get()
    m.circuit_breaker_opens_total.labels(name="test_cb").inc()
    after = m.circuit_breaker_opens_total.labels(name="test_cb")._value.get()
    assert after == before + 1


def test_faithfulness_score_sets_value():
    m.faithfulness_score.labels(team_id="ops").set(0.92)
    val = m.faithfulness_score.labels(team_id="ops")._value.get()
    assert abs(val - 0.92) < 0.001