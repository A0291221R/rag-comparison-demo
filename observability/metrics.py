"""
observability/metrics.py — Prometheus metrics for the RAG system.

Metrics:
  - rag_query_latency_seconds    histogram  by pipeline + node
  - rag_token_usage_total        counter    by model + pipeline
  - rag_cost_usd_total           counter
  - rag_fallback_triggered_total counter
  - rag_retrieval_relevance_score gauge
"""
from __future__ import annotations

from contextlib import contextmanager
from typing import Generator

try:
    from prometheus_client import (
        Counter, Gauge, Histogram, REGISTRY,
        CollectorRegistry,
    )
    PROMETHEUS_AVAILABLE = True
except ImportError:
    PROMETHEUS_AVAILABLE = False


def _make_metrics() -> dict:
    if not PROMETHEUS_AVAILABLE:
        return {}

    return {
        "query_latency": Histogram(
            "rag_query_latency_seconds",
            "End-to-end query latency",
            labelnames=["pipeline", "node"],
            buckets=[0.1, 0.25, 0.5, 1.0, 2.0, 5.0, 10.0, 30.0],
        ),
        "token_usage": Counter(
            "rag_token_usage_total",
            "Total tokens used",
            labelnames=["model", "pipeline", "token_type"],
        ),
        "cost_usd": Counter(
            "rag_cost_usd_total",
            "Total estimated USD cost",
            labelnames=["pipeline"],
        ),
        "fallback_triggered": Counter(
            "rag_fallback_triggered_total",
            "Number of times fallback was triggered",
            labelnames=["pipeline", "fallback_level"],
        ),
        "retrieval_relevance": Gauge(
            "rag_retrieval_relevance_score",
            "Average retrieval relevance score",
            labelnames=["pipeline"],
        ),
        "active_queries": Gauge(
            "rag_active_queries",
            "Number of queries currently being processed",
            labelnames=["pipeline"],
        ),
    }


# Module-level metrics singleton
try:
    METRICS = _make_metrics()
except Exception:
    METRICS = {}


def record_query_latency(pipeline: str, node: str, latency_seconds: float) -> None:
    if m := METRICS.get("query_latency"):
        m.labels(pipeline=pipeline, node=node).observe(latency_seconds)


def record_token_usage(
    pipeline: str, model: str, prompt_tokens: int, completion_tokens: int
) -> None:
    if m := METRICS.get("token_usage"):
        m.labels(model=model, pipeline=pipeline, token_type="prompt").inc(prompt_tokens)
        m.labels(model=model, pipeline=pipeline, token_type="completion").inc(completion_tokens)


def record_cost(pipeline: str, cost_usd: float) -> None:
    if m := METRICS.get("cost_usd"):
        m.labels(pipeline=pipeline).inc(cost_usd)


def record_fallback(pipeline: str, level: str) -> None:
    if m := METRICS.get("fallback_triggered"):
        m.labels(pipeline=pipeline, fallback_level=level).inc()


def record_relevance_score(pipeline: str, score: float) -> None:
    if m := METRICS.get("retrieval_relevance"):
        m.labels(pipeline=pipeline).set(score)


@contextmanager
def track_active_query(pipeline: str) -> Generator[None, None, None]:
    if m := METRICS.get("active_queries"):
        m.labels(pipeline=pipeline).inc()
    try:
        yield
    finally:
        if m := METRICS.get("active_queries"):
            m.labels(pipeline=pipeline).dec()
