"""
orchestrator/state.py — Unified TypedDict state shared by both pipelines.

Both AgenticRAGState and GraphRAGState inherit from BaseRAGState so the
top-level orchestrator can treat them uniformly.
"""
from __future__ import annotations

from typing import Any, Annotated
from typing_extensions import TypedDict
import operator


# ── Base state (shared by both pipelines) ─────────────────────────────────────

class BaseRAGState(TypedDict, total=False):
    # Query
    query: str
    rewritten_query: str
    session_id: str
    trace_id: str

    # Output
    final_answer: str
    sources: list[dict[str, Any]]

    # Control
    fallback_triggered: bool
    error: str | None
    iteration: int

    # Token / cost tracking
    token_usage: dict[str, int]      # {"prompt": N, "completion": N, "total": N}
    estimated_cost_usd: float

    # Routing metadata
    pipeline_used: str               # "agentic_rag" | "graph_rag" | "parallel"
    node_timings: dict[str, float]   # node_name → latency_ms


# ── Agentic RAG state ─────────────────────────────────────────────────────────

class AgenticRAGState(BaseRAGState, total=False):
    # Retrieval
    retrieved_chunks: list[dict[str, Any]]   # serialized Documents
    relevance_scores: list[float]
    retrieval_strategy: str

    # Agent internals
    agent_scratchpad: list[dict[str, Any]]   # intermediate reasoning steps
    self_reflection_score: float

    # Guardrail results
    input_guardrail_result: dict[str, Any]
    output_guardrail_result: dict[str, Any]


# ── GraphRAG state ────────────────────────────────────────────────────────────

class GraphRAGState(BaseRAGState, total=False):
    # Entity extraction
    entities_extracted: list[str]

    # Graph data
    graph_subgraph: dict[str, Any]          # {"nodes": [...], "edges": [...]}
    community_summaries: list[str]
    graph_traversal_trace: list[dict[str, Any]]  # step-by-step traversal log
    cypher_queries_executed: list[str]

    # Chunk retrieval from matched nodes
    retrieved_chunks: list[dict[str, Any]]

    # Guardrail results
    input_guardrail_result: dict[str, Any]
    output_guardrail_result: dict[str, Any]


# ── Orchestrator top-level state ──────────────────────────────────────────────

class OrchestratorState(TypedDict, total=False):
    query: str
    session_id: str
    trace_id: str
    pipeline_mode: str          # "agentic" | "graph" | "parallel" | "auto"

    # Results from both pipelines (populated in parallel mode)
    agentic_result: AgenticRAGState
    graph_result: GraphRAGState

    # Chosen final answer
    final_answer: str
    pipeline_used: str
    comparison_metrics: dict[str, Any]
    token_usage: dict[str, int]
