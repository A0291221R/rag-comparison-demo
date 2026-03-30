"""
orchestrator/graph.py — Top-level LangGraph orchestrator.

Modes:
  - "auto"     : classify query → route to best pipeline
  - "agentic"  : force Agentic RAG
  - "graph"    : force GraphRAG
  - "parallel" : run both simultaneously via asyncio.gather, return comparison

Also enforces: token budget, recursion limit, per-node timeouts.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Literal

import structlog

from orchestrator.state import OrchestratorState, AgenticRAGState, GraphRAGState
from core.config import settings

from observability.metrics import (
    record_query_latency, record_token_usage, record_cost,
    record_fallback, record_relevance_score, track_active_query,
)

logger = structlog.get_logger(__name__)


# ── Query classifier ──────────────────────────────────────────────────────────

async def classify_query(query: str) -> Literal["agentic", "graph"]:
    """
    Classify whether a query benefits more from Agentic RAG or GraphRAG.

    GraphRAG: entity-heavy, relationship queries, "how are X and Y related"
    Agentic:  open-domain, reasoning-heavy, summarization
    """
    if not settings.feature_graphrag_enabled:
        return "agentic"

    llm_model = settings.llm_classification_model
    from langchain_openai import ChatOpenAI
    llm = ChatOpenAI(
        model=llm_model,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=0.0,
    )

    prompt = (
        "Classify this query as either 'graph' or 'agentic'.\n\n"
        "Choose 'graph' if the query:\n"
        "- Asks about relationships between specific named entities\n"
        "- Involves multi-hop reasoning over connected concepts\n"
        "- Contains multiple proper nouns that likely appear in a knowledge graph\n"
        "- Asks 'how are X and Y related', 'what connects X to Y'\n\n"
        "Choose 'agentic' if the query:\n"
        "- Is open-domain or general knowledge\n"
        "- Requires summarization or reasoning over documents\n"
        "- Has few or no specific named entities\n"
        "- Asks for explanations, comparisons, or analysis\n\n"
        "Output ONLY one word: 'graph' or 'agentic'\n\n"
        f"Query: {query}"
    )
    response = await llm.ainvoke(prompt)
    result = response.content.strip().lower()
    classification: Literal["agentic", "graph"] = "graph" if "graph" in result else "agentic"
    logger.info("query_classified", query=query[:60], classification=classification)
    return classification


# ── Token budget enforcer ─────────────────────────────────────────────────────

def check_token_budget(state: dict[str, Any]) -> bool:
    """Returns True if within budget, False if exceeded."""
    usage = state.get("token_usage", {})
    total = usage.get("total", 0)
    if total >= settings.token_budget_per_query:
        logger.warning(
            "token_budget_exceeded",
            used=total,
            budget=settings.token_budget_per_query,
        )
        return False
    return True


# ── Cost estimator ─────────────────────────────────────────────────────────────

COST_PER_1K_TOKENS = {
    "gpt-4o": {"input": 0.005, "output": 0.015},
    "gpt-4o-mini": {"input": 0.00015, "output": 0.0006},
    "text-embedding-3-small": {"input": 0.00002, "output": 0.0},
}


def estimate_cost(token_usage: dict[str, int], model: str) -> float:
    rates = COST_PER_1K_TOKENS.get(model, {"input": 0.003, "output": 0.006})
    prompt_cost = (token_usage.get("prompt", 0) / 1000) * rates["input"]
    completion_cost = (token_usage.get("completion", 0) / 1000) * rates["output"]
    return round(prompt_cost + completion_cost, 6)


# ── Main orchestrator ──────────────────────────────────────────────────────────

class RAGOrchestrator:
    """
    Entry point for all RAG queries.
    Handles routing, parallel execution, token budgeting, and comparison.
    """

    async def run(
        self,
        query: str,
        mode: Literal["auto", "agentic", "graph", "parallel"] = "auto",
        session_id: str | None = None,
    ) -> OrchestratorState:
        trace_id = str(uuid.uuid4())
        session_id = session_id or str(uuid.uuid4())
        start = time.perf_counter()

        logger.info(
            "orchestrator_start",
            query=query[:60],
            mode=mode,
            trace_id=trace_id,
        )

        state: OrchestratorState = {
            "query": query,
            "session_id": session_id,
            "trace_id": trace_id,
            "pipeline_mode": mode,
            "token_usage": {"prompt": 0, "completion": 0, "total": 0},
        }

        # Route
        if mode == "auto":
            classification = await classify_query(query)
            mode = classification

        with track_active_query(mode):
            if mode == "parallel":
                state = await self._run_parallel(state, session_id)
            elif mode == "graph":
                state = await self._run_graph_rag(state, session_id)
            else:
                state = await self._run_agentic_rag(state, session_id)

        # Aggregate cost estimate
        usage = state.get("token_usage", {})
        state["comparison_metrics"] = {
            **(state.get("comparison_metrics") or {}),
            "total_latency_ms": (time.perf_counter() - start) * 1000,
            "estimated_cost_usd": estimate_cost(usage, settings.llm_generation_model),
        }

        pipeline = state.get("pipeline_used", "unknown")
        latency_s = state["comparison_metrics"]["total_latency_ms"] / 1000
        usage = state.get("token_usage", {})
        cost = state["comparison_metrics"].get("estimated_cost_usd", 0.0)

        # Record Prometheus metrics
        record_query_latency(pipeline, "total", latency_s)
        record_token_usage(
            pipeline,
            settings.llm_generation_model,
            usage.get("prompt", 0),
            usage.get("completion", 0),
        )
        record_cost(pipeline, cost)

        # Record node-level timings
        metrics = state.get("comparison_metrics") or {}
        for pipe_key in ("agentic", "graph"):
            pipe_metrics = metrics.get(pipe_key, {})
            for node, node_ms in (pipe_metrics.get("node_timings") or {}).items():
                record_query_latency(pipe_key, node, node_ms / 1000)
            if pipe_metrics.get("fallback_triggered"):
                record_fallback(pipe_key, "vector")

        # Record retrieval relevance scores — only Agentic RAG has explicit relevance scoring
        agentic = state.get("agentic_result") or {}
        scores = agentic.get("relevance_scores", [])
        if scores:
            record_relevance_score("agentic_rag", sum(scores) / len(scores))
 
        # GraphRAG: use output guardrail grounding score as a proxy for relevance
        graph = state.get("graph_result") or {}
        graph_guardrail = (graph.get("output_guardrail_result") or {})
        graph_score = graph_guardrail.get("score")
        if graph_score is not None:
            record_relevance_score("graph_rag", float(graph_score))

        logger.info(
            "orchestrator_complete",
            trace_id=trace_id,
            pipeline=pipeline,
            latency_ms=round(state["comparison_metrics"]["total_latency_ms"], 1),
        )
        return state

    async def _run_agentic_rag(
        self, state: OrchestratorState, session_id: str
    ) -> OrchestratorState:
        from pipelines.agentic_rag.graph import run_agentic_rag
        result = await run_agentic_rag(state["query"], session_id=session_id)
        state["agentic_result"] = result
        state["final_answer"] = result.get("final_answer", "")
        state["pipeline_used"] = "agentic_rag"
        state["token_usage"] = result.get("token_usage", {})
        state["comparison_metrics"] = {
            "agentic": {
                "node_timings": result.get("node_timings", {}),
                "fallback_triggered": result.get("fallback_triggered", False),
                "retrieval_count": len(result.get("retrieved_chunks", [])),
                "iterations": result.get("iteration", 0),
            }
        }
        return state

    async def _run_graph_rag(
        self, state: OrchestratorState, session_id: str
    ) -> OrchestratorState:
        from pipelines.graph_rag.graph import run_graph_rag
        result = await run_graph_rag(state["query"], session_id=session_id)
        state["graph_result"] = result
        state["final_answer"] = result.get("final_answer", "")
        state["pipeline_used"] = "graph_rag"
        state["token_usage"] = result.get("token_usage", {})
        state["comparison_metrics"] = {
            "graph": {
                "node_timings": result.get("node_timings", {}),
                "entities_extracted": result.get("entities_extracted", []),
                "graph_node_count": len(
                    result.get("graph_subgraph", {}).get("nodes", [])
                ),
                "graph_edge_count": len(
                    result.get("graph_subgraph", {}).get("edges", [])
                ),
            }
        }
        return state

    async def _run_parallel(
        self, state: OrchestratorState, session_id: str
    ) -> OrchestratorState:
        """Run both pipelines in parallel, return side-by-side comparison."""
        from pipelines.agentic_rag.graph import run_agentic_rag
        from pipelines.graph_rag.graph import run_graph_rag

        agentic_task = asyncio.create_task(
            run_agentic_rag(state["query"], session_id=f"{session_id}-agentic")
        )
        graph_task = asyncio.create_task(
            run_graph_rag(state["query"], session_id=f"{session_id}-graph")
        )

        agentic_result, graph_result = await asyncio.gather(
            agentic_task, graph_task, return_exceptions=True
        )

        if isinstance(agentic_result, Exception):
            logger.error("agentic_parallel_failed", error=str(agentic_result))
            agentic_result = {"final_answer": f"Error: {agentic_result}", "token_usage": {}}

        if isinstance(graph_result, Exception):
            logger.error("graph_parallel_failed", error=str(graph_result))
            graph_result = {"final_answer": f"Error: {graph_result}", "token_usage": {}}

        state["agentic_result"] = agentic_result  # type: ignore
        state["graph_result"] = graph_result  # type: ignore
        state["pipeline_used"] = "parallel"

        # Combined token usage
        a_usage = agentic_result.get("token_usage", {})  # type: ignore
        g_usage = graph_result.get("token_usage", {})  # type: ignore
        state["token_usage"] = {
            "prompt": a_usage.get("prompt", 0) + g_usage.get("prompt", 0),
            "completion": a_usage.get("completion", 0) + g_usage.get("completion", 0),
            "total": a_usage.get("total", 0) + g_usage.get("total", 0),
        }

        state["comparison_metrics"] = {
            "agentic": {
                "answer": agentic_result.get("final_answer", ""),  # type: ignore
                "node_timings": agentic_result.get("node_timings", {}),  # type: ignore
                "token_usage": a_usage,
                "fallback_triggered": agentic_result.get("fallback_triggered", False),  # type: ignore
                "retrieval_count": len(agentic_result.get("retrieved_chunks", [])),  # type: ignore
            },
            "graph": {
                "answer": graph_result.get("final_answer", ""),  # type: ignore
                "node_timings": graph_result.get("node_timings", {}),  # type: ignore
                "token_usage": g_usage,
                "entities_extracted": graph_result.get("entities_extracted", []),  # type: ignore
                "graph_nodes": len(graph_result.get("graph_subgraph", {}).get("nodes", [])),  # type: ignore
                "graph_edges": len(graph_result.get("graph_subgraph", {}).get("edges", [])),  # type: ignore
            },
        }
        return state


# Module-level singleton
_orchestrator: RAGOrchestrator | None = None


def get_orchestrator() -> RAGOrchestrator:
    global _orchestrator
    if _orchestrator is None:
        _orchestrator = RAGOrchestrator()
    return _orchestrator
