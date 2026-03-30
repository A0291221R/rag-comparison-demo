"""
pipelines/agentic_rag/graph.py — Agentic RAG LangGraph pipeline.

Nodes:
  query_rewrite → retrieve → grade_relevance → generate → self_reflect
                                     ↓ (low relevance)
                                  fallback

Conditional edges enforce max_iterations and relevance thresholds.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Literal

import structlog
from langchain_core.documents import Document as LCDocument
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from orchestrator.state import AgenticRAGState
from core.config import settings
from core.guardrails.validators import InputGuardrail, OutputGuardrail
from core.fallback.chain import FallbackChain, FallbackLevel
from observability.tracing import trace_node

logger = structlog.get_logger(__name__)


# ── LLM helpers ───────────────────────────────────────────────────────────────

def _get_llm(cheap: bool = False) -> Any:
    """Model routing: cheap model for classification, expensive for generation."""
    from langchain_openai import ChatOpenAI
    model = settings.llm_classification_model if cheap else settings.llm_generation_model
    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=0.0,
    )


def _accumulate_tokens(state: AgenticRAGState, usage: dict[str, int]) -> None:
    existing = state.get("token_usage", {"prompt": 0, "completion": 0, "total": 0})
    for k in ("prompt", "completion", "total"):
        existing[k] = existing.get(k, 0) + usage.get(k, 0)
    state["token_usage"] = existing


# ── Node: guardrails (input) ───────────────────────────────────────────────────
@trace_node("guardrails_node")
async def guardrails_node(state: AgenticRAGState) -> AgenticRAGState:
    node_start = time.perf_counter()
    guardrail = InputGuardrail()
    result = guardrail.validate(state["query"])
    state["input_guardrail_result"] = {
        "result": result.result.value,
        "reason": result.reason,
        "score": result.score,
    }
    if result.sanitized_text:
        state["query"] = result.sanitized_text

    timings = state.get("node_timings", {})
    timings["guardrails_node"] = (time.perf_counter() - node_start) * 1000
    state["node_timings"] = timings
    return state


# ── Node: query rewrite ────────────────────────────────────────────────────────
@trace_node("query_rewrite_node")
async def query_rewrite_node(state: AgenticRAGState) -> AgenticRAGState:
    node_start = time.perf_counter()
    llm = _get_llm(cheap=True)
    query = state["query"]

    prompt = (
        "Rewrite the following search query to improve retrieval accuracy. "
        "Make it more specific, expand acronyms, and add relevant synonyms. "
        "Output ONLY the rewritten query, nothing else.\n\n"
        f"Original query: {query}"
    )
    response = await llm.ainvoke(prompt)
    rewritten = response.content.strip()
    state["rewritten_query"] = rewritten

    # Track token usage
    if hasattr(response, "usage_metadata") and response.usage_metadata:
        _accumulate_tokens(state, {
            "prompt": response.usage_metadata.get("input_tokens", 0),
            "completion": response.usage_metadata.get("output_tokens", 0),
            "total": response.usage_metadata.get("total_tokens", 0),
        })

    timings = state.get("node_timings", {})
    timings["query_rewrite_node"] = (time.perf_counter() - node_start) * 1000
    state["node_timings"] = timings

    logger.info("query_rewritten", original=query[:60], rewritten=rewritten[:60])
    return state


# ── Node: retrieve ─────────────────────────────────────────────────────────────
@trace_node("retrieve_node")
async def retrieve_node(state: AgenticRAGState) -> AgenticRAGState:
    node_start = time.perf_counter()
    query = state.get("rewritten_query") or state["query"]

    try:
        from core.embeddings.base import get_embedding_provider
        from core.retrieval.vector_store import HybridRetriever

        provider = get_embedding_provider(
            use_finetuned=settings.feature_finetuned_embeddings
        )
        embed_result = await provider.embed_query(query)
        query_vector = embed_result.vectors[0]

        retriever = HybridRetriever(use_reranking=settings.reranking_enabled)
        result = await retriever.retrieve(
            query=query,
            query_vector=query_vector,
            top_k=settings.retrieval_top_k,
        )
        state["retrieved_chunks"] = [
            {
                "id": doc.id,
                "content": doc.content,
                "metadata": doc.metadata,
                "score": doc.score,
                "source": doc.source,
            }
            for doc in result.documents
        ]
        state["retrieval_strategy"] = result.strategy
        _accumulate_tokens(state, {
            "prompt": embed_result.token_count,
            "completion": 0,
            "total": embed_result.token_count,
        })
    except Exception as exc:
        logger.error("retrieve_node_failed", error=str(exc))
        state["retrieved_chunks"] = []
        state["error"] = f"Retrieval error: {exc}"

    timings = state.get("node_timings", {})
    timings["retrieve_node"] = (time.perf_counter() - node_start) * 1000
    state["node_timings"] = timings
    return state


# ── Node: grade relevance ──────────────────────────────────────────────────────
@trace_node("grade_relevance_node")
async def grade_relevance_node(state: AgenticRAGState) -> AgenticRAGState:
    node_start = time.perf_counter()
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        state["relevance_scores"] = []
        return state

    llm = _get_llm(cheap=True)
    query = state.get("rewritten_query") or state["query"]
    scores: list[float] = []

    # Grade each chunk (in practice batch this for efficiency)
    grade_tasks = []
    for chunk in chunks[:settings.retrieval_top_k]:
        prompt = (
            "Rate how relevant this document chunk is to the query on a scale of 0.0 to 1.0. "
            "Output ONLY a float number, nothing else.\n\n"
            f"Query: {query}\n\nChunk: {chunk['content'][:500]}"
        )
        grade_tasks.append(llm.ainvoke(prompt))

    responses = await asyncio.gather(*grade_tasks, return_exceptions=True)
    for resp in responses:
        try:
            score = float(str(resp.content).strip()) if not isinstance(resp, Exception) else 0.0  # type: ignore
            scores.append(max(0.0, min(1.0, score)))
        except (ValueError, AttributeError):
            scores.append(0.5)

    state["relevance_scores"] = scores

    timings = state.get("node_timings", {})
    timings["grade_relevance_node"] = (time.perf_counter() - node_start) * 1000
    state["node_timings"] = timings
    return state


# ── Node: generate ─────────────────────────────────────────────────────────────
@trace_node("generate_node")
async def generate_node(state: AgenticRAGState) -> AgenticRAGState:
    node_start = time.perf_counter()
    query = state.get("rewritten_query") or state["query"]
    chunks = state.get("retrieved_chunks", [])
    scores = state.get("relevance_scores", [])

    # Filter to relevant chunks above threshold
    relevant_chunks = [
        c for c, s in zip(chunks, scores + [1.0] * len(chunks))
        if s >= settings.relevance_threshold
    ] or chunks[:3]  # fallback to top-3 if none pass threshold

    context = "\n\n---\n\n".join(
        f"[Source: {c.get('source', 'unknown')}]\n{c['content']}"
        for c in relevant_chunks
    )

    llm = _get_llm(cheap=False)
    prompt = (
        "You are a helpful assistant. Answer the question using ONLY the provided context. "
        "If the context doesn't contain enough information, say so explicitly. "
        "Always cite your sources.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    response = await llm.ainvoke(prompt)
    state["final_answer"] = response.content.strip()

    # Track sources
    state["sources"] = [
        {"id": c.get("id", ""), "source": c.get("source", ""), "score": c.get("score", 0.0)}
        for c in relevant_chunks
    ]

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        _accumulate_tokens(state, {
            "prompt": response.usage_metadata.get("input_tokens", 0),
            "completion": response.usage_metadata.get("output_tokens", 0),
            "total": response.usage_metadata.get("total_tokens", 0),
        })

    timings = state.get("node_timings", {})
    timings["generate_node"] = (time.perf_counter() - node_start) * 1000
    state["node_timings"] = timings
    return state


# ── Node: self-reflect ─────────────────────────────────────────────────────────
@trace_node("self_reflect_node")
async def self_reflect_node(state: AgenticRAGState) -> AgenticRAGState:
    node_start = time.perf_counter()
    llm = _get_llm(cheap=True)
    query = state["query"]
    answer = state.get("final_answer", "")

    prompt = (
        "Evaluate this answer for quality. Score from 0.0 (poor) to 1.0 (excellent). "
        "Consider: completeness, factual grounding, relevance to question. "
        "Output ONLY a float score.\n\n"
        f"Question: {query}\n\nAnswer: {answer}"
    )
    response = await llm.ainvoke(prompt)
    try:
        score = float(str(response.content).strip())
        score = max(0.0, min(1.0, score))
    except ValueError:
        score = 0.5
    state["self_reflection_score"] = score

    timings = state.get("node_timings", {})
    timings["self_reflect_node"] = (time.perf_counter() - node_start) * 1000
    state["node_timings"] = timings
    return state


# ── Node: fallback ─────────────────────────────────────────────────────────────
@trace_node("fallback_node")
async def fallback_node(state: AgenticRAGState) -> AgenticRAGState:
    node_start = time.perf_counter()
    logger.warning("fallback_triggered", query=state["query"][:60])
    state["fallback_triggered"] = True

    chain = FallbackChain()

    async def graph_search(query: str) -> str | None:
        """Try Neo4j graph search as fallback."""
        try:
            from graph_db.neo4j_client import Neo4jClient
            client = Neo4jClient()
            results = await client.fulltext_search(query, limit=5)
            if results:
                context = "\n".join(r.get("content", "") for r in results)
                llm = _get_llm(cheap=False)
                resp = await llm.ainvoke(
                    f"Answer using this context:\n{context}\n\nQuestion: {query}"
                )
                return resp.content
        except Exception:
            pass
        return None

    chain.register(FallbackLevel.GRAPH, graph_search)

    if settings.feature_web_search_fallback:
        async def web_search(query: str) -> str | None:
            # Placeholder: integrate with Tavily or SerpAPI
            return None
        chain.register(FallbackLevel.WEB, web_search)

    result = await chain.run(state["query"])
    state["final_answer"] = result.answer

    timings = state.get("node_timings", {})
    timings["fallback_node"] = (time.perf_counter() - node_start) * 1000
    state["node_timings"] = timings
    return state


# ── Node: output guardrails ────────────────────────────────────────────────────
@trace_node("output_guardrails_node")
async def output_guardrails_node(state: AgenticRAGState) -> AgenticRAGState:
    node_start = time.perf_counter()
    guardrail = OutputGuardrail()
    chunks = state.get("retrieved_chunks", [])
    context = [c["content"] for c in chunks]
    result = guardrail.validate(
        answer=state.get("final_answer", ""),
        context_chunks=context,
        trace_id=state.get("trace_id", ""),
    )
    state["output_guardrail_result"] = {
        "result": result.result.value,
        "reason": result.reason,
        "score": result.score,
    }
    if result.result.value == "block":
        state["final_answer"] = (
            "I'm unable to provide an answer that meets quality requirements. "
            "Please try rephrasing your question."
        )

    timings = state.get("node_timings", {})
    timings["output_guardrails_node"] = (time.perf_counter() - node_start) * 1000
    state["node_timings"] = timings
    return state


# ── Conditional edge functions ────────────────────────────────────────────────

def route_after_grading(
    state: AgenticRAGState,
) -> Literal["generate_node", "fallback_node"]:
    scores = state.get("relevance_scores", [])
    iteration = state.get("iteration", 0)
    avg_score = sum(scores) / len(scores) if scores else 0.0

    if avg_score < settings.relevance_threshold and iteration < settings.max_agent_iterations:
        state["iteration"] = iteration + 1
        return "fallback_node"
    return "generate_node"


def route_after_reflection(
    state: AgenticRAGState,
) -> Literal["output_guardrails_node", "retrieve_node"]:
    reflection_score = state.get("self_reflection_score", 1.0)
    iteration = state.get("iteration", 0)

    if reflection_score < 0.6 and iteration < settings.max_agent_iterations:
        state["iteration"] = iteration + 1
        return "retrieve_node"
    return "output_guardrails_node"


def route_after_input_guardrail(
    state: AgenticRAGState,
) -> Literal["query_rewrite_node", "__end__"]:
    result = state.get("input_guardrail_result", {})
    if result.get("result") == "block":
        state["final_answer"] = f"Request blocked: {result.get('reason', 'policy violation')}"
        return "__end__"
    return "query_rewrite_node"


# ── Graph compilation ─────────────────────────────────────────────────────────

def build_agentic_rag_graph(checkpointer: Any = None) -> Any:
    """
    Compile and return the Agentic RAG LangGraph StateGraph.
    Optionally accepts a checkpointer for state persistence.
    """
    graph = StateGraph(AgenticRAGState)

    # Register nodes
    graph.add_node("guardrails_node", guardrails_node)
    graph.add_node("query_rewrite_node", query_rewrite_node)
    graph.add_node("retrieve_node", retrieve_node)
    graph.add_node("grade_relevance_node", grade_relevance_node)
    graph.add_node("generate_node", generate_node)
    graph.add_node("self_reflect_node", self_reflect_node)
    graph.add_node("fallback_node", fallback_node)
    graph.add_node("output_guardrails_node", output_guardrails_node)

    # Entry point
    graph.set_entry_point("guardrails_node")

    # Edges
    graph.add_conditional_edges(
        "guardrails_node",
        route_after_input_guardrail,
        {
            "query_rewrite_node": "query_rewrite_node",
            "__end__": END,
        },
    )
    graph.add_edge("query_rewrite_node", "retrieve_node")
    graph.add_edge("retrieve_node", "grade_relevance_node")
    graph.add_conditional_edges(
        "grade_relevance_node",
        route_after_grading,
        {
            "generate_node": "generate_node",
            "fallback_node": "fallback_node",
        },
    )
    graph.add_edge("fallback_node", "output_guardrails_node")
    graph.add_edge("generate_node", "self_reflect_node")
    graph.add_conditional_edges(
        "self_reflect_node",
        route_after_reflection,
        {
            "output_guardrails_node": "output_guardrails_node",
            "retrieve_node": "retrieve_node",
        },
    )
    graph.add_edge("output_guardrails_node", END)

    compiled = graph.compile(
        checkpointer=checkpointer or MemorySaver(),
        interrupt_before=[],  # set to ["generate_node"] for HITL
    )
    compiled.config = {"recursion_limit": settings.recursion_limit}
    return compiled


# Module-level singleton
_graph_instance: Any = None


def get_agentic_rag_graph() -> Any:
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = build_agentic_rag_graph()
    return _graph_instance


async def run_agentic_rag(query: str, session_id: str | None = None) -> AgenticRAGState:
    """High-level entry point: run Agentic RAG for a query."""
    graph = get_agentic_rag_graph()
    trace_id = str(uuid.uuid4())
    session_id = session_id or str(uuid.uuid4())

    initial_state: AgenticRAGState = {
        "query": query,
        "session_id": session_id,
        "trace_id": trace_id,
        "iteration": 0,
        "pipeline_used": "agentic_rag",
        "token_usage": {"prompt": 0, "completion": 0, "total": 0},
        "node_timings": {},
        "fallback_triggered": False,
    }

    config = {
        "configurable": {"thread_id": session_id},
        "recursion_limit": settings.recursion_limit,
    }

    try:
        final_state = await asyncio.wait_for(
            graph.ainvoke(initial_state, config=config),
            timeout=settings.node_timeout_seconds * 5,
        )
    except asyncio.TimeoutError:
        logger.error("agentic_rag_timeout", trace_id=trace_id)
        initial_state["final_answer"] = "Request timed out. Please try again."
        initial_state["error"] = "timeout"
        return initial_state

    return final_state
