"""
pipelines/graph_rag/graph.py — GraphRAG LangGraph pipeline.

Nodes:
  guardrails → entity_extract → graph_traverse → [community_summary] →
  chunk_retrieve → generate → output_guardrails

The graph_traversal_trace is populated at each step for UI visualization.
"""
from __future__ import annotations

import asyncio
import time
import uuid
from typing import Any, Literal

import structlog
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

from orchestrator.state import GraphRAGState
from core.config import settings
from core.guardrails.validators import InputGuardrail, OutputGuardrail
from observability.tracing import trace_node

logger = structlog.get_logger(__name__)


def _get_llm(cheap: bool = False) -> Any:
    from langchain_openai import ChatOpenAI
    model = settings.llm_classification_model if cheap else settings.llm_generation_model
    return ChatOpenAI(
        model=model,
        api_key=settings.openai_api_key.get_secret_value(),
        temperature=0.0,
    )


def _accumulate_tokens(state: GraphRAGState, usage: dict[str, int]) -> None:
    existing = state.get("token_usage", {"prompt": 0, "completion": 0, "total": 0})
    for k in ("prompt", "completion", "total"):
        existing[k] = existing.get(k, 0) + usage.get(k, 0)
    state["token_usage"] = existing


# ── Node: input guardrails ────────────────────────────────────────────────────
@trace_node("guardrails_node")
async def guardrails_node(state: GraphRAGState) -> GraphRAGState:
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


# ── Node: entity extraction ───────────────────────────────────────────────────
@trace_node("entity_extract_node")
async def entity_extract_node(state: GraphRAGState) -> GraphRAGState:
    """
    Extract named entities from the query using LLM.
    In production, augment with spaCy NER for efficiency.
    """
    node_start = time.perf_counter()
    query = state["query"]
    llm = _get_llm(cheap=True)

    prompt = (
        "Extract all named entities (people, organizations, technologies, concepts, "
        "locations, events) from this query. "
        "Return as a comma-separated list. If none found, return 'NONE'.\n\n"
        f"Query: {query}"
    )
    response = await llm.ainvoke(prompt)
    raw = response.content.strip()
    entities = [e.strip() for e in raw.split(",") if e.strip() and e.strip() != "NONE"]
    state["entities_extracted"] = entities

    trace = state.get("graph_traversal_trace", [])
    trace.append({"step": "entity_extraction", "entities": entities, "query": query})
    state["graph_traversal_trace"] = trace

    if hasattr(response, "usage_metadata") and response.usage_metadata:
        _accumulate_tokens(state, {
            "prompt": response.usage_metadata.get("input_tokens", 0),
            "completion": response.usage_metadata.get("output_tokens", 0),
            "total": response.usage_metadata.get("total_tokens", 0),
        })

    timings = state.get("node_timings", {})
    timings["entity_extract_node"] = (time.perf_counter() - node_start) * 1000
    state["node_timings"] = timings
    logger.info("entities_extracted", count=len(entities), entities=entities[:5])
    return state


# ── Node: graph traversal ──────────────────────────────────────────────────────
@trace_node("graph_traverse_node")
async def graph_traverse_node(state: GraphRAGState) -> GraphRAGState:
    """
    Multi-hop Cypher traversal from extracted entities.
    Builds a subgraph of nodes and edges for context and visualization.
    """
    node_start = time.perf_counter()
    entities = state.get("entities_extracted", [])

    if not entities:
        state["graph_subgraph"] = {"nodes": [], "edges": []}
        return state

    try:
        from graph_db.neo4j_client import Neo4jClient
        client = Neo4jClient()

        # Step 1: find entity nodes in the graph
        matched = await client.find_entities(entities)
        entity_ids = [e["id"] for e in matched]

        trace = state.get("graph_traversal_trace", [])
        trace.append({
            "step": "entity_lookup",
            "matched": matched,
            "entity_ids": entity_ids,
        })

        # Step 2: multi-hop traversal
        if entity_ids:
            subgraph = await client.multi_hop_traverse(entity_ids, max_hops=2)
        else:
            subgraph = {"nodes": [], "edges": []}

        trace.append({"step": "graph_traversal", "subgraph_summary": {
            "node_count": len(subgraph["nodes"]),
            "edge_count": len(subgraph["edges"]),
        }})

        state["graph_subgraph"] = subgraph
        state["graph_traversal_trace"] = trace

        # Store Cypher queries for debugging/logging
        cypher_queries = state.get("cypher_queries_executed", [])
        cypher_queries.append(f"MATCH entities: {entities}")
        cypher_queries.append(f"TRAVERSE from {len(entity_ids)} entities, 2 hops")
        state["cypher_queries_executed"] = cypher_queries

    except Exception as exc:
        logger.error("graph_traverse_failed", error=str(exc))
        state["graph_subgraph"] = {"nodes": [], "edges": []}

    timings = state.get("node_timings", {})
    timings["graph_traverse_node"] = (time.perf_counter() - node_start) * 1000
    state["node_timings"] = timings
    return state


# ── Node: community summary (optional) ────────────────────────────────────────
@trace_node("community_summary_node")
async def community_summary_node(state: GraphRAGState) -> GraphRAGState:
    """
    Summarize graph neighborhoods (Microsoft GraphRAG approach).
    Only runs when feature_community_summaries is enabled.
    """
    node_start = time.perf_counter()
    if not settings.feature_community_summaries:
        state["community_summaries"] = []
        return state

    subgraph = state.get("graph_subgraph", {})
    nodes = subgraph.get("nodes", [])
    if not nodes:
        state["community_summaries"] = []
        return state

    llm = _get_llm(cheap=True)
    # Group nodes into communities (simplified: treat all nodes as one community)
    node_names = [n.get("name", "") for n in nodes[:20]]
    prompt = (
        "Summarize what these entities have in common and their relationships "
        "in 2-3 sentences:\n\n"
        + "\n".join(f"- {n}" for n in node_names if n)
    )
    response = await llm.ainvoke(prompt)
    state["community_summaries"] = [response.content.strip()]

    timings = state.get("node_timings", {})
    timings["community_summary_node"] = (time.perf_counter() - node_start) * 1000
    state["node_timings"] = timings
    return state


# ── Node: chunk retrieval from matched nodes ───────────────────────────────────
@trace_node("chunk_retrieve_node")
async def chunk_retrieve_node(state: GraphRAGState) -> GraphRAGState:
    """Fetch text chunks associated with matched graph entities."""
    node_start = time.perf_counter()
    subgraph = state.get("graph_subgraph", {})
    entity_nodes = [
        n["id"] for n in subgraph.get("nodes", [])
        if n.get("label") == "Entity"
    ]

    chunks: list[dict[str, Any]] = []
    if entity_nodes:
        try:
            from graph_db.neo4j_client import Neo4jClient
            client = Neo4jClient()
            raw_chunks = await client.get_chunks_for_entities(
                entity_nodes[:20], limit=settings.retrieval_top_k
            )
            chunks = raw_chunks
        except Exception as exc:
            logger.error("chunk_retrieve_failed", error=str(exc))

    # Fallback: vector search if no chunks from graph
    if not chunks:
        try:
            from core.embeddings.base import get_embedding_provider
            from core.retrieval.vector_store import HybridRetriever

            provider = get_embedding_provider()
            embed = await provider.embed_query(state["query"])
            retriever = HybridRetriever(use_reranking=settings.reranking_enabled)
            result = await retriever.retrieve(
                query=state["query"],
                query_vector=embed.vectors[0],
                top_k=settings.retrieval_top_k,
            )
            chunks = [
                {"id": d.id, "content": d.content, "metadata": d.metadata, "source": d.source}
                for d in result.documents
            ]
        except Exception as exc:
            logger.error("fallback_chunk_retrieve_failed", error=str(exc))

    state["retrieved_chunks"] = chunks

    timings = state.get("node_timings", {})
    timings["chunk_retrieve_node"] = (time.perf_counter() - node_start) * 1000
    state["node_timings"] = timings
    return state


# ── Node: generate ─────────────────────────────────────────────────────────────
@trace_node("generate_node")
async def generate_node(state: GraphRAGState) -> GraphRAGState:
    node_start = time.perf_counter()
    llm = _get_llm(cheap=False)
    query = state["query"]
    chunks = state.get("retrieved_chunks", [])
    community_summaries = state.get("community_summaries", [])
    subgraph = state.get("graph_subgraph", {})

    # Build rich context from graph + text chunks
    graph_context = ""
    if subgraph.get("nodes"):
        node_names = [n.get("name", "") for n in subgraph["nodes"][:15] if n.get("name")]
        edge_desc = [
            f"{e.get('source')} --[{e.get('type')}]--> {e.get('target')}"
            for e in subgraph.get("edges", [])[:10]
        ]
        graph_context = (
            "**Graph Context:**\n"
            f"Entities: {', '.join(node_names)}\n"
            f"Relationships:\n" + "\n".join(edge_desc)
        )

    community_context = (
        "**Community Summary:**\n" + "\n".join(community_summaries)
        if community_summaries else ""
    )

    chunk_context = "\n\n---\n\n".join(
        f"[Source: {c.get('source', 'graph')}]\n{c['content']}"
        for c in chunks[:settings.rerank_top_k]
    )

    full_context = "\n\n".join(filter(None, [graph_context, community_context, chunk_context]))

    prompt = (
        "You are a knowledgeable assistant with access to a knowledge graph and documents. "
        "Answer the question using the provided graph context and document chunks. "
        "Leverage entity relationships to provide a more connected, accurate answer. "
        "Cite your sources.\n\n"
        f"Context:\n{full_context}\n\n"
        f"Question: {query}\n\n"
        "Answer:"
    )
    response = await llm.ainvoke(prompt)
    state["final_answer"] = response.content.strip()
    state["sources"] = [
        {"id": c.get("id", ""), "source": c.get("source", "graph"), "score": 1.0}
        for c in chunks[:5]
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


# ── Node: output guardrails ────────────────────────────────────────────────────
@trace_node("output_guardrails_node")
async def output_guardrails_node(state: GraphRAGState) -> GraphRAGState:
    node_start = time.perf_counter()
    guardrail = OutputGuardrail()
    chunks = state.get("retrieved_chunks", [])
    result = guardrail.validate(
        answer=state.get("final_answer", ""),
        context_chunks=[c["content"] for c in chunks],
        trace_id=state.get("trace_id", ""),
    )
    state["output_guardrail_result"] = {
        "result": result.result.value,
        "reason": result.reason,
        "score": result.score,
    }
    if result.result.value == "block":
        state["final_answer"] = "Unable to generate a quality answer. Please rephrase."

    timings = state.get("node_timings", {})
    timings["output_guardrails_node"] = (time.perf_counter() - node_start) * 1000
    state["node_timings"] = timings
    return state


# ── Conditional edges ──────────────────────────────────────────────────────────

def route_after_input_guardrail(
    state: GraphRAGState,
) -> Literal["entity_extract_node", "__end__"]:
    result = state.get("input_guardrail_result", {})
    if result.get("result") == "block":
        state["final_answer"] = f"Request blocked: {result.get('reason')}"
        return "__end__"
    return "entity_extract_node"


# ── Graph compilation ──────────────────────────────────────────────────────────

def build_graph_rag_graph(checkpointer: Any = None) -> Any:
    graph = StateGraph(GraphRAGState)

    graph.add_node("guardrails_node", guardrails_node)
    graph.add_node("entity_extract_node", entity_extract_node)
    graph.add_node("graph_traverse_node", graph_traverse_node)
    graph.add_node("community_summary_node", community_summary_node)
    graph.add_node("chunk_retrieve_node", chunk_retrieve_node)
    graph.add_node("generate_node", generate_node)
    graph.add_node("output_guardrails_node", output_guardrails_node)

    graph.set_entry_point("guardrails_node")

    graph.add_conditional_edges(
        "guardrails_node",
        route_after_input_guardrail,
        {"entity_extract_node": "entity_extract_node", "__end__": END},
    )
    graph.add_edge("entity_extract_node", "graph_traverse_node")
    graph.add_edge("graph_traverse_node", "community_summary_node")
    graph.add_edge("community_summary_node", "chunk_retrieve_node")
    graph.add_edge("chunk_retrieve_node", "generate_node")
    graph.add_edge("generate_node", "output_guardrails_node")
    graph.add_edge("output_guardrails_node", END)

    compiled = graph.compile(checkpointer=checkpointer or MemorySaver())
    compiled.config = {"recursion_limit": settings.recursion_limit}
    return compiled


_graph_instance: Any = None


def get_graph_rag_graph() -> Any:
    global _graph_instance
    if _graph_instance is None:
        _graph_instance = build_graph_rag_graph()
    return _graph_instance


async def run_graph_rag(query: str, session_id: str | None = None) -> GraphRAGState:
    graph = get_graph_rag_graph()
    trace_id = str(uuid.uuid4())
    session_id = session_id or str(uuid.uuid4())

    initial_state: GraphRAGState = {
        "query": query,
        "session_id": session_id,
        "trace_id": trace_id,
        "pipeline_used": "graph_rag",
        "token_usage": {"prompt": 0, "completion": 0, "total": 0},
        "node_timings": {},
        "fallback_triggered": False,
        "graph_traversal_trace": [],
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
        logger.error("graph_rag_timeout", trace_id=trace_id)
        initial_state["final_answer"] = "Request timed out."
        initial_state["error"] = "timeout"
        return initial_state

    return final_state
