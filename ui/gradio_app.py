"""
ui/gradio_app.py — Side-by-side Agentic RAG vs GraphRAG comparison dashboard.
Compatible with Gradio 4.x and 5.x/6.x.
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any

import gradio as gr
import plotly.graph_objects as go

import warnings
import logging
logging.getLogger("asyncio").setLevel(logging.CRITICAL)
warnings.filterwarnings("ignore", message=".*Event loop is closed.*")

# ── Async runner ──────────────────────────────────────────────────────────────

# def run_async(coro: Any) -> Any:
#     """Run an async coroutine from sync Gradio handlers."""
#     try:
#         loop = asyncio.get_event_loop()
#         if loop.is_running():
#             import concurrent.futures
#             with concurrent.futures.ThreadPoolExecutor() as pool:
#                 future = pool.submit(asyncio.run, coro)
#                 return future.result(timeout=120)
#         return loop.run_until_complete(coro)
#     except RuntimeError:
#         return asyncio.run(coro)

def run_async(coro: Any) -> Any:
    """Always run in a fresh thread with its own event loop — safe for Gradio/AnyIO on Windows."""
    import concurrent.futures
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        future = pool.submit(asyncio.run, coro)
        return future.result(timeout=120)

# ── Core query runner ─────────────────────────────────────────────────────────

def run_query(query: str, pipeline_mode: str, session_id: str) -> tuple:
    if not query.strip():
        return ("Please enter a query.", "", "", None, None, "{}", "{}")

    import requests
    from core.config import settings

    mode_map = {
        "🤖 Agentic RAG Only": "agentic",
        "🕸️ GraphRAG Only": "graph",
        "⚡ Both (Parallel)": "parallel",
        "🎯 Auto-Route": "auto",
    }
    mode = mode_map.get(pipeline_mode, "auto")

    start = time.perf_counter()
    try:
        resp = requests.post(
            f"http://localhost:{settings.api_port}/api/v1/query",
            json={"query": query, "pipeline": mode, "session_id": session_id or None},
            headers={"X-API-Key": settings.api_key},
            timeout=120,
        )
        resp.raise_for_status()
        api_result = resp.json()
        # Fetch full trace for detailed metrics
        trace_id = api_result.get("trace_id", "")
        try:
            trace_resp = requests.get(
                f"http://localhost:{settings.api_port}/api/v1/traces/{trace_id}",
                headers={"X-API-Key": settings.api_key},
                timeout=10,
            )
            result = trace_resp.json() if trace_resp.ok else api_result
        except Exception:
            result = api_result
        result["pipeline_used"] = api_result.get("pipeline_used", mode)
        result["final_answer"] = api_result.get("final_answer", "")
        result["token_usage"] = api_result.get("token_usage", {})
        result["comparison_metrics"] = api_result.get("comparison_metrics", {})
    except Exception as exc:
        error_msg = f"❌ Error: {exc}"
        return (error_msg, error_msg, "", None, None, "{}", "{}")

    total_ms = (time.perf_counter() - start) * 1000

    agentic = result.get("agentic_result") or {}
    graph = result.get("graph_result") or {}
    metrics = result.get("comparison_metrics") or {}

    agentic_answer = agentic.get("final_answer") or result.get("final_answer", "—")
    graph_answer = graph.get("final_answer") or (
        result.get("final_answer", "—") if mode in ("graph", "auto") else "—"
    )

    # Metrics table
    a_metrics = metrics.get("agentic", {})
    g_metrics = metrics.get("graph", {})
    a_tokens = a_metrics.get("token_usage", {}) or agentic.get("token_usage", {})
    g_tokens = g_metrics.get("token_usage", {}) or graph.get("token_usage", {})

    metrics_md = f"""
| Metric | Agentic RAG | GraphRAG |
|--------|------------|---------|
| **Latency** | {sum((a_metrics.get('node_timings') or {}).values()):.0f} ms | {sum((g_metrics.get('node_timings') or {}).values()):.0f} ms |
| **Tokens** | {a_tokens.get('total', '—')} | {g_tokens.get('total', '—')} |
| **Retrieval count** | {a_metrics.get('retrieval_count', '—')} | {g_metrics.get('graph_nodes', '—')} nodes |
| **Fallback** | {'✅' if a_metrics.get('fallback_triggered') else '❌'} | — |
| **Entities** | — | {', '.join((g_metrics.get('entities_extracted') or [])[:5]) or '—'} |
| **Graph edges** | — | {g_metrics.get('graph_edges', '—')} |
| **Wall clock** | {total_ms:.0f} ms | {total_ms:.0f} ms |
"""

    # Timing chart
    a_timings = a_metrics.get("node_timings") or agentic.get("node_timings", {})
    g_timings = g_metrics.get("node_timings") or graph.get("node_timings", {})
    all_nodes = sorted(set(list(a_timings.keys()) + list(g_timings.keys())))

    timing_fig = go.Figure(data=[
        go.Bar(name="Agentic RAG", x=all_nodes,
               y=[a_timings.get(n, 0) for n in all_nodes], marker_color="#6366f1"),
        go.Bar(name="GraphRAG", x=all_nodes,
               y=[g_timings.get(n, 0) for n in all_nodes], marker_color="#10b981"),
    ])
    timing_fig.update_layout(
        title="Node Latency (ms)", barmode="group",
        xaxis_tickangle=-30, yaxis_title="ms",
        height=300, margin=dict(l=40, r=20, t=60, b=80),
    )

    # Graph viz
    subgraph = graph.get("graph_subgraph") or {}
    graph_html = _build_graph_html(subgraph)

    # Traces
    agentic_trace = json.dumps(
        {k: v for k, v in agentic.items() if k != "retrieved_chunks"}, indent=2, default=str
    )
    graph_trace = json.dumps(
        {k: v for k, v in graph.items() if k != "retrieved_chunks"}, indent=2, default=str
    )

    return (agentic_answer, graph_answer, metrics_md, timing_fig, graph_html,
            agentic_trace, graph_trace)


def _build_graph_html(subgraph: dict) -> str:
    nodes = subgraph.get("nodes", [])
    edges = subgraph.get("edges", [])
    if not nodes:
        return "<p style='color:#888;text-align:center;padding:40px'>No graph data for this query.</p>"
    try:
        from pyvis.network import Network  # type: ignore
        net = Network(height="400px", width="100%", bgcolor="#1e1e2e", font_color="#cdd6f4")
        color_map = {"Entity": "#6366f1", "Chunk": "#10b981", "Document": "#f59e0b"}
        for node in nodes[:50]:
            nid = node.get("id", "")
            label = (node.get("name") or nid)[:30]
            net.add_node(nid, label=label, color=color_map.get(node.get("label", ""), "#94a3b8"))
        for edge in edges[:100]:
            src, tgt = edge.get("source"), edge.get("target")
            if src and tgt:
                net.add_edge(src, tgt, label=edge.get("type", ""))
        return net.generate_html()
    except ImportError:
        node_list = "\n".join(f"• {n.get('name', n.get('id', ''))}" for n in nodes[:20])
        return f"<pre style='padding:16px;background:#1e1e2e;color:#cdd6f4'>{node_list}</pre>"


def submit_feedback(trace_str: str, rating: str) -> str:
    import requests
    from core.config import settings
    try:
        trace = json.loads(trace_str) if trace_str else {}
        trace_id = trace.get("trace_id", "unknown")
        requests.post(
            f"http://localhost:{settings.api_port}/api/v1/feedback",
            json={"trace_id": trace_id, "rating": rating},
            headers={"X-API-Key": settings.api_key},
            timeout=5,
        )
        return f"✅ Feedback ({rating}) submitted"
    except Exception as exc:
        return f"⚠️ Could not submit: {exc}"


# ── UI ────────────────────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Agentic RAG vs GraphRAG") as demo:

        gr.Markdown("""
# 🔍 Agentic RAG vs GraphRAG — Side-by-Side Comparison
Compare two production-grade retrieval pipelines on any query.
**Agentic RAG**: query rewriting · relevance grading · self-reflection loop.
**GraphRAG**: entity extraction → Neo4j multi-hop traversal → context-grounded generation.
""")

        with gr.Row():
            with gr.Column(scale=4):
                query_input = gr.Textbox(
                    label="Query",
                    placeholder="e.g. How do transformers relate to attention mechanisms?",
                    lines=2,
                )
            with gr.Column(scale=2):
                pipeline_selector = gr.Radio(
                    choices=["🎯 Auto-Route", "⚡ Both (Parallel)",
                             "🤖 Agentic RAG Only", "🕸️ GraphRAG Only"],
                    value="⚡ Both (Parallel)",
                    label="Pipeline Mode",
                )

        with gr.Row():
            session_id_input = gr.Textbox(label="Session ID (optional)", scale=2)
            run_btn = gr.Button("▶  Run Query", variant="primary", scale=1)
            clear_btn = gr.Button("🗑  Clear", scale=1)

        gr.Examples(
            examples=[
                ["How are transformers and attention mechanisms related?", "⚡ Both (Parallel)"],
                ["What are the key differences between BERT and GPT?", "🎯 Auto-Route"],
                ["Which researchers co-authored papers on graph neural networks?", "🕸️ GraphRAG Only"],
                ["Summarize the main findings on retrieval-augmented generation", "🤖 Agentic RAG Only"],
            ],
            inputs=[query_input, pipeline_selector],
        )

        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🤖 Agentic RAG")
                agentic_answer = gr.Markdown(value="*Run a query to see results…*")
            with gr.Column():
                gr.Markdown("### 🕸️ GraphRAG")
                graph_answer = gr.Markdown(value="*Run a query to see results…*")

        with gr.Accordion("📊 Comparison Metrics", open=True):
            metrics_table = gr.Markdown(value="*Metrics will appear after first query.*")

        with gr.Accordion("⏱ Node Latency Breakdown", open=False):
            timing_chart = gr.Plot()

        with gr.Accordion("🕸️ GraphRAG Subgraph", open=False):
            graph_viz = gr.HTML(value="<p style='color:#888;padding:16px'>Run a query to visualize the subgraph.</p>")

        with gr.Accordion("🔎 Execution Traces", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Agentic RAG**")
                    agentic_trace = gr.JSON()
                with gr.Column():
                    gr.Markdown("**GraphRAG**")
                    graph_trace = gr.JSON()

        with gr.Row():
            thumbs_up_btn = gr.Button("👍 Helpful", variant="secondary")
            thumbs_down_btn = gr.Button("👎 Not Helpful", variant="secondary")
            feedback_status = gr.Textbox(label="", interactive=False, max_lines=1)

        last_agentic_trace = gr.State("{}")
        last_graph_trace = gr.State("{}")

        def on_run(query, pipeline, session_id):
            results = run_query(query, pipeline, session_id)
            a_ans, g_ans, metrics, timing, gviz, a_str, g_str = results
            try:
                a_trace = json.loads(a_str)
                g_trace = json.loads(g_str)
            except Exception:
                a_trace, g_trace = {}, {}
            return (a_ans, g_ans, metrics, timing, gviz,
                    a_trace, g_trace, a_str, g_str)

        outputs = [agentic_answer, graph_answer, metrics_table, timing_chart,
                   graph_viz, agentic_trace, graph_trace,
                   last_agentic_trace, last_graph_trace]

        run_btn.click(fn=on_run,
                      inputs=[query_input, pipeline_selector, session_id_input],
                      outputs=outputs)
        query_input.submit(fn=on_run,
                           inputs=[query_input, pipeline_selector, session_id_input],
                           outputs=outputs)
        clear_btn.click(
            fn=lambda: ("", "", "*Metrics will appear after first query.*", None,
                        "<p style='color:#888;padding:16px'>Run a query to visualize the subgraph.</p>",
                        {}, {}, "{}", "{}"),
            outputs=outputs,
        )
        thumbs_up_btn.click(fn=lambda t: submit_feedback(t, "thumbs_up"),
                            inputs=[last_agentic_trace], outputs=[feedback_status])
        thumbs_down_btn.click(fn=lambda t: submit_feedback(t, "thumbs_down"),
                              inputs=[last_agentic_trace], outputs=[feedback_status])

    return demo


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":

    from observability.logging import setup_logging
    from observability.tracing import setup_tracing, setup_langsmith
    setup_logging(json_output=False)
    setup_tracing()
    setup_langsmith()

    demo = build_ui()
    demo.launch(
        server_name="localhost",
        server_port=7860,
        share=False,
        debug=True,
    )