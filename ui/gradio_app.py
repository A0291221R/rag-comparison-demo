"""
ui/gradio_app.py — Side-by-side Agentic RAG vs GraphRAG comparison dashboard.

Layout:
  - Shared query input + pipeline selector
  - Two-column answer panels
  - Comparison metrics table
  - Graph subgraph visualizer (GraphRAG)
  - Node timing breakdown bar chart
  - Full trace JSON expander
  - Feedback buttons
"""
from __future__ import annotations

import asyncio
import json
import time
from typing import Any, Generator

import gradio as gr
import plotly.graph_objects as go

from core.config import settings


# ── Async runner helper ───────────────────────────────────────────────────────

def run_async(coro: Any) -> Any:
    """Run an async coroutine from sync Gradio handlers."""
    try:
        loop = asyncio.get_event_loop()
        if loop.is_running():
            import concurrent.futures
            with concurrent.futures.ThreadPoolExecutor() as pool:
                future = pool.submit(asyncio.run, coro)
                return future.result(timeout=120)
        return loop.run_until_complete(coro)
    except RuntimeError:
        return asyncio.run(coro)


# ── Core query runner ──────────────────────────────────────────────────────────

def run_query(
    query: str,
    pipeline_mode: str,
    session_id: str,
) -> tuple[str, str, str, Any, Any, str, str]:
    """
    Execute the query and return:
      agentic_answer, graph_answer, metrics_md, timing_chart, graph_viz,
      agentic_trace_json, graph_trace_json
    """
    if not query.strip():
        return ("Please enter a query.", "", "", None, None, "{}", "{}")

    from orchestrator.graph import get_orchestrator

    mode_map = {
        "🤖 Agentic RAG Only": "agentic",
        "🕸️ GraphRAG Only": "graph",
        "⚡ Both (Parallel)": "parallel",
        "🎯 Auto-Route": "auto",
    }
    mode = mode_map.get(pipeline_mode, "auto")

    start = time.perf_counter()
    orchestrator = get_orchestrator()
    try:
        result = run_async(orchestrator.run(query=query, mode=mode, session_id=session_id or None))
    except Exception as exc:
        error_msg = f"❌ Error: {exc}"
        return (error_msg, error_msg, "", None, None, "{}", "{}")

    total_ms = (time.perf_counter() - start) * 1000

    # Extract pipeline results
    agentic = result.get("agentic_result") or {}
    graph = result.get("graph_result") or {}
    metrics = result.get("comparison_metrics") or {}

    agentic_answer = agentic.get("final_answer") or result.get("final_answer", "—")
    graph_answer = graph.get("final_answer") or (result.get("final_answer", "—") if mode in ("graph", "auto") else "—")

    # ── Metrics markdown ──────────────────────────────────────────────────────
    def fmt_timings(timings: dict) -> str:
        if not timings:
            return "—"
        return " | ".join(f"`{k}`: {v:.0f}ms" for k, v in timings.items())

    a_metrics = metrics.get("agentic", {})
    g_metrics = metrics.get("graph", {})
    a_tokens = a_metrics.get("token_usage", {}) or agentic.get("token_usage", {})
    g_tokens = g_metrics.get("token_usage", {}) or graph.get("token_usage", {})

    metrics_md = f"""
| Metric | Agentic RAG | GraphRAG |
|--------|------------|---------|
| **Total latency** | {sum(a_metrics.get('node_timings', {}).values()):.0f} ms | {sum(g_metrics.get('node_timings', {}).values()):.0f} ms |
| **Tokens (total)** | {a_tokens.get('total', '—')} | {g_tokens.get('total', '—')} |
| **Retrieval count** | {a_metrics.get('retrieval_count', '—')} | {g_metrics.get('graph_nodes', '—')} nodes |
| **Fallback** | {'✅ Yes' if a_metrics.get('fallback_triggered') else '❌ No'} | — |
| **Entities extracted** | — | {', '.join(g_metrics.get('entities_extracted', [])[:5]) or '—'} |
| **Graph edges** | — | {g_metrics.get('graph_edges', '—')} |
| **Iterations** | {a_metrics.get('iterations', 0)} | — |
| **Wall clock** | {total_ms:.0f} ms | {total_ms:.0f} ms |
"""

    # ── Timing bar chart (Plotly) ──────────────────────────────────────────────
    a_timings = a_metrics.get("node_timings") or agentic.get("node_timings", {})
    g_timings = g_metrics.get("node_timings") or graph.get("node_timings", {})

    all_nodes = sorted(set(list(a_timings.keys()) + list(g_timings.keys())))
    timing_fig = go.Figure(data=[
        go.Bar(
            name="Agentic RAG",
            x=all_nodes,
            y=[a_timings.get(n, 0) for n in all_nodes],
            marker_color="#6366f1",
        ),
        go.Bar(
            name="GraphRAG",
            x=all_nodes,
            y=[g_timings.get(n, 0) for n in all_nodes],
            marker_color="#10b981",
        ),
    ])
    timing_fig.update_layout(
        title="Node Latency Breakdown (ms)",
        barmode="group",
        xaxis_tickangle=-30,
        yaxis_title="Latency (ms)",
        plot_bgcolor="#1e1e2e",
        paper_bgcolor="#1e1e2e",
        font_color="#cdd6f4",
        legend=dict(orientation="h", yanchor="bottom", y=1.02),
        height=300,
        margin=dict(l=40, r=20, t=60, b=80),
    )

    # ── Graph subgraph visualizer (pyvis → HTML) ───────────────────────────────
    subgraph = graph.get("graph_subgraph") or {}
    graph_html = _build_graph_html(subgraph)

    # ── Trace JSON ─────────────────────────────────────────────────────────────
    agentic_trace = json.dumps(
        {k: v for k, v in agentic.items() if k != "retrieved_chunks"}, indent=2, default=str
    )
    graph_trace = json.dumps(
        {k: v for k, v in graph.items() if k != "retrieved_chunks"}, indent=2, default=str
    )

    return (
        agentic_answer,
        graph_answer,
        metrics_md,
        timing_fig,
        graph_html,
        agentic_trace,
        graph_trace,
    )


def _build_graph_html(subgraph: dict) -> str:
    """Generate an interactive pyvis network graph HTML string."""
    nodes = subgraph.get("nodes", [])
    edges = subgraph.get("edges", [])

    if not nodes:
        return "<p style='color:#888; text-align:center; padding:40px'>No graph data available for this query.</p>"

    try:
        from pyvis.network import Network  # type: ignore
        net = Network(height="400px", width="100%", bgcolor="#1e1e2e", font_color="#cdd6f4")
        net.set_options("""
        {
          "nodes": {"shape": "dot", "size": 12, "font": {"size": 11}},
          "edges": {"smooth": {"type": "dynamic"}},
          "physics": {"stabilization": {"iterations": 100}}
        }
        """)
        color_map = {"Entity": "#6366f1", "Chunk": "#10b981", "Document": "#f59e0b"}
        for node in nodes[:50]:
            nid = node.get("id", "")
            label = (node.get("name") or nid)[:30]
            color = color_map.get(node.get("label", ""), "#94a3b8")
            net.add_node(nid, label=label, color=color, title=node.get("name", ""))

        for edge in edges[:100]:
            src, tgt = edge.get("source"), edge.get("target")
            if src and tgt:
                net.add_edge(src, tgt, title=edge.get("type", ""), label=edge.get("type", ""))

        return net.generate_html()
    except ImportError:
        # Fallback: plain text summary
        node_list = "\n".join(
            f"• [{n.get('label', '?')}] {n.get('name', n.get('id', ''))}"
            for n in nodes[:20]
        )
        edge_list = "\n".join(
            f"  {e.get('source')} —[{e.get('type')}]→ {e.get('target')}"
            for e in edges[:20]
        )
        return f"""
<div style='font-family:monospace; padding:16px; background:#1e1e2e; color:#cdd6f4; border-radius:8px'>
<b>Graph Nodes ({len(nodes)}):</b><br><pre>{node_list}</pre>
<b>Graph Edges ({len(edges)}):</b><br><pre>{edge_list}</pre>
</div>
"""


def submit_feedback(trace_json: str, rating: str) -> str:
    """Send feedback to the API."""
    import requests
    try:
        trace = json.loads(trace_json) if trace_json else {}
        trace_id = trace.get("trace_id", "unknown")
        requests.post(
            f"http://localhost:{settings.api_port}/api/v1/feedback",
            json={"trace_id": trace_id, "rating": rating},
            headers={"X-API-Key": settings.api_key},
            timeout=5,
        )
        return f"✅ Feedback ({rating}) submitted for trace {trace_id[:8]}…"
    except Exception as exc:
        return f"⚠️ Could not submit feedback: {exc}"


# ── Gradio UI layout ───────────────────────────────────────────────────────────

def build_ui() -> gr.Blocks:
    with gr.Blocks(
        title="Agentic RAG vs GraphRAG",
        theme=gr.themes.Base(
            primary_hue="violet",
            secondary_hue="emerald",
            neutral_hue="slate",
        ),
        css="""
        .answer-box { min-height: 200px; }
        .pipeline-label { font-weight: 700; font-size: 1.1em; }
        footer { display: none !important; }
        """,
    ) as demo:

        # ── Header ─────────────────────────────────────────────────────────────
        gr.Markdown(
            """
            # 🔍 Agentic RAG vs GraphRAG — Side-by-Side Comparison
            Compare two production-grade retrieval pipelines on any query.
            **Agentic RAG**: query rewriting, relevance grading, self-reflection loop.
            **GraphRAG**: entity extraction → Neo4j multi-hop traversal → context-grounded generation.
            """
        )

        # ── Controls ──────────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column(scale=4):
                query_input = gr.Textbox(
                    label="Query",
                    placeholder="e.g. How do transformers relate to attention mechanisms?",
                    lines=2,
                    elem_id="query-input",
                )
            with gr.Column(scale=2):
                pipeline_selector = gr.Radio(
                    choices=[
                        "🎯 Auto-Route",
                        "⚡ Both (Parallel)",
                        "🤖 Agentic RAG Only",
                        "🕸️ GraphRAG Only",
                    ],
                    value="⚡ Both (Parallel)",
                    label="Pipeline Mode",
                )

        with gr.Row():
            session_id_input = gr.Textbox(
                label="Session ID (optional)",
                placeholder="Leave blank for auto",
                scale=2,
            )
            run_btn = gr.Button("▶  Run Query", variant="primary", scale=1)
            clear_btn = gr.Button("🗑  Clear", scale=1)

        # ── Example queries ───────────────────────────────────────────────────
        gr.Examples(
            examples=[
                ["How are transformers and attention mechanisms related?", "⚡ Both (Parallel)"],
                ["What are the key differences between BERT and GPT architectures?", "🎯 Auto-Route"],
                ["Which researchers co-authored papers on graph neural networks?", "🕸️ GraphRAG Only"],
                ["Summarize the main findings on retrieval-augmented generation", "🤖 Agentic RAG Only"],
            ],
            inputs=[query_input, pipeline_selector],
            label="Example Queries",
        )

        # ── Answer panels ──────────────────────────────────────────────────────
        with gr.Row():
            with gr.Column():
                gr.Markdown("### 🤖 Agentic RAG", elem_classes=["pipeline-label"])
                agentic_answer = gr.Markdown(
                    value="*Run a query to see results…*",
                    elem_classes=["answer-box"],
                )
            with gr.Column():
                gr.Markdown("### 🕸️ GraphRAG", elem_classes=["pipeline-label"])
                graph_answer = gr.Markdown(
                    value="*Run a query to see results…*",
                    elem_classes=["answer-box"],
                )

        # ── Metrics table ──────────────────────────────────────────────────────
        with gr.Accordion("📊 Comparison Metrics", open=True):
            metrics_table = gr.Markdown(value="*Metrics will appear after first query.*")

        # ── Timing chart ──────────────────────────────────────────────────────
        with gr.Accordion("⏱ Node Latency Breakdown", open=False):
            timing_chart = gr.Plot()

        # ── Graph visualizer ──────────────────────────────────────────────────
        with gr.Accordion("🕸️ GraphRAG Subgraph Visualization", open=False):
            graph_viz = gr.HTML(value="<p style='color:#888; padding:16px'>Run a query to visualize the knowledge subgraph.</p>")

        # ── Trace JSON ─────────────────────────────────────────────────────────
        with gr.Accordion("🔎 Full Execution Traces (JSON)", open=False):
            with gr.Row():
                with gr.Column():
                    gr.Markdown("**Agentic RAG Trace**")
                    agentic_trace = gr.JSON()
                with gr.Column():
                    gr.Markdown("**GraphRAG Trace**")
                    graph_trace = gr.JSON()

        # ── Feedback ──────────────────────────────────────────────────────────
        with gr.Row():
            gr.Markdown("### 📣 Feedback")
        with gr.Row():
            thumbs_up_btn = gr.Button("👍 Helpful", variant="secondary")
            thumbs_down_btn = gr.Button("👎 Not Helpful", variant="secondary")
            feedback_status = gr.Textbox(label="", interactive=False, max_lines=1)

        # State for trace IDs
        last_agentic_trace = gr.State("{}")
        last_graph_trace = gr.State("{}")

        # ── Event handlers ─────────────────────────────────────────────────────
        def on_run(query: str, pipeline: str, session_id: str) -> tuple:
            results = run_query(query, pipeline, session_id)
            a_ans, g_ans, metrics, timing, gviz, a_trace_str, g_trace_str = results
            try:
                a_trace = json.loads(a_trace_str)
                g_trace = json.loads(g_trace_str)
            except Exception:
                a_trace, g_trace = {}, {}
            return (
                a_ans, g_ans, metrics, timing, gviz,
                a_trace, g_trace,
                a_trace_str, g_trace_str,
            )

        run_btn.click(
            fn=on_run,
            inputs=[query_input, pipeline_selector, session_id_input],
            outputs=[
                agentic_answer, graph_answer, metrics_table,
                timing_chart, graph_viz,
                agentic_trace, graph_trace,
                last_agentic_trace, last_graph_trace,
            ],
        )
        query_input.submit(
            fn=on_run,
            inputs=[query_input, pipeline_selector, session_id_input],
            outputs=[
                agentic_answer, graph_answer, metrics_table,
                timing_chart, graph_viz,
                agentic_trace, graph_trace,
                last_agentic_trace, last_graph_trace,
            ],
        )
        clear_btn.click(
            fn=lambda: ("", "", "*Metrics will appear after first query.*", None,
                        "<p style='color:#888;padding:16px'>Run a query to visualize the subgraph.</p>",
                        {}, {}, "{}", "{}"),
            inputs=[],
            outputs=[
                agentic_answer, graph_answer, metrics_table,
                timing_chart, graph_viz,
                agentic_trace, graph_trace,
                last_agentic_trace, last_graph_trace,
            ],
        )
        thumbs_up_btn.click(
            fn=lambda t: submit_feedback(t, "thumbs_up"),
            inputs=[last_agentic_trace],
            outputs=[feedback_status],
        )
        thumbs_down_btn.click(
            fn=lambda t: submit_feedback(t, "thumbs_down"),
            inputs=[last_agentic_trace],
            outputs=[feedback_status],
        )

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
        server_name="0.0.0.0",
        server_port=7860,
        share=False,
        show_error=True,
    )
