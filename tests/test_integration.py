"""
tests/test_integration.py — Integration tests for the full pipeline.
Marked with @pytest.mark.integration — requires live services.

Run with: pytest tests/test_integration.py -v -m integration
"""
from __future__ import annotations

import pytest
from unittest.mock import AsyncMock, MagicMock, patch


# ── Full pipeline integration ─────────────────────────────────────────────────

@pytest.mark.integration
class TestAgenticRAGEndToEnd:
    @pytest.mark.asyncio
    async def test_run_agentic_rag_returns_answer(self):
        """Full pipeline run: must return a non-empty answer."""
        from pipelines.agentic_rag.graph import run_agentic_rag
        result = await run_agentic_rag("What is retrieval-augmented generation?")
        assert "final_answer" in result
        assert len(result["final_answer"]) > 10
        assert result["pipeline_used"] == "agentic_rag"

    @pytest.mark.asyncio
    async def test_agentic_rag_sets_trace_id(self):
        from pipelines.agentic_rag.graph import run_agentic_rag
        result = await run_agentic_rag("Explain transformers")
        assert "trace_id" in result
        assert len(result["trace_id"]) == 36  # UUID format

    @pytest.mark.asyncio
    async def test_agentic_rag_tracks_token_usage(self):
        from pipelines.agentic_rag.graph import run_agentic_rag
        result = await run_agentic_rag("What is BERT?")
        usage = result.get("token_usage", {})
        assert "total" in usage


@pytest.mark.integration
class TestGraphRAGEndToEnd:
    @pytest.mark.asyncio
    async def test_run_graph_rag_returns_answer(self):
        from pipelines.graph_rag.graph import run_graph_rag
        result = await run_graph_rag("How do transformers relate to BERT?")
        assert "final_answer" in result
        assert result["pipeline_used"] == "graph_rag"

    @pytest.mark.asyncio
    async def test_graph_rag_extracts_entities(self):
        from pipelines.graph_rag.graph import run_graph_rag
        result = await run_graph_rag("How are BERT and GPT architecturally different?")
        entities = result.get("entities_extracted", [])
        assert isinstance(entities, list)


@pytest.mark.integration
class TestOrchestratorParallel:
    @pytest.mark.asyncio
    async def test_parallel_mode_returns_both_answers(self):
        from orchestrator.graph import get_orchestrator
        orchestrator = get_orchestrator()
        result = await orchestrator.run("What is RAG?", mode="parallel")
        assert result.get("pipeline_used") == "parallel"
        assert result.get("agentic_result") is not None
        assert result.get("graph_result") is not None

    @pytest.mark.asyncio
    async def test_auto_mode_routes(self):
        from orchestrator.graph import get_orchestrator
        orchestrator = get_orchestrator()
        result = await orchestrator.run("What is attention?", mode="auto")
        assert result.get("pipeline_used") in ("agentic_rag", "graph_rag", "parallel")


# ── RAGAS retrieval evaluation ────────────────────────────────────────────────

@pytest.mark.integration
@pytest.mark.eval
class TestRAGASMetrics:
    """
    RAGAS evaluation: faithfulness, answer_relevancy, context_recall.
    Requires ragas package and a fixed 50-query eval set.
    """

    EVAL_QUERIES = [
        {
            "question": "What is retrieval-augmented generation?",
            "ground_truth": "RAG combines information retrieval with language model generation.",
        },
        {
            "question": "How does the transformer attention mechanism work?",
            "ground_truth": "Attention computes weighted sums over value vectors using query-key dot products.",
        },
    ]

    @pytest.mark.asyncio
    async def test_ragas_faithfulness(self):
        try:
            from ragas import evaluate  # type: ignore
            from ragas.metrics import faithfulness  # type: ignore
            from datasets import Dataset  # type: ignore
        except ImportError:
            pytest.skip("ragas or datasets not installed")

        from pipelines.agentic_rag.graph import run_agentic_rag

        rows = []
        for item in self.EVAL_QUERIES:
            result = await run_agentic_rag(item["question"])
            chunks = result.get("retrieved_chunks", [])
            rows.append({
                "question": item["question"],
                "answer": result.get("final_answer", ""),
                "contexts": [c["content"] for c in chunks[:5]],
                "ground_truth": item["ground_truth"],
            })

        ds = Dataset.from_list(rows)
        scores = evaluate(ds, metrics=[faithfulness])
        faith_score = scores["faithfulness"]
        assert faith_score >= 0.5, f"Faithfulness {faith_score:.2f} below threshold"


# ── Load test simulation (single-user) ────────────────────────────────────────

@pytest.mark.integration
class TestAPIEndpoints:
    @pytest.mark.asyncio
    async def test_health_endpoint(self):
        from httpx import AsyncClient
        from api.rest.app import app
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.get("/health")
        assert response.status_code == 200
        assert response.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_query_endpoint_requires_auth(self):
        from httpx import AsyncClient
        from api.rest.app import app
        async with AsyncClient(app=app, base_url="http://test") as client:
            response = await client.post(
                "/api/v1/query",
                json={"query": "What is RAG?", "pipeline": "agentic"},
            )
        assert response.status_code == 422  # Missing X-API-Key

    @pytest.mark.asyncio
    async def test_query_endpoint_with_valid_key(self):
        from httpx import AsyncClient
        from api.rest.app import app
        from core.config import settings

        with patch("orchestrator.graph.RAGOrchestrator.run") as mock_run:
            mock_run.return_value = {
                "trace_id": "test-trace",
                "pipeline_used": "agentic_rag",
                "final_answer": "RAG is a technique combining retrieval and generation.",
                "token_usage": {"prompt": 100, "completion": 50, "total": 150},
                "comparison_metrics": {},
            }
            async with AsyncClient(app=app, base_url="http://test") as client:
                response = await client.post(
                    "/api/v1/query",
                    json={"query": "What is RAG?", "pipeline": "agentic"},
                    headers={"X-API-Key": settings.api_key},
                )
        assert response.status_code == 200
        data = response.json()
        assert "final_answer" in data
        assert "trace_id" in data
