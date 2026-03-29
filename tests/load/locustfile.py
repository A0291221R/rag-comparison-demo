"""
tests/load/locustfile.py — Load test for the RAG API.
Target: 100 RPS, P95 < 3s.

Run: locust -f tests/load/locustfile.py --host=http://localhost:8000
"""
from __future__ import annotations

import random

from locust import HttpUser, task, between

SAMPLE_QUERIES = [
    "What is retrieval-augmented generation?",
    "How does transformer attention work?",
    "Explain the difference between BERT and GPT",
    "What are the key components of a RAG pipeline?",
    "How do knowledge graphs improve information retrieval?",
    "What is fine-tuning in the context of language models?",
    "Describe multi-hop reasoning in graph databases",
    "What are vector embeddings and how are they used?",
    "Explain contrastive learning for embedding models",
    "How does hybrid search combine dense and sparse retrieval?",
]

API_KEY = "dev-secret-key"


class RAGUser(HttpUser):
    wait_time = between(0.5, 2.0)

    @task(3)
    def query_agentic(self):
        self.client.post(
            "/api/v1/query",
            json={
                "query": random.choice(SAMPLE_QUERIES),
                "pipeline": "agentic",
            },
            headers={"X-API-Key": API_KEY},
            name="POST /api/v1/query [agentic]",
        )

    @task(3)
    def query_graph(self):
        self.client.post(
            "/api/v1/query",
            json={
                "query": random.choice(SAMPLE_QUERIES),
                "pipeline": "graph",
            },
            headers={"X-API-Key": API_KEY},
            name="POST /api/v1/query [graph]",
        )

    @task(1)
    def query_parallel(self):
        self.client.post(
            "/api/v1/query",
            json={
                "query": random.choice(SAMPLE_QUERIES),
                "pipeline": "parallel",
            },
            headers={"X-API-Key": API_KEY},
            name="POST /api/v1/query [parallel]",
        )

    @task(1)
    def list_pipelines(self):
        self.client.get(
            "/api/v1/pipelines",
            headers={"X-API-Key": API_KEY},
            name="GET /api/v1/pipelines",
        )

    @task(1)
    def health_check(self):
        self.client.get("/health", name="GET /health")
