"""
core/config.py — Central settings loaded from environment / .env file.
All modules import from here; never read os.environ directly.
"""
from __future__ import annotations

from functools import lru_cache
from typing import Literal

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── LLM ──────────────────────────────────────────────────────────────────
    openai_api_key: SecretStr = Field(default="")
    anthropic_api_key: SecretStr = Field(default="")
    llm_classification_model: str = "gpt-4o-mini"
    llm_generation_model: str = "gpt-4o"
    embedding_model: str = "text-embedding-3-small"
    embedding_model_ft: str = "embed-ft-v1"

    # ── LangSmith ────────────────────────────────────────────────────────────
    langchain_tracing_v2: bool = True
    langchain_endpoint: str = "https://api.smith.langchain.com"
    langchain_api_key: SecretStr = Field(default="")
    langchain_project: str = "rag-comparison-demo"

    # ── Qdrant ───────────────────────────────────────────────────────────────
    qdrant_url: str = "http://localhost:6333"
    qdrant_api_key: str = ""
    qdrant_collection: str = "rag_demo"
    qdrant_collection_ft: str = "rag_demo_ft"

    # ── Neo4j ────────────────────────────────────────────────────────────────
    neo4j_uri: str = "bolt://localhost:7687"
    neo4j_user: str = "neo4j"
    neo4j_password: SecretStr = Field(default="password")
    neo4j_database: str = "neo4j"

    # ── Neptune (optional) ───────────────────────────────────────────────────
    neptune_endpoint: str = ""
    neptune_port: int = 8182
    neptune_use_iam: bool = False

    # ── Redis ────────────────────────────────────────────────────────────────
    redis_url: str = "redis://localhost:6379"
    redis_ttl_seconds: int = 3600
    semantic_cache_threshold: float = 0.95

    # ── Postgres ─────────────────────────────────────────────────────────────
    postgres_url: str = "postgresql://postgres:password@localhost:5432/rag_demo"

    # ── OpenTelemetry ────────────────────────────────────────────────────────
    otel_exporter_otlp_endpoint: str = "http://localhost:4317"
    otel_service_name: str = "rag-comparison-demo"

    # ── API ──────────────────────────────────────────────────────────────────
    api_key: str = "dev-secret-key"
    api_rate_limit_per_minute: int = 60
    api_port: int = 8000
    grpc_port: int = 50051

    # ── Pipeline Tuning ──────────────────────────────────────────────────────
    retrieval_top_k: int = 10
    rerank_top_k: int = 5
    reranking_enabled: bool = True
    relevance_threshold: float = 0.65
    max_agent_iterations: int = 3
    recursion_limit: int = 25
    token_budget_per_query: int = 8000
    node_timeout_seconds: int = 30

    # ── Feature Flags ────────────────────────────────────────────────────────
    feature_graphrag_enabled: bool = True
    feature_finetuned_embeddings: bool = False
    feature_web_search_fallback: bool = False
    feature_community_summaries: bool = False
    shadow_mode_enabled: bool = False


@lru_cache(maxsize=1)
def get_settings() -> Settings:
    """Return singleton settings instance."""
    return Settings()


# Module-level convenience alias
settings = get_settings()
