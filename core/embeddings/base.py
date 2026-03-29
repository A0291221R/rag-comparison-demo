"""
core/embeddings/base.py — Abstract EmbeddingProvider interface + concrete implementations.

Supports:
  - OpenAI text-embedding-3-small / large
  - SentenceTransformers (local / fine-tuned)
  - Fine-tuned adapter (version-tagged for rollback)
"""
from __future__ import annotations

import hashlib
import time
from abc import ABC, abstractmethod
from typing import Any

import structlog
from pydantic import BaseModel

logger = structlog.get_logger(__name__)


# ── Schema ────────────────────────────────────────────────────────────────────

class EmbeddingResult(BaseModel):
    vectors: list[list[float]]
    model_name: str
    model_version: str
    latency_ms: float
    token_count: int


# ── Abstract base ─────────────────────────────────────────────────────────────

class EmbeddingProvider(ABC):
    """All embedding providers must implement this interface."""

    model_name: str
    model_version: str
    dimension: int

    @abstractmethod
    async def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        """Embed a list of documents."""

    @abstractmethod
    async def embed_query(self, text: str) -> EmbeddingResult:
        """Embed a single query string (may use different instruction prefix)."""

    def fingerprint(self, text: str) -> str:
        """Stable cache key combining text hash + model version."""
        h = hashlib.sha256(f"{self.model_version}::{text}".encode()).hexdigest()[:16]
        return f"emb:{self.model_name}:{h}"


# ── OpenAI implementation ─────────────────────────────────────────────────────

class OpenAIEmbeddingProvider(EmbeddingProvider):
    """
    Wraps OpenAI text-embedding-3-* models.
    Handles batching and retries automatically via the openai client.
    """

    SUPPORTED_MODELS = {
        "text-embedding-3-small": {"dimension": 1536, "max_batch": 2048},
        "text-embedding-3-large": {"dimension": 3072, "max_batch": 2048},
        "text-embedding-ada-002": {"dimension": 1536, "max_batch": 2048},
    }

    def __init__(self, model_name: str = "text-embedding-3-small", version: str = "v1"):
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"Unsupported model: {model_name}")
        self.model_name = model_name
        self.model_version = version
        self.dimension = self.SUPPORTED_MODELS[model_name]["dimension"]
        self._max_batch = self.SUPPORTED_MODELS[model_name]["max_batch"]
        self._client: Any = None  # lazy init

    def _get_client(self) -> Any:
        if self._client is None:
            from openai import AsyncOpenAI
            from core.config import settings
            self._client = AsyncOpenAI(
                api_key=settings.openai_api_key.get_secret_value()
            )
        return self._client

    async def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        client = self._get_client()
        start = time.perf_counter()
        # batch in chunks to stay under API limits
        all_vectors: list[list[float]] = []
        total_tokens = 0
        for i in range(0, len(texts), self._max_batch):
            batch = texts[i : i + self._max_batch]
            response = await client.embeddings.create(
                model=self.model_name, input=batch
            )
            all_vectors.extend([d.embedding for d in response.data])
            total_tokens += response.usage.total_tokens

        latency_ms = (time.perf_counter() - start) * 1000
        logger.info(
            "embedded_documents",
            model=self.model_name,
            count=len(texts),
            tokens=total_tokens,
            latency_ms=round(latency_ms, 2),
        )
        return EmbeddingResult(
            vectors=all_vectors,
            model_name=self.model_name,
            model_version=self.model_version,
            latency_ms=latency_ms,
            token_count=total_tokens,
        )

    async def embed_query(self, text: str) -> EmbeddingResult:
        result = await self.embed_documents([text])
        return result


# ── SentenceTransformers (local / fine-tuned) ─────────────────────────────────

class SentenceTransformerProvider(EmbeddingProvider):
    """
    Local embedding using sentence-transformers.
    Used as offline fallback or for fine-tuned adapter.
    """

    def __init__(
        self,
        model_name: str = "sentence-transformers/all-MiniLM-L6-v2",
        version: str = "v1",
        device: str = "cpu",
    ):
        self.model_name = model_name
        self.model_version = version
        self.device = device
        self.dimension = 384  # default; updated after model load
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import SentenceTransformer  # type: ignore
            self._model = SentenceTransformer(self.model_name, device=self.device)
            self.dimension = self._model.get_sentence_embedding_dimension()
        return self._model

    async def embed_documents(self, texts: list[str]) -> EmbeddingResult:
        import asyncio
        model = self._get_model()
        start = time.perf_counter()
        loop = asyncio.get_event_loop()
        # Run CPU-bound work in thread pool
        vectors = await loop.run_in_executor(
            None, lambda: model.encode(texts, convert_to_numpy=True).tolist()
        )
        latency_ms = (time.perf_counter() - start) * 1000
        return EmbeddingResult(
            vectors=vectors,
            model_name=self.model_name,
            model_version=self.model_version,
            latency_ms=latency_ms,
            token_count=sum(len(t.split()) for t in texts),
        )

    async def embed_query(self, text: str) -> EmbeddingResult:
        return await self.embed_documents([text])


# ── Factory ───────────────────────────────────────────────────────────────────

def get_embedding_provider(use_finetuned: bool = False) -> EmbeddingProvider:
    """
    Factory: returns the right provider based on feature flags.
    Always returns a versioned provider for rollback support.
    """
    from core.config import settings

    if use_finetuned and settings.feature_finetuned_embeddings:
        return SentenceTransformerProvider(
            model_name=f"models/{settings.embedding_model_ft}",
            version=settings.embedding_model_ft,
        )
    return OpenAIEmbeddingProvider(
        model_name=settings.embedding_model,
        version="v1",
    )
