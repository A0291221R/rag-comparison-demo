"""
core/retrieval/vector_store.py — Qdrant-backed retrieval with:
  - Dense + sparse (BM25) hybrid search via RRF fusion
  - Cross-encoder re-ranking
  - Redis semantic cache
  - Prometheus instrumentation
"""
from __future__ import annotations

import asyncio
import json
import time
from dataclasses import dataclass, field
from typing import Any

import structlog
from pydantic import BaseModel

logger = structlog.get_logger(__name__)


# ── Document schema ───────────────────────────────────────────────────────────

@dataclass
class Document:
    id: str
    content: str
    metadata: dict[str, Any] = field(default_factory=dict)
    score: float = 0.0
    embedding_model: str = ""
    source: str = ""


class RetrievalResult(BaseModel):
    documents: list[Document]
    query: str
    strategy: str          # "dense" | "hybrid" | "reranked"
    latency_ms: float
    cache_hit: bool = False


# ── Qdrant client wrapper ─────────────────────────────────────────────────────

class QdrantVectorStore:
    """
    Manages dense + sparse vectors in Qdrant.
    Supports named collections for base vs fine-tuned embeddings.
    """

    def __init__(self, collection_name: str | None = None):
        from core.config import settings
        self._settings = settings
        self._collection = collection_name or settings.qdrant_collection
        self._client: Any = None
        self._async_client: Any = None

    def _get_client(self) -> Any:
        if self._client is None:
            from qdrant_client import QdrantClient
            self._client = QdrantClient(
                url=self._settings.qdrant_url,
                api_key=self._settings.qdrant_api_key or None,
                check_compatibility=False,
            )
        return self._client

    def _get_async_client(self) -> Any:
        if self._async_client is None:
            from qdrant_client import AsyncQdrantClient
            self._async_client = AsyncQdrantClient(
                url=self._settings.qdrant_url,
                api_key=self._settings.qdrant_api_key or None,
                check_compatibility=False,
            )
        return self._async_client

    async def ensure_collection(self, dimension: int) -> None:
        """Create collection with HNSW config if it doesn't exist."""
        from qdrant_client.models import (
            Distance, HnswConfigDiff, VectorParams,
            SparseVectorParams, SparseIndexParams,
        )
        client = self._get_async_client()
        existing = await client.get_collections()
        names = [c.name for c in existing.collections]
        if self._collection not in names:
            await client.create_collection(
                collection_name=self._collection,
                vectors_config={
                    "dense": VectorParams(
                        size=dimension,
                        distance=Distance.COSINE,
                        hnsw_config=HnswConfigDiff(
                            m=16,
                            ef_construct=200,
                        ),
                    )
                },
                sparse_vectors_config={
                    "sparse": SparseVectorParams(
                        index=SparseIndexParams(on_disk=False)
                    )
                },
            )
            logger.info("created_qdrant_collection", collection=self._collection, dim=dimension)

    async def upsert(
        self,
        documents: list[Document],
        vectors: list[list[float]],
        sparse_vectors: list[dict[str, Any]] | None = None,
    ) -> None:
        """Insert or update documents with their vectors."""
        from qdrant_client.models import PointStruct, SparseVector
        client = self._get_async_client()
        points = []
        for i, (doc, vec) in enumerate(zip(documents, vectors)):
            payload = {
                "content": doc.content,
                "metadata": doc.metadata,
                "source": doc.source,
                "embedding_model": doc.embedding_model,
            }
            sparse = None
            if sparse_vectors and i < len(sparse_vectors):
                sv = sparse_vectors[i]
                sparse = {"sparse": SparseVector(
                    indices=sv["indices"], values=sv["values"]
                )}
            points.append(PointStruct(
                id=doc.id,
                vector={"dense": vec, **(sparse or {})},
                payload=payload,
            ))
        await client.upsert(collection_name=self._collection, points=points)

    async def dense_search(
        self, query_vector: list[float], top_k: int = 10
    ) -> list[Document]:
        from qdrant_client.models import SearchRequest
        client = self._get_async_client()
        results = await client.query_points(
            collection_name=self._collection,
            query=query_vector,
            using="dense",
            limit=top_k,
            with_payload=True,
        )
        return [
            Document(
                id=str(r.id),
                content=r.payload["content"],  # type: ignore
                metadata=r.payload.get("metadata", {}),  # type: ignore
                score=r.score,
                source=r.payload.get("source", ""),  # type: ignore
                embedding_model=r.payload.get("embedding_model", ""),  # type: ignore
            )
            for r in results.points
        ]

    async def hybrid_search(
        self,
        query_vector: list[float],
        sparse_query: dict[str, Any],
        top_k: int = 10,
    ) -> list[Document]:
        """RRF fusion of dense + sparse results."""
        from qdrant_client.models import (
            Prefetch, FusionQuery, Fusion,
            SparseVector, Query,
        )
        client = self._get_async_client()
        results = await client.query_points(
            collection_name=self._collection,
            prefetch=[
                Prefetch(
                    query=query_vector,
                    using="dense",
                    limit=top_k * 2,
                ),
                Prefetch(
                    query=SparseVector(
                        indices=sparse_query["indices"],
                        values=sparse_query["values"],
                    ),
                    using="sparse",
                    limit=top_k * 2,
                ),
            ],
            query=FusionQuery(fusion=Fusion.RRF),
            limit=top_k,
            with_payload=True,
        )
        return [
            Document(
                id=str(r.id),
                content=r.payload["content"],  # type: ignore
                metadata=r.payload.get("metadata", {}),  # type: ignore
                score=r.score,
                source=r.payload.get("source", ""),  # type: ignore
            )
            for r in results.points
        ]


# ── BM25 sparse vector helper ─────────────────────────────────────────────────

class BM25Encoder:
    """
    Lightweight BM25 for sparse vector generation.
    In production, use a pre-trained BM25 encoder or Qdrant's built-in sparse.
    """

    def __init__(self) -> None:
        self._idf: dict[str, float] = {}
        self._vocab: dict[str, int] = {}

    def fit(self, corpus: list[str]) -> None:
        import math
        from collections import Counter
        N = len(corpus)
        df: Counter[str] = Counter()
        for doc in corpus:
            tokens = set(doc.lower().split())
            df.update(tokens)
        self._idf = {t: math.log((N - c + 0.5) / (c + 0.5) + 1) for t, c in df.items()}
        self._vocab = {t: i for i, t in enumerate(sorted(self._idf.keys()))}

    def encode(self, text: str) -> dict[str, Any]:
        from collections import Counter
        tokens = text.lower().split()
        tf = Counter(tokens)
        indices, values = [], []
        for token, count in tf.items():
            if token in self._vocab:
                indices.append(self._vocab[token])
                values.append(float(count) * self._idf.get(token, 1.0))
        return {"indices": indices, "values": values}


# ── Cross-encoder re-ranker ───────────────────────────────────────────────────

class CrossEncoderReranker:
    """Re-rank retrieved documents using a cross-encoder model."""

    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model_name = model_name
        self._model: Any = None

    def _get_model(self) -> Any:
        if self._model is None:
            from sentence_transformers import CrossEncoder  # type: ignore
            self._model = CrossEncoder(self.model_name)
        return self._model

    async def rerank(
        self, query: str, documents: list[Document], top_k: int = 5
    ) -> list[Document]:
        loop = asyncio.get_event_loop()
        model = self._get_model()
        pairs = [(query, doc.content) for doc in documents]
        scores: list[float] = await loop.run_in_executor(
            None, lambda: model.predict(pairs).tolist()
        )
        for doc, score in zip(documents, scores):
            doc.score = score
        ranked = sorted(documents, key=lambda d: d.score, reverse=True)
        return ranked[:top_k]


# ── Main retriever orchestrating all of the above ────────────────────────────

class HybridRetriever:
    """
    Full retrieval pipeline:
      1. Dense search
      2. Sparse BM25 search
      3. RRF fusion (hybrid)
      4. Cross-encoder re-ranking
      5. Redis semantic cache
    """

    def __init__(
        self,
        vector_store: QdrantVectorStore | None = None,
        reranker: CrossEncoderReranker | None = None,
        use_reranking: bool = True,
    ):
        from core.config import settings
        self._settings = settings
        self._vs = vector_store or QdrantVectorStore()
        self._reranker = reranker or CrossEncoderReranker()
        self._use_reranking = use_reranking
        self._redis: Any = None
        self._bm25: BM25Encoder | None = None

    def _get_redis(self) -> Any:
        if self._redis is None:
            import redis.asyncio as aioredis
            self._redis = aioredis.from_url(self._settings.redis_url)
        return self._redis

    async def _check_cache(self, cache_key: str) -> list[Document] | None:
        try:
            r = self._get_redis()
            raw = await r.get(cache_key)
            if raw:
                data = json.loads(raw)
                return [Document(**d) for d in data]
        except Exception:
            pass
        return None

    async def _set_cache(self, cache_key: str, docs: list[Document]) -> None:
        try:
            r = self._get_redis()
            payload = json.dumps([
                {"id": d.id, "content": d.content, "metadata": d.metadata,
                 "score": d.score, "source": d.source, "embedding_model": d.embedding_model}
                for d in docs
            ])
            await r.setex(cache_key, self._settings.redis_ttl_seconds, payload)
        except Exception:
            pass

    async def retrieve(
        self,
        query: str,
        query_vector: list[float],
        top_k: int | None = None,
        use_hybrid: bool = True,
    ) -> RetrievalResult:
        from core.config import settings
        top_k = top_k or settings.retrieval_top_k
        rerank_k = settings.rerank_top_k
        start = time.perf_counter()
        cache_key = f"retrieval:{hashlib.sha256((query + str(top_k)).encode()).hexdigest()[:16]}"  # type: ignore

        # Cache check
        cached = await self._check_cache(cache_key)
        if cached:
            return RetrievalResult(
                documents=cached, query=query, strategy="cached",
                latency_ms=0.0, cache_hit=True,
            )

        # Dense retrieval
        docs = await self._vs.dense_search(query_vector, top_k=top_k)
        strategy = "dense"

        # Hybrid (add sparse) if BM25 is fitted
        if use_hybrid and self._bm25 is not None:
            sparse_vec = self._bm25.encode(query)
            docs = await self._vs.hybrid_search(
                query_vector, sparse_vec, top_k=top_k
            )
            strategy = "hybrid"

        # Re-ranking
        if self._use_reranking and docs:
            docs = await self._reranker.rerank(query, docs, top_k=rerank_k)
            strategy = "reranked"

        await self._set_cache(cache_key, docs)

        latency_ms = (time.perf_counter() - start) * 1000
        logger.info("retrieval_complete", strategy=strategy, k=len(docs), latency_ms=round(latency_ms, 1))

        return RetrievalResult(
            documents=docs, query=query, strategy=strategy, latency_ms=latency_ms
        )


import hashlib  # noqa: E402 (placed here to avoid circular)
