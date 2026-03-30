"""
api/rest/app.py — FastAPI REST API for the RAG comparison system.

Endpoints:
  POST /api/v1/query       — run query through selected pipeline(s)
  GET  /api/v1/pipelines   — list available pipelines + status
  GET  /api/v1/traces/{id} — fetch execution trace
  POST /api/v1/feedback    — RLHF-style feedback

Auth: API key in X-API-Key header
Rate limiting: 60 req/min via slowapi + Redis
"""
from __future__ import annotations

import time
import uuid
from typing import Any, Literal

import structlog
from fastapi import FastAPI, HTTPException, Depends, Header, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
from slowapi import Limiter, _rate_limit_exceeded_handler
from slowapi.errors import RateLimitExceeded
from slowapi.util import get_remote_address

from core.config import settings
# Force metrics registration at startup — must import before first request
import observability.metrics  # noqa: F401

# Initialize OTel tracing
from observability.tracing import setup_tracing, setup_langsmith
setup_tracing()
setup_langsmith()

logger = structlog.get_logger(__name__)

# ── Rate limiter ───────────────────────────────────────────────────────────────
limiter = Limiter(key_func=get_remote_address, storage_uri=settings.redis_url)

# ── App ────────────────────────────────────────────────────────────────────────
app = FastAPI(
    title="RAG Comparison Demo API",
    description="Agentic RAG vs GraphRAG side-by-side comparison",
    version="0.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
    openapi_url="/api/openapi.json",
)
app.state.limiter = limiter
app.add_exception_handler(RateLimitExceeded, _rate_limit_exceeded_handler)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# In-memory trace store (replace with Redis/Postgres in production)
_trace_store: dict[str, Any] = {}


# ── Auth ──────────────────────────────────────────────────────────────────────

async def verify_api_key(x_api_key: str = Header(..., alias="X-API-Key")) -> str:
    if x_api_key != settings.api_key:
        raise HTTPException(status_code=401, detail="Invalid API key")
    return x_api_key


# ── Request/Response schemas ───────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., min_length=1, max_length=2000)
    pipeline: Literal["auto", "agentic", "graph", "parallel"] = "auto"
    session_id: str | None = None


class QueryResponse(BaseModel):
    trace_id: str
    pipeline_used: str
    final_answer: str
    sources: list[dict[str, Any]] = []
    comparison_metrics: dict[str, Any] = {}
    token_usage: dict[str, int] = {}
    latency_ms: float


class FeedbackRequest(BaseModel):
    trace_id: str
    rating: Literal["thumbs_up", "thumbs_down"]
    comment: str | None = None


class PipelineStatus(BaseModel):
    name: str
    enabled: bool
    description: str


# ── Middleware: correlation ID ─────────────────────────────────────────────────

@app.middleware("http")
async def add_correlation_id(request: Request, call_next: Any) -> Any:
    correlation_id = request.headers.get("X-Correlation-ID", str(uuid.uuid4()))
    structlog.contextvars.bind_contextvars(correlation_id=correlation_id)
    response = await call_next(request)
    response.headers["X-Correlation-ID"] = correlation_id
    return response


# ── Health ─────────────────────────────────────────────────────────────────────

@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok", "version": "0.1.0"}


# ── Endpoints ─────────────────────────────────────────────────────────────────

@app.post("/api/v1/query", response_model=QueryResponse)
@limiter.limit(f"{settings.api_rate_limit_per_minute}/minute")
async def query_endpoint(
    request: Request,
    body: QueryRequest,
    _api_key: str = Depends(verify_api_key),
) -> QueryResponse:
    """Run a query through the selected RAG pipeline(s)."""
    start = time.perf_counter()

    # Import here to avoid circular imports at module load
    from orchestrator.graph import get_orchestrator

    orchestrator = get_orchestrator()
    try:
        result = await orchestrator.run(
            query=body.query,
            mode=body.pipeline,
            session_id=body.session_id,
        )
    except Exception as exc:
        logger.error("query_endpoint_error", error=str(exc))
        raise HTTPException(status_code=500, detail=f"Pipeline error: {exc}") from exc

    latency_ms = (time.perf_counter() - start) * 1000
    trace_id = result.get("trace_id", str(uuid.uuid4()))

    # Store trace for later retrieval
    _trace_store[trace_id] = {
        "trace_id": trace_id,
        "query": body.query,
        "pipeline": result.get("pipeline_used"),
        "agentic_result": result.get("agentic_result"),
        "graph_result": result.get("graph_result"),
        "comparison_metrics": result.get("comparison_metrics", {}),
        "token_usage": result.get("token_usage", {}),
        "timestamp": time.time(),
    }

    # Extract sources from whichever pipeline ran
    sources: list[dict[str, Any]] = []
    if result.get("agentic_result"):
        sources = result["agentic_result"].get("sources", [])  # type: ignore
    if result.get("graph_result"):
        sources = result["graph_result"].get("sources", [])  # type: ignore

    return QueryResponse(
        trace_id=trace_id,
        pipeline_used=result.get("pipeline_used", "unknown"),
        final_answer=result.get("final_answer", ""),
        sources=sources,
        comparison_metrics=result.get("comparison_metrics", {}),
        token_usage=result.get("token_usage", {}),
        latency_ms=round(latency_ms, 1),
    )


@app.get("/api/v1/pipelines", response_model=list[PipelineStatus])
async def list_pipelines(
    _api_key: str = Depends(verify_api_key),
) -> list[PipelineStatus]:
    return [
        PipelineStatus(
            name="agentic_rag",
            enabled=True,
            description="Agentic RAG with query rewriting, relevance grading, and self-reflection",
        ),
        PipelineStatus(
            name="graph_rag",
            enabled=settings.feature_graphrag_enabled,
            description="GraphRAG with entity extraction and Neo4j multi-hop traversal",
        ),
        PipelineStatus(
            name="parallel",
            enabled=settings.feature_graphrag_enabled,
            description="Run both pipelines simultaneously for side-by-side comparison",
        ),
    ]


@app.get("/api/v1/traces/{trace_id}")
async def get_trace(
    trace_id: str,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, Any]:
    trace = _trace_store.get(trace_id)
    if not trace:
        raise HTTPException(status_code=404, detail=f"Trace {trace_id} not found")
    return trace


@app.post("/api/v1/feedback")
async def submit_feedback(
    body: FeedbackRequest,
    _api_key: str = Depends(verify_api_key),
) -> dict[str, str]:
    """Collect RLHF-style feedback for a query response."""
    logger.info(
        "feedback_received",
        trace_id=body.trace_id,
        rating=body.rating,
        comment=body.comment,
    )
    # TODO: persist to Postgres for RLHF pipeline
    return {"status": "ok", "trace_id": body.trace_id}


# ── Prometheus metrics endpoint ────────────────────────────────────────────────

@app.get("/metrics")
async def metrics() -> Any:
    from prometheus_client import generate_latest, CONTENT_TYPE_LATEST
    from fastapi.responses import Response
    return Response(content=generate_latest(), media_type=CONTENT_TYPE_LATEST)
