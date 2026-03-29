"""
observability/tracing.py — OpenTelemetry + LangSmith trace setup.

Call setup_tracing() once at application startup.
Use @trace_node decorator on LangGraph nodes for automatic span creation.
"""
from __future__ import annotations

import functools
import time
from typing import Any, Callable, TypeVar, Awaitable

import structlog

logger = structlog.get_logger(__name__)

F = TypeVar("F", bound=Callable[..., Awaitable[Any]])


def setup_tracing() -> None:
    """Initialize OpenTelemetry tracing pipeline."""
    try:
        from opentelemetry import trace
        from opentelemetry.sdk.trace import TracerProvider
        from opentelemetry.sdk.trace.export import BatchSpanProcessor
        from opentelemetry.exporter.otlp.proto.grpc.trace_exporter import OTLPSpanExporter
        from opentelemetry.sdk.resources import Resource
        from core.config import settings

        resource = Resource.create({"service.name": settings.otel_service_name})
        provider = TracerProvider(resource=resource)
        exporter = OTLPSpanExporter(endpoint=settings.otel_exporter_otlp_endpoint)
        provider.add_span_processor(BatchSpanProcessor(exporter))
        trace.set_tracer_provider(provider)
        logger.info("otel_tracing_initialized", endpoint=settings.otel_exporter_otlp_endpoint)
    except Exception as exc:
        logger.warning("otel_tracing_init_failed", error=str(exc))


def setup_langsmith() -> None:
    """Configure LangSmith tracing for LangGraph."""
    try:
        import os
        from core.config import settings
        if settings.langchain_tracing_v2:
            os.environ["LANGCHAIN_TRACING_V2"] = "true"
            os.environ["LANGCHAIN_ENDPOINT"] = settings.langchain_endpoint
            os.environ["LANGCHAIN_API_KEY"] = settings.langchain_api_key.get_secret_value()
            os.environ["LANGCHAIN_PROJECT"] = settings.langchain_project
            logger.info("langsmith_configured", project=settings.langchain_project)
    except Exception as exc:
        logger.warning("langsmith_setup_failed", error=str(exc))


def trace_node(node_name: str) -> Callable[[F], F]:
    """
    Decorator for LangGraph nodes: wraps execution in an OTel span.

    Usage:
        @trace_node("retrieve_node")
        async def retrieve_node(state: AgenticRAGState) -> AgenticRAGState:
            ...
    """
    def decorator(fn: F) -> F:
        @functools.wraps(fn)
        async def wrapper(*args: Any, **kwargs: Any) -> Any:
            try:
                from opentelemetry import trace
                tracer = trace.get_tracer("rag-comparison-demo")
            except ImportError:
                return await fn(*args, **kwargs)

            # Extract trace_id from state if present
            state = args[0] if args else {}
            trace_id = state.get("trace_id", "unknown") if isinstance(state, dict) else "unknown"

            with tracer.start_as_current_span(node_name) as span:
                span.set_attribute("rag.node", node_name)
                span.set_attribute("rag.trace_id", trace_id)
                start = time.perf_counter()
                try:
                    result = await fn(*args, **kwargs)
                    span.set_attribute("rag.success", True)
                    return result
                except Exception as exc:
                    span.set_attribute("rag.success", False)
                    span.set_attribute("rag.error", str(exc))
                    span.record_exception(exc)
                    raise
                finally:
                    span.set_attribute(
                        "rag.latency_ms",
                        round((time.perf_counter() - start) * 1000, 1)
                    )
        return wrapper  # type: ignore
    return decorator
