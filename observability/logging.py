"""
observability/logging.py — Structured JSON logging via structlog.

Call setup_logging() at app startup.
All modules should: import structlog; logger = structlog.get_logger(__name__)
"""
from __future__ import annotations

import logging
import sys


def setup_logging(level: str = "INFO", json_output: bool = True) -> None:
    """
    Configure structlog for structured JSON logging.

    Log fields included on every entry:
      trace_id, pipeline, node, latency_ms, token_count, model_version
    """
    import structlog

    shared_processors: list = [
        structlog.contextvars.merge_contextvars,
        structlog.stdlib.add_log_level,
        structlog.stdlib.add_logger_name,
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
    ]

    if json_output:
        renderer = structlog.processors.JSONRenderer()
    else:
        renderer = structlog.dev.ConsoleRenderer(colors=True)  # type: ignore

    structlog.configure(
        processors=shared_processors + [
            structlog.stdlib.ProcessorFormatter.wrap_for_formatter,
        ],
        logger_factory=structlog.stdlib.LoggerFactory(),
        wrapper_class=structlog.stdlib.BoundLogger,
        cache_logger_on_first_use=True,
    )

    formatter = structlog.stdlib.ProcessorFormatter(
        processor=renderer,
        foreign_pre_chain=shared_processors,
    )

    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(formatter)

    root_logger = logging.getLogger()
    root_logger.handlers = [handler]
    root_logger.setLevel(getattr(logging, level.upper(), logging.INFO))

    # Silence noisy third-party loggers
    for noisy in ("httpx", "httpcore", "openai", "neo4j", "qdrant_client"):
        logging.getLogger(noisy).setLevel(logging.WARNING)
