"""
core/fallback/chain.py — Layered fallback chain.

Order: vector search → graph search → web search → "I don't know"
Configurable thresholds; each step is only tried if the previous fails.
"""
from __future__ import annotations

import time
from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable, Awaitable

import structlog

logger = structlog.get_logger(__name__)


class FallbackLevel(str, Enum):
    VECTOR = "vector"
    GRAPH = "graph"
    WEB = "web"
    NONE = "none"


@dataclass
class FallbackResult:
    answer: str
    level_used: FallbackLevel
    latency_ms: float
    triggered: bool


FallbackFn = Callable[[str], Awaitable[str | None]]


class FallbackChain:
    """
    Executes strategies in order, advancing to next if current returns None
    or raises an exception.

    Usage:
        chain = FallbackChain(settings)
        chain.register(FallbackLevel.VECTOR, vector_fn)
        chain.register(FallbackLevel.GRAPH, graph_fn)
        chain.register(FallbackLevel.WEB, web_fn)
        result = await chain.run(query)
    """

    DEFAULT_ANSWER = (
        "I don't have enough information to answer this question confidently. "
        "Please try rephrasing or providing more context."
    )

    def __init__(self) -> None:
        self._handlers: list[tuple[FallbackLevel, FallbackFn]] = []

    def register(self, level: FallbackLevel, fn: FallbackFn) -> "FallbackChain":
        self._handlers.append((level, fn))
        return self

    async def run(self, query: str) -> FallbackResult:
        start = time.perf_counter()
        for level, fn in self._handlers:
            try:
                result = await fn(query)
                if result:
                    latency_ms = (time.perf_counter() - start) * 1000
                    triggered = level != FallbackLevel.VECTOR
                    logger.info(
                        "",
                        level=level,
                        triggered=triggered,
                        latency_ms=round(latency_ms, 1),
                    )
                    return FallbackResult(
                        answer=result,
                        level_used=level,
                        latency_ms=latency_ms,
                        triggered=triggered,
                    )
            except Exception as exc:
                logger.warning("fallback_level_failed", level=level, error=str(exc))
                continue

        latency_ms = (time.perf_counter() - start) * 1000
        logger.warning("fallback_exhausted", query=query[:80])
        return FallbackResult(
            answer=self.DEFAULT_ANSWER,
            level_used=FallbackLevel.NONE,
            latency_ms=latency_ms,
            triggered=True,
        )
