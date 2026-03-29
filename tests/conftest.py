"""
tests/conftest.py — Pytest configuration and shared fixtures.
"""
from __future__ import annotations

import os
import pytest

# Set test environment before any imports
os.environ.setdefault("OPENAI_API_KEY", "sk-test-key")
os.environ.setdefault("ANTHROPIC_API_KEY", "sk-ant-test-key")
os.environ.setdefault("LANGCHAIN_TRACING_V2", "false")
os.environ.setdefault("NEO4J_PASSWORD", "testpassword")
os.environ.setdefault("API_KEY", "dev-secret-key")
os.environ.setdefault("REDIS_URL", "redis://localhost:6379")
os.environ.setdefault("QDRANT_URL", "http://localhost:6333")
os.environ.setdefault("NEO4J_URI", "bolt://localhost:7687")


def pytest_configure(config):
    config.addinivalue_line("markers", "integration: marks tests requiring live services")
    config.addinivalue_line("markers", "eval: marks evaluation/benchmark tests")


@pytest.fixture(scope="session")
def event_loop_policy():
    """Use default asyncio event loop policy."""
    import asyncio
    return asyncio.DefaultEventLoopPolicy()
