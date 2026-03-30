FROM python:3.11-slim-bookworm

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Copy full project first (editable install requires README.md + source tree)
COPY . .

# Install deps — explicit list avoids pyproject.toml editable-install issues in Docker
RUN pip install --no-cache-dir \
        "langchain>=0.3.0" \
        "langchain-core>=0.3.0" \
        "langchain-openai>=0.2.0" \
        "langchain-anthropic>=0.3.0" \
        "langgraph>=0.2.0" \
        "langsmith>=0.1.0" \
        "qdrant-client>=1.9.0" \
        "httpx>=0.27.0" \
        "pydantic>=2.7.0" \
        "pydantic-settings>=2.3.0" \
        "python-dotenv>=1.0.0" \
        "structlog>=24.2.0" \
        "tenacity>=8.3.0" \
        "tiktoken>=0.7.0" \
        "redis>=5.0.0" \
        "neo4j>=5.20.0" \
        "spacy>=3.7.0" \
        "networkx>=3.3" \
        "gradio>=4.31.0" \
        "plotly>=5.22.0" \
        "pandas>=2.2.0" \
        "opentelemetry-api>=1.24.0" \
        "opentelemetry-sdk>=1.24.0" \
        "opentelemetry-exporter-otlp>=1.24.0" \
        "prometheus-client>=0.20.0" \
        "sentence-transformers>=3.0.0" \
        "fastapi>=0.111.0" \
        "uvicorn[standard]>=0.29.0" \
        "slowapi>=0.1.9" \
    && python -m spacy download en_core_web_sm \
    && pip cache purge

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8000 50051 7860

CMD ["uvicorn", "api.rest.app:app", "--host", "0.0.0.0", "--port", "8000"]