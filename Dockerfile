FROM python:3.11-slim-bookworm

WORKDIR /app

# System dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential curl git \
    && rm -rf /var/lib/apt/lists/*

# Install Python dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir -e ".[api,ui,observability,graphrag]" \
    && python -m spacy download en_core_web_sm

# Copy source
COPY . .

# Non-root user
RUN useradd -m -u 1000 appuser && chown -R appuser /app
USER appuser

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONPATH=/app

EXPOSE 8000 50051 7860

# Default: run REST API (override in docker-compose for UI or gRPC)
CMD ["uvicorn", "api.rest.app:app", "--host", "0.0.0.0", "--port", "8000"]
