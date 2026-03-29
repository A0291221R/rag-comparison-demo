# RAG Comparison Demo
### Agentic RAG vs GraphRAG — Production-Grade Side-by-Side Comparison

[![Python 3.11+](https://img.shields.io/badge/python-3.11+-blue.svg)](https://www.python.org/)
[![LangGraph](https://img.shields.io/badge/LangGraph-0.2+-purple.svg)](https://github.com/langchain-ai/langgraph)
[![Neo4j](https://img.shields.io/badge/Neo4j-5.x-green.svg)](https://neo4j.com/)

---

## Architecture

```
┌─────────────────────────────────────────────────────────────────────┐
│                          Gradio UI (port 7860)                      │
│               Side-by-side answer + metrics + subgraph viz          │
└─────────────────────────┬───────────────────────────────────────────┘
                           │
┌─────────────────────────▼───────────────────────────────────────────┐
│                     FastAPI REST / gRPC API                          │
│               Auth · Rate Limiting · Trace Store                     │
└─────────────────────────┬───────────────────────────────────────────┘
                           │
┌─────────────────────────▼───────────────────────────────────────────┐
│                      LangGraph Orchestrator                          │
│         auto-route · parallel mode · token budget · timeouts        │
└──────────────┬──────────────────────────────────┬───────────────────┘
               │                                  │
┌──────────────▼──────────────┐    ┌──────────────▼──────────────────┐
│       Agentic RAG Graph     │    │         GraphRAG Graph           │
│                             │    │                                  │
│  guardrails                 │    │  guardrails                      │
│     → query_rewrite         │    │    → entity_extract              │
│     → retrieve              │    │    → graph_traverse (Neo4j)      │
│     → grade_relevance       │    │    → community_summary           │
│     → [fallback]            │    │    → chunk_retrieve              │
│     → generate              │    │    → generate                    │
│     → self_reflect          │    │    → output_guardrails           │
│     → output_guardrails     │    │                                  │
└──────────────┬──────────────┘    └──────────────┬───────────────────┘
               │                                  │
   ┌───────────▼──────────────────────────────────▼───────────┐
   │                    Data Layer                             │
   │  Qdrant (vectors)  ·  Neo4j (graph)  ·  Redis (cache)   │
   └───────────────────────────────────────────────────────────┘
               │
   ┌───────────▼──────────────────────────────────────────────┐
   │               Observability Stack                         │
   │  LangSmith (traces)  ·  OTel → Jaeger  ·  Prometheus    │
   │  Grafana dashboards  ·  Structured JSON logs (structlog) │
   └───────────────────────────────────────────────────────────┘
```

---

## Quickstart

### Prerequisites
- Docker + Docker Compose
- Python 3.11+
- OpenAI API key (or Anthropic)

### 1. Clone and configure

```bash
git clone https://github.com/yourorg/rag-comparison-demo
cd rag-comparison-demo
cp .env.example .env
# Edit .env and set OPENAI_API_KEY, etc.
```

### 2. Start infrastructure

```bash
make docker-up
# Neo4j → http://localhost:7474
# Qdrant → http://localhost:6333/dashboard
# Grafana → http://localhost:3000 (admin/admin)
# Jaeger → http://localhost:16686
```

### 3. Install Python dependencies

```bash
make install
# or with uv: make install-uv
```

### 4. Ingest sample corpus

```bash
make ingest
# Loads data/sample_corpus/ into Qdrant + Neo4j
```

### 5. Run the UI

```bash
make run-gradio
# → http://localhost:7860
```

### 6. Run the API

```bash
make run-api
# → http://localhost:8000/docs
```

---

## Project Structure

```
rag-comparison-demo/
├── core/                   # Shared building blocks
│   ├── embeddings/         # EmbeddingProvider interface + OpenAI / ST implementations
│   ├── retrieval/          # Qdrant hybrid search + cross-encoder reranking
│   ├── guardrails/         # Input/output validation (injection, PII, grounding)
│   └── fallback/           # Layered fallback chain
├── pipelines/
│   ├── agentic_rag/        # LangGraph Agentic RAG (rewrite→retrieve→grade→generate→reflect)
│   └── graph_rag/          # LangGraph GraphRAG (entity→traverse→chunk→generate)
├── orchestrator/           # Top-level router, parallel mode, token budget
├── graph_db/               # Neo4j async client + schema + Cypher templates
├── api/
│   ├── rest/               # FastAPI: /query, /pipelines, /traces, /feedback
│   └── grpc/               # gRPC streaming service (.proto + server)
├── ui/                     # Gradio side-by-side dashboard
├── observability/          # OTel tracing, Prometheus metrics, structlog
├── finetuning/             # Embedding fine-tuning (dataset prep, train, eval)
├── data/                   # Corpus ingestion pipeline + sample documents
├── tests/                  # Unit, integration, RAGAS eval, Locust load tests
├── docs/                   # Architecture, design decisions, rollback runbook
├── prompts/                # Versioned prompt templates (YAML)
└── docker-compose.yml      # Neo4j, Qdrant, Redis, Postgres, OTel, Jaeger, Grafana
```

---

## Demo Queries

Three queries showcasing each pipeline's strengths:

| Query | Best Pipeline | Why |
|-------|--------------|-----|
| *"How are BERT and GPT architecturally related to the original Transformer?"* | **GraphRAG** | Multi-hop entity relationships |
| *"Summarize the key ideas in retrieval-augmented generation"* | **Agentic RAG** | Open-domain synthesis |
| *"Which researchers co-authored work connecting attention and graph networks?"* | **GraphRAG** | Author→paper→concept traversal |

---

## Phase Completion

| Phase | Tasks | Status |
|-------|-------|--------|
| Phase 1 | Bootstrap, corpus, embeddings | ✅ |
| Phase 2 | Agentic RAG + GraphRAG pipelines | ✅ |
| Phase 3 | Orchestrator + Gradio UI | ✅ |
| Phase 4 | REST/gRPC API + Observability | ✅ |
| Phase 5 | Tests + Fine-tuning + Docs | ✅ |

---

## Key Design Decisions

See [`docs/design_decisions.md`](docs/design_decisions.md) for deep-dive on:
- Why LangGraph over custom state machines
- RRF fusion vs learned sparse retrieval
- Neo4j 5.x vector index vs dedicated vector store
- Token budget enforcement strategy
- Embedding fine-tuning pipeline design

## Rollback

See [`docs/rollback.md`](docs/rollback.md) for the model/prompt rollback runbook.
