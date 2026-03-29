# TODO.md — Agentic RAG vs GraphRAG: Comparison & Demo

> **Goal:** Build a side-by-side demo comparing Agentic RAG and GraphRAG pipelines,
> orchestrated via LangGraph, with a Gradio UI.
> Designed to showcase production-grade AI engineering competencies across
> retrieval strategy, graph databases, agent orchestration, observability, and reliability.

---

## 0. Project Bootstrap

### 0.1 Repo & Environment
- [ ] Init monorepo: `rag-comparison-demo/`
- [ ] Create `.env.example` with all required keys (OpenAI/Anthropic, Neo4j, AWS Neptune if used, LangSmith)
- [ ] Set up `pyproject.toml` with dependency groups: `core`, `graphrag`, `ui`, `observability`, `dev`
- [ ] Pin versions for reproducibility (`uv` or `poetry.lock`)
- [ ] Add `Makefile` with targets: `install`, `run-gradio`, `test`, `lint`, `docker-up`

### 0.2 Folder Structure
```
rag-comparison-demo/
├── core/
│   ├── embeddings/          # Embedding models, fine-tuned adapters
│   ├── retrieval/           # Vector search, graph traversal
│   ├── guardrails/          # Input/output validation
│   └── fallback/            # Fallback chains and retry logic
├── pipelines/
│   ├── agentic_rag/         # Agentic RAG LangGraph graph
│   └── graph_rag/           # GraphRAG LangGraph graph
├── orchestrator/
│   ├── state.py             # Shared TypedDict state schema
│   ├── router.py            # Pipeline router node
│   └── graph.py             # Top-level LangGraph compilation
├── graph_db/
│   ├── neo4j_client.py      # Neo4j driver + Cypher helpers
│   ├── neptune_client.py    # AWS Neptune SPARQL/Gremlin client (optional)
│   └── schema/              # Node/edge definitions, indexes
├── api/
│   ├── rest/                # FastAPI REST endpoints
│   └── grpc/                # gRPC service definitions (.proto)
├── ui/
│   └── gradio_app.py        # Side-by-side comparison dashboard
├── observability/
│   ├── tracing.py           # LangSmith / OpenTelemetry setup
│   ├── metrics.py           # Latency, token usage, cost tracking
│   └── logging.py           # Structured JSON logging
├── finetuning/
│   ├── dataset_prep.py      # Training data prep scripts
│   ├── train.py             # LoRA / QLoRA fine-tuning runner
│   └── eval.py              # Fine-tuned model evaluation
├── data/
│   └── sample_corpus/       # Self-contained demo documents
├── tests/
├── docker-compose.yml
└── TODO.md
```

### 0.3 Docker Services
- [ ] `docker-compose.yml` with services:
  - `neo4j` (Neo4j 5.x, ports 7474/7687)
  - `qdrant` (vector store)
  - `redis` (cache + rate limiting)
  - `otel-collector` (OpenTelemetry)
  - `app` (the Python backend)
- [ ] Health checks for all services
- [ ] Volume mounts for Neo4j data persistence

---

## 1. Sample Corpus & Data Prep

- [ ] Choose a demo domain (e.g. scientific papers, tech docs, financial reports — pick one coherent set)
- [ ] Collect 50–200 documents (PDFs or markdown)
- [ ] Write `data/ingest.py`: chunk, clean, deduplicate
- [ ] Store raw docs + metadata in SQLite or PostgreSQL for traceability

---

## 2. Embeddings & Vector Store (Req #6)

### 2.1 Embedding Models
- [ ] Integrate `text-embedding-3-small` and `text-embedding-3-large` (OpenAI)
- [ ] Add `sentence-transformers` as local/offline alternative
- [ ] Abstract behind `core/embeddings/base.py` (`EmbeddingProvider` interface)

### 2.2 Fine-Tuned Embedding Adapter (Req #3)
- [ ] Prepare contrastive training pairs from sample corpus (`finetuning/dataset_prep.py`)
- [ ] Fine-tune embedding model using `sentence-transformers` training API or LoRA adapter on base LLM
- [ ] Evaluate: MRR, NDCG, Recall@K before vs after fine-tuning
- [ ] Register fine-tuned model with version tag (e.g. `embed-ft-v1`)
- [ ] Add model version to all embedding calls for rollback support (Req #8)

### 2.3 Vector Store & Search Optimization (Req #6)
- [ ] Ingest chunks into Qdrant (or Weaviate)
- [ ] Implement HNSW index tuning: `m`, `ef_construction` params
- [ ] Add hybrid search: dense + sparse (BM25) with RRF fusion
- [ ] Implement re-ranking layer (Cohere Rerank or cross-encoder)
- [ ] Benchmark: P50/P95 latency per retrieval strategy
- [ ] Cache frequent queries in Redis with TTL (Req #8)

---

## 3. Agentic RAG Pipeline (Req #6, #7)

### 3.1 LangGraph Graph Design
- [ ] Define `AgenticRAGState` TypedDict:
  ```python
  class AgenticRAGState(TypedDict):
      query: str
      rewritten_query: str
      retrieved_chunks: list[Document]
      relevance_scores: list[float]
      agent_scratchpad: list[dict]
      final_answer: str
      fallback_triggered: bool
      trace_id: str
      token_usage: dict
  ```
- [ ] Implement nodes:
  - `query_rewrite_node` — rewrite/expand query for better retrieval
  - `retrieve_node` — vector search with hybrid + reranking
  - `grade_relevance_node` — LLM-based relevance scoring of chunks
  - `generate_node` — final answer generation with context
  - `self_reflect_node` — evaluate answer quality, trigger re-retrieval if needed
  - `fallback_node` — fallback to broader retrieval or web search (Req #6)
  - `guardrails_node` — input/output validation (Req #6)
- [ ] Wire conditional edges:
  - After grading: if relevance < threshold → re-retrieve or fallback
  - After reflection: if answer quality low → loop back (max 3 iterations)
- [ ] Set `recursion_limit` on graph to prevent infinite loops (Req #7)
- [ ] Compile graph with `checkpointer` for state persistence

### 3.2 Guardrails & Fallback (Req #6)
- [ ] Input guardrail: block prompt injection, PII detection (use `guardrails-ai` or `nemo-guardrails`)
- [ ] Output guardrail: hallucination check, citation grounding score
- [ ] Fallback chain: vector search → graph search → web search → "I don't know"
- [ ] Configurable fallback thresholds via env/config file

---

## 4. GraphRAG Pipeline (Req #4, #6, #7)

### 4.1 Knowledge Graph Construction
- [ ] Design graph schema:
  - Nodes: `Document`, `Chunk`, `Entity`, `Concept`, `Author`
  - Edges: `MENTIONS`, `RELATED_TO`, `PART_OF`, `AUTHORED_BY`
- [ ] Implement NER + relation extraction pipeline (spaCy or LLM-based)
- [ ] Ingest into **Neo4j**:
  - [ ] Write `graph_db/neo4j_client.py` with connection pool, retry logic
  - [ ] Create Cypher ingestion scripts in `graph_db/schema/`
  - [ ] Add full-text + vector indexes on Neo4j 5.x
- [ ] (Optional) Mirror to **AWS Neptune**:
  - [ ] `graph_db/neptune_client.py` with Gremlin traversal client
  - [ ] SPARQL query examples for RDF use case

### 4.2 GraphRAG LangGraph Graph
- [ ] Define `GraphRAGState` TypedDict (same base fields + graph-specific):
  ```python
  class GraphRAGState(TypedDict):
      query: str
      entities_extracted: list[str]
      graph_subgraph: dict           # nodes + edges retrieved
      community_summaries: list[str] # optional: community detection
      retrieved_chunks: list[Document]
      final_answer: str
      graph_traversal_trace: list    # for UI visualization
      trace_id: str
      token_usage: dict
  ```
- [ ] Implement nodes:
  - `entity_extract_node` — extract entities from query
  - `graph_traverse_node` — Cypher/Gremlin multi-hop traversal from entities
  - `community_summary_node` — summarize graph neighborhoods (optional Microsoft GraphRAG approach)
  - `chunk_retrieve_node` — fetch associated text chunks from matched nodes
  - `generate_node` — answer generation with graph context
  - `guardrails_node` — shared with Agentic RAG
- [ ] Implement Cypher query templates for common traversal patterns
- [ ] Visualize subgraph trace in UI (nodes/edges retrieved per query)

---

## 5. LangGraph Orchestrator (Req #7)

- [ ] `orchestrator/state.py` — unified base state shared by both pipelines
- [ ] `orchestrator/router.py` — route incoming query to Agentic RAG or GraphRAG (or both in parallel)
- [ ] `orchestrator/graph.py` — top-level graph:
  - Entry: `classify_query_node` (decide pipeline based on query type)
  - Parallel branch support via `Send` API for side-by-side comparison mode
- [ ] State management:
  - [ ] Use LangGraph `MemorySaver` for short-term (session) state
  - [ ] Use PostgreSQL checkpointer for long-term persistence
- [ ] Controlled execution boundaries:
  - [ ] `recursion_limit` per graph
  - [ ] Timeout per node (wrap with `asyncio.wait_for`)
  - [ ] Max token budget enforced at orchestrator level (Req #8)
- [ ] Safe multi-step reasoning:
  - [ ] Log every node transition with state diff
  - [ ] Human-in-the-loop breakpoint support (LangGraph `interrupt_before`)

---

## 6. API Layer (Req #5)

### 6.1 REST API (FastAPI)
- [ ] `POST /api/v1/query` — run query through selected pipeline(s)
- [ ] `GET /api/v1/pipelines` — list available pipelines + status
- [ ] `GET /api/v1/traces/{trace_id}` — fetch full execution trace
- [ ] `POST /api/v1/feedback` — RLHF-style feedback collection
- [ ] Auth: API key middleware
- [ ] Rate limiting: `slowapi` + Redis backend
- [ ] OpenAPI spec auto-generated, exported to `api/openapi.json`

### 6.2 gRPC Service (Req #5)
- [ ] Define `api/grpc/rag_service.proto`:
  ```protobuf
  service RAGService {
    rpc Query (QueryRequest) returns (stream QueryResponse);
    rpc GetTrace (TraceRequest) returns (TraceResponse);
  }
  ```
- [ ] Implement server with `grpcio` + `grpcio-tools`
- [ ] Streaming responses for token-by-token generation

### 6.3 Event-Driven Architecture (Req #5)
- [ ] Add Redis Streams or Kafka topic: `rag.query.requested` → `rag.answer.generated`
- [ ] Async query worker: decouple API from inference for throughput scaling (Req #8)
- [ ] Dead letter queue for failed inference jobs

---

## 7. UI Layer — Gradio (`ui/gradio_app.py`)

- [ ] Two-column `gr.Blocks` layout: left = Agentic RAG, right = GraphRAG
- [ ] Shared query input that fires both pipelines in parallel
- [ ] Answer + source attribution per pipeline
- [ ] Retrieved chunks viewer (expandable, with relevance scores)
- [ ] Graph subgraph visualizer for GraphRAG results (`gr.Plot` or embedded `pyvis` HTML)
- [ ] Comparison metrics table: latency, tokens, retrieval count, answer length
- [ ] Latency breakdown per node (bar chart via `gr.BarPlot`)
- [ ] Full trace JSON expander (`gr.JSON`)
- [ ] Thumbs up/down feedback buttons (posts to `/api/v1/feedback`)
- [ ] Token usage + cost estimate display per pipeline

---

## 8. Observability & Production Reliability (Req #8)

### 8.1 Tracing
- [ ] Integrate **LangSmith** for LangGraph trace capture
- [ ] Add **OpenTelemetry** spans around every node execution
- [ ] Trace ID propagated through all state objects and API responses
- [ ] Export traces to Jaeger or Grafana Tempo

### 8.2 Metrics
- [ ] Instrument with Prometheus metrics:
  - `rag_query_latency_seconds` (histogram, by pipeline + node)
  - `rag_token_usage_total` (counter, by model + pipeline)
  - `rag_cost_usd_total` (counter)
  - `rag_fallback_triggered_total` (counter)
  - `rag_retrieval_relevance_score` (gauge)
- [ ] Grafana dashboard JSON: latency P50/P95, cost/day, fallback rate

### 8.3 Structured Logging
- [ ] JSON logs via `structlog`
- [ ] Log fields: `trace_id`, `pipeline`, `node`, `latency_ms`, `token_count`, `model_version`
- [ ] Correlation ID middleware in FastAPI

### 8.4 Versioning & Rollback (Req #8)
- [ ] Version every prompt template (e.g. `prompts/generate_v2.yaml`)
- [ ] Version every model used: store `model_name + version` in state + logs
- [ ] Feature flag system (env-based or `flagsmith`) to switch models/prompts without redeploy
- [ ] Shadow mode: run new prompt version alongside old, compare outputs before cutover
- [ ] Rollback runbook documented in `docs/rollback.md`

### 8.5 Cost & Token Efficiency (Req #8)
- [ ] Token budget enforcer: hard cap per query at orchestrator level
- [ ] Prompt compression: `LLMLingua` or similar for long context reduction
- [ ] Model routing: cheap model for classification/grading, expensive model for generation only
- [ ] Cache identical query+context responses in Redis (semantic cache via Qdrant)

---

## 9. Testing

- [ ] Unit tests for each LangGraph node (mock LLM calls)
- [ ] Integration tests: full pipeline runs against local Neo4j + Qdrant
- [ ] Retrieval eval: `RAGAS` metrics (faithfulness, answer relevancy, context recall)
- [ ] GraphRAG vs Agentic RAG comparison benchmark: fixed 50-query eval set
- [ ] Load test: `locust` — target 100 RPS, P95 < 3s
- [ ] Guardrails test suite: adversarial prompts, PII inputs

---

## 10. Documentation & Demo

- [ ] `README.md`: architecture diagram, quickstart, env setup
- [ ] Architecture diagram (`docs/architecture.png`) — LangGraph flow + system components
- [ ] `docs/design_decisions.md` — why Agentic RAG vs GraphRAG, tradeoffs
- [ ] `docs/rollback.md` — model/prompt rollback runbook
- [ ] Demo script: 3 example queries that showcase GraphRAG's advantage (entity-heavy) vs Agentic RAG's advantage (open-domain)
- [ ] Record Loom walkthrough (optional)

---

## Priority Order (Quick Start)

```
Phase 1 (Day 1–2):   0 + 1 + 2.1–2.3   → env, data, embeddings, vector store
Phase 2 (Day 3–4):   3 + 4              → both RAG pipelines working end-to-end
Phase 3 (Day 5):     5 + 7              → orchestrator + Gradio UI connected
Phase 4 (Day 6):     6 + 8              → API layer + observability
Phase 5 (Day 7):     9 + 10             → tests, benchmarks, docs
```

---

*Last updated: 2026-03-29*
