# Model & Prompt Rollback Runbook

## Overview

This runbook documents the procedure for rolling back model versions, prompt templates,
or embedding adapters in the RAG comparison demo system.

---

## 1. Rolling Back an LLM Model

**Trigger**: Regression in answer quality, latency spike, or cost anomaly detected in Grafana.

### Steps

1. **Identify the bad version** via LangSmith trace comparison:
   - Open https://smith.langchain.com → Project: `rag-comparison-demo`
   - Filter by model name and date range
   - Compare RAGAS scores before/after the change

2. **Update `.env`** (or environment variable):
   ```bash
   # Before (bad version)
   LLM_GENERATION_MODEL=gpt-4o-2025-04-01

   # After (stable version)
   LLM_GENERATION_MODEL=gpt-4o-2024-11-20
   ```

3. **Redeploy** (zero-downtime with Docker):
   ```bash
   docker-compose up -d --no-deps app
   ```

4. **Verify** via smoke test:
   ```bash
   curl -X POST http://localhost:8000/api/v1/query \
     -H "X-API-Key: $API_KEY" \
     -H "Content-Type: application/json" \
     -d '{"query": "What is RAG?", "pipeline": "agentic"}'
   ```

---

## 2. Rolling Back a Prompt Template

Each prompt is versioned under `prompts/`. The current version is specified in `.env`
or feature flags.

### Steps

1. **Check prompt history** in `prompts/`:
   ```
   prompts/generate_v1.yaml   ← stable baseline
   prompts/generate_v2.yaml   ← current (problematic)
   ```

2. **Set feature flag** to revert:
   - In `.env`: comment out the new prompt version reference
   - Or via `flagsmith` dashboard if using a feature flag service

3. **Shadow mode test** (before full rollback):
   Set `SHADOW_MODE_ENABLED=true` — this runs both prompt versions in parallel,
   logs both outputs, but only returns the old version's result to users.
   Useful for comparing without user impact.

4. **Confirm rollback** in structured logs:
   ```bash
   docker-compose logs app | grep '"prompt_version"'
   ```

---

## 3. Rolling Back an Embedding Model / Fine-Tuned Adapter

**Trigger**: MRR or Recall@10 drops significantly in `finetuning/eval.py` output.

### Steps

1. **Disable fine-tuned adapter**:
   ```bash
   FEATURE_FINETUNED_EMBEDDINGS=false
   ```
   This immediately routes to the base `text-embedding-3-small` model.

2. **Re-ingest** if switching embedding dimensions:
   ```bash
   make ingest
   ```
   (This re-creates Qdrant collections with the correct vector dimensions.)

3. **Register old adapter version** for re-deployment:
   - Check `models/` directory for `version_manifest.json` files
   - Set `EMBEDDING_MODEL_FT=embed-ft-v0` (previous stable version)
   - Set `FEATURE_FINETUNED_EMBEDDINGS=true`

---

## 4. Monitoring Signals That Trigger Rollback

| Signal | Threshold | Source |
|--------|-----------|--------|
| RAGAS faithfulness drop | < 0.6 | LangSmith eval |
| Fallback rate spike | > 20% of queries | Grafana `rag_fallback_triggered_total` |
| P95 latency increase | > 5s | Grafana `rag_query_latency_seconds` |
| Cost per query increase | > 2× baseline | Grafana `rag_cost_usd_total` |
| Error rate | > 5% | Grafana / app logs |

---

## 5. Emergency Rollback Checklist

- [ ] Identify failing component (LLM / prompt / embedding / retrieval)
- [ ] Check LangSmith traces for error patterns
- [ ] Set environment variable to previous stable version
- [ ] Redeploy with `docker-compose up -d --no-deps app`
- [ ] Run smoke test (see Section 1, Step 4)
- [ ] Verify Grafana dashboard returns to baseline
- [ ] Post-mortem: document root cause in `docs/post-mortems/`
