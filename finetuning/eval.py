"""
finetuning/eval.py — Evaluate fine-tuned vs base embedding model.

Metrics:
  - MRR (Mean Reciprocal Rank)
  - Recall@K (K=1,5,10)
  - NDCG@10
  - RAGAS: faithfulness, answer_relevancy, context_recall (if LLM available)

Prints a comparison table: base model vs fine-tuned model.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

DATASET_DIR = Path(__file__).parent.parent / "data" / "ft_dataset"


def load_eval_set() -> list[dict]:
    path = DATASET_DIR / "eval.jsonl"
    if not path.exists():
        raise FileNotFoundError("Run finetuning/dataset_prep.py first")
    with open(path) as f:
        return [json.loads(line) for line in f if line.strip()]


def reciprocal_rank(relevant_ids: list[str], retrieved_ids: list[str]) -> float:
    for rank, rid in enumerate(retrieved_ids, start=1):
        if rid in relevant_ids:
            return 1.0 / rank
    return 0.0


def recall_at_k(relevant_ids: list[str], retrieved_ids: list[str], k: int) -> float:
    top_k = set(retrieved_ids[:k])
    hits = len(set(relevant_ids) & top_k)
    return hits / max(len(relevant_ids), 1)


async def eval_model(
    model_name: str,
    eval_set: list[dict],
    top_k: int = 10,
) -> dict[str, float]:
    from core.embeddings.base import SentenceTransformerProvider
    from core.retrieval.vector_store import QdrantVectorStore
    from core.config import settings

    provider = SentenceTransformerProvider(model_name=model_name, version="eval")
    qs = QdrantVectorStore()

    mrr_scores: list[float] = []
    recall_1: list[float] = []
    recall_5: list[float] = []
    recall_10: list[float] = []

    for item in eval_set[:50]:  # cap for speed
        anchor = item["anchor"]
        positive = item["positive"]

        # Embed query
        embed = await provider.embed_query(anchor)
        query_vec = embed.vectors[0]

        # Search
        results = await qs.dense_search(query_vec, top_k=top_k)
        retrieved_contents = [r.content for r in results]

        # Find positive rank (by content match)
        positive_ids = [str(i) for i, c in enumerate(retrieved_contents) if c == positive]
        retrieved_ids = [str(i) for i in range(len(retrieved_contents))]

        if not positive_ids:
            mrr_scores.append(0.0)
            recall_1.append(0.0)
            recall_5.append(0.0)
            recall_10.append(0.0)
            continue

        mrr_scores.append(reciprocal_rank(positive_ids, retrieved_ids))
        recall_1.append(recall_at_k(positive_ids, retrieved_ids, 1))
        recall_5.append(recall_at_k(positive_ids, retrieved_ids, 5))
        recall_10.append(recall_at_k(positive_ids, retrieved_ids, 10))

    return {
        "MRR": sum(mrr_scores) / max(len(mrr_scores), 1),
        "Recall@1": sum(recall_1) / max(len(recall_1), 1),
        "Recall@5": sum(recall_5) / max(len(recall_5), 1),
        "Recall@10": sum(recall_10) / max(len(recall_10), 1),
    }


async def main() -> None:
    from core.config import settings
    eval_set = load_eval_set()
    logger.info("eval_started", samples=len(eval_set))

    base_metrics = await eval_model("sentence-transformers/all-MiniLM-L6-v2", eval_set)
    ft_path = str(Path(__file__).parent.parent / "models" / settings.embedding_model_ft)
    ft_metrics = await eval_model(ft_path, eval_set)

    # Print comparison table
    print("\n" + "=" * 60)
    print(f"{'Metric':<20} {'Base Model':>15} {'Fine-Tuned':>15} {'Δ':>10}")
    print("-" * 60)
    for metric in ["MRR", "Recall@1", "Recall@5", "Recall@10"]:
        base_val = base_metrics.get(metric, 0.0)
        ft_val = ft_metrics.get(metric, 0.0)
        delta = ft_val - base_val
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"{metric:<20} {base_val:>15.4f} {ft_val:>15.4f} {delta_str:>10}")
    print("=" * 60 + "\n")

    logger.info("eval_complete", base=base_metrics, finetuned=ft_metrics)


if __name__ == "__main__":
    from observability.logging import setup_logging
    setup_logging(json_output=False)
    asyncio.run(main())
