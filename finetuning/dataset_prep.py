"""
finetuning/dataset_prep.py — Prepare contrastive training pairs for embedding fine-tuning.

Output: HuggingFace Dataset with (anchor, positive, negative) triples
stored as data/ft_dataset/train.jsonl and eval.jsonl.
"""
from __future__ import annotations

import json
import random
import sqlite3
from pathlib import Path

import structlog

logger = structlog.get_logger(__name__)

DB_PATH = Path(__file__).parent.parent / "data" / "ingest_lineage.db"
OUTPUT_DIR = Path(__file__).parent.parent / "data" / "ft_dataset"


def load_chunks_from_db() -> list[dict]:
    conn = sqlite3.connect(str(DB_PATH))
    rows = conn.execute(
        "SELECT id, doc_id FROM chunks WHERE status = 'ingested'"
    ).fetchall()
    conn.close()
    return [{"id": r[0], "doc_id": r[1]} for r in rows]


async def generate_qa_pairs(
    chunks: list[dict],
    n_pairs: int = 500,
) -> list[dict]:
    """
    Use an LLM to generate synthetic (query, relevant_chunk) pairs
    from the ingested corpus.
    """
    from core.config import settings
    from langchain_openai import ChatOpenAI
    from core.retrieval.vector_store import QdrantVectorStore

    llm = ChatOpenAI(
        model=settings.llm_classification_model,
        api_key=settings.openai_api_key.get_secret_value(),
    )
    qs = QdrantVectorStore()
    client = qs._get_client()

    pairs = []
    sampled = random.sample(chunks, min(n_pairs, len(chunks)))

    for chunk_meta in sampled:
        # Fetch chunk content from Qdrant
        results = client.retrieve(
            collection_name=settings.qdrant_collection,
            ids=[chunk_meta["id"]],
            with_payload=True,
        )
        if not results:
            continue
        content = results[0].payload.get("content", "")
        if len(content) < 50:
            continue

        # Generate a natural query for this chunk
        prompt = (
            "Generate a concise natural language question that this text passage answers. "
            "Output ONLY the question, nothing else.\n\n"
            f"Passage: {content[:500]}"
        )
        try:
            resp = await llm.ainvoke(prompt)
            question = resp.content.strip()
            pairs.append({
                "anchor": question,
                "positive": content,
                "doc_id": chunk_meta["doc_id"],
                "chunk_id": chunk_meta["id"],
            })
        except Exception as exc:
            logger.warning("qa_gen_failed", error=str(exc))

    return pairs


def build_triplets(pairs: list[dict]) -> list[dict]:
    """
    Build (anchor, positive, negative) triplets for contrastive learning.
    Negative: a chunk from a different document.
    """
    triplets = []
    by_doc: dict[str, list[dict]] = {}
    for p in pairs:
        by_doc.setdefault(p["doc_id"], []).append(p)

    for pair in pairs:
        # Find a hard negative: chunk from a different doc
        other_docs = [d for d in by_doc if d != pair["doc_id"] and by_doc[d]]
        if not other_docs:
            continue
        neg_doc = random.choice(other_docs)
        neg_pair = random.choice(by_doc[neg_doc])
        triplets.append({
            "anchor": pair["anchor"],
            "positive": pair["positive"],
            "negative": neg_pair["positive"],
        })
    return triplets


def save_dataset(triplets: list[dict], output_dir: Path = OUTPUT_DIR) -> None:
    output_dir.mkdir(parents=True, exist_ok=True)
    random.shuffle(triplets)
    split = int(len(triplets) * 0.9)
    train, eval_ = triplets[:split], triplets[split:]

    for split_name, data in [("train", train), ("eval", eval_)]:
        path = output_dir / f"{split_name}.jsonl"
        with open(path, "w") as f:
            for item in data:
                f.write(json.dumps(item) + "\n")
        logger.info("dataset_saved", split=split_name, count=len(data), path=str(path))


if __name__ == "__main__":
    import asyncio
    from observability.logging import setup_logging
    setup_logging(json_output=False)

    async def main() -> None:
        chunks = load_chunks_from_db()
        if not chunks:
            logger.error("no_chunks_found", msg="Run data/ingest.py first")
            return
        pairs = await generate_qa_pairs(chunks, n_pairs=200)
        triplets = build_triplets(pairs)
        save_dataset(triplets)
        logger.info("dataset_ready", total_triplets=len(triplets))

    asyncio.run(main())
