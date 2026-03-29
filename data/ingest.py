"""
data/ingest.py — Document ingestion pipeline.

Steps:
  1. Load PDFs and markdown from data/sample_corpus/
  2. Chunk with configurable overlap
  3. Deduplicate using content hash
  4. Embed and store in Qdrant (dense + sparse)
  5. Extract entities and store in Neo4j
  6. Track lineage in SQLite
"""
from __future__ import annotations

import asyncio
import hashlib
import json
import sqlite3
import time
import uuid
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import structlog

logger = structlog.get_logger(__name__)

CORPUS_DIR = Path(__file__).parent / "sample_corpus"
DB_PATH = Path(__file__).parent / "ingest_lineage.db"


# ── Document model ─────────────────────────────────────────────────────────────

@dataclass
class RawDocument:
    id: str
    path: str
    content: str
    metadata: dict[str, Any]


@dataclass
class Chunk:
    id: str
    doc_id: str
    content: str
    chunk_index: int
    metadata: dict[str, Any]
    content_hash: str


# ── Lineage DB ─────────────────────────────────────────────────────────────────

def init_lineage_db(db_path: Path = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(str(db_path))
    conn.execute("""
        CREATE TABLE IF NOT EXISTS documents (
            id TEXT PRIMARY KEY, path TEXT, ingested_at REAL,
            chunk_count INTEGER, embedding_model TEXT
        )
    """)
    conn.execute("""
        CREATE TABLE IF NOT EXISTS chunks (
            id TEXT PRIMARY KEY, doc_id TEXT, chunk_index INTEGER,
            content_hash TEXT, status TEXT
        )
    """)
    conn.commit()
    return conn


# ── Text splitter ──────────────────────────────────────────────────────────────

def split_text(
    text: str,
    chunk_size: int = 512,
    overlap: int = 64,
) -> list[str]:
    """Split text into overlapping chunks by word boundary."""
    words = text.split()
    chunks: list[str] = []
    start = 0
    while start < len(words):
        end = min(start + chunk_size, len(words))
        chunk = " ".join(words[start:end])
        chunks.append(chunk)
        if end >= len(words):
            break
        start += chunk_size - overlap
    return chunks


# ── Document loaders ───────────────────────────────────────────────────────────

def load_markdown(path: Path) -> RawDocument:
    content = path.read_text(encoding="utf-8")
    return RawDocument(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, str(path))),
        path=str(path),
        content=content,
        metadata={"source": path.name, "type": "markdown"},
    )


def load_pdf(path: Path) -> RawDocument:
    try:
        import pdfplumber  # type: ignore
        with pdfplumber.open(str(path)) as pdf:
            text = "\n\n".join(
                page.extract_text() or "" for page in pdf.pages
            )
    except ImportError:
        logger.warning("pdfplumber_not_installed", path=str(path))
        text = f"[PDF content from {path.name} — install pdfplumber to extract]"
    return RawDocument(
        id=str(uuid.uuid5(uuid.NAMESPACE_URL, str(path))),
        path=str(path),
        content=text,
        metadata={"source": path.name, "type": "pdf"},
    )


def load_corpus(corpus_dir: Path = CORPUS_DIR) -> list[RawDocument]:
    docs: list[RawDocument] = []
    for path in sorted(corpus_dir.rglob("*.md")):
        docs.append(load_markdown(path))
    for path in sorted(corpus_dir.rglob("*.pdf")):
        docs.append(load_pdf(path))
    logger.info("corpus_loaded", doc_count=len(docs))
    return docs


# ── Entity extractor (lightweight) ────────────────────────────────────────────

async def extract_entities_from_chunk(
    chunk: Chunk,
    llm: Any,
) -> list[dict[str, Any]]:
    prompt = (
        "Extract named entities from this text as JSON array. "
        "Each entity: {\"name\": str, \"type\": str (PERSON/ORG/TECH/CONCEPT/LOCATION)}.\n"
        "Return ONLY the JSON array, no explanation.\n\n"
        f"Text: {chunk.content[:600]}"
    )
    try:
        response = await llm.ainvoke(prompt)
        raw = response.content.strip()
        # Strip markdown code fences
        if raw.startswith("```"):
            raw = raw.split("\n", 1)[1].rsplit("```", 1)[0]
        return json.loads(raw)
    except Exception:
        return []


# ── Main ingestion pipeline ────────────────────────────────────────────────────

async def ingest(
    corpus_dir: Path = CORPUS_DIR,
    chunk_size: int = 512,
    overlap: int = 64,
    skip_neo4j: bool = False,
) -> None:
    from core.config import settings
    from core.embeddings.base import get_embedding_provider
    from core.retrieval.vector_store import QdrantVectorStore, Document
    from langchain_openai import ChatOpenAI

    conn = init_lineage_db()
    provider = get_embedding_provider()
    qs = QdrantVectorStore()

    # Ensure Qdrant collection
    await qs.ensure_collection(dimension=provider.dimension)

    # Load corpus
    raw_docs = load_corpus(corpus_dir)
    if not raw_docs:
        logger.warning("no_documents_found", corpus_dir=str(corpus_dir))
        return

    llm = ChatOpenAI(
        model=settings.llm_classification_model,
        api_key=settings.openai_api_key.get_secret_value(),
    )

    # Neo4j client (optional)
    neo4j_client = None
    if not skip_neo4j:
        try:
            from graph_db.neo4j_client import Neo4jClient
            neo4j_client = Neo4jClient()
            await neo4j_client.setup_schema()
        except Exception as exc:
            logger.warning("neo4j_unavailable", error=str(exc))

    total_chunks = 0
    for doc in raw_docs:
        logger.info("ingesting_document", doc=doc.metadata["source"])

        # Split
        texts = split_text(doc.content, chunk_size=chunk_size, overlap=overlap)
        chunks = [
            Chunk(
                id=str(uuid.uuid5(uuid.NAMESPACE_URL, f"{doc.id}::{i}")),
                doc_id=doc.id,
                content=t,
                chunk_index=i,
                metadata={**doc.metadata, "chunk_index": i},
                content_hash=hashlib.sha256(t.encode()).hexdigest(),
            )
            for i, t in enumerate(texts)
        ]

        # Dedup: skip already-ingested chunks by content hash
        cursor = conn.execute(
            "SELECT content_hash FROM chunks WHERE doc_id = ?", (doc.id,)
        )
        known_hashes = {row[0] for row in cursor.fetchall()}
        new_chunks = [c for c in chunks if c.content_hash not in known_hashes]
        if not new_chunks:
            logger.info("skipping_doc_already_ingested", doc=doc.metadata["source"])
            continue

        # Embed
        texts_to_embed = [c.content for c in new_chunks]
        embed_result = await provider.embed_documents(texts_to_embed)

        # Store in Qdrant
        qdrant_docs = [
            Document(
                id=c.id, content=c.content,
                metadata=c.metadata, source=doc.metadata["source"],
                embedding_model=provider.model_name,
            )
            for c in new_chunks
        ]
        await qs.upsert(qdrant_docs, embed_result.vectors)

        # Store in Neo4j
        if neo4j_client:
            await neo4j_client.upsert_document(doc.id, doc.metadata)
            for chunk, vec in zip(new_chunks, embed_result.vectors):
                await neo4j_client.upsert_chunk(
                    chunk.id, doc.id, chunk.content, vec, chunk.metadata
                )
                # Extract and store entities
                entities = await extract_entities_from_chunk(chunk, llm)
                for ent in entities:
                    ent_id = str(uuid.uuid5(uuid.NAMESPACE_URL, ent["name"].lower()))
                    await neo4j_client.upsert_entity(
                        ent_id, ent["name"], ent.get("type", "CONCEPT"), chunk.id
                    )

        # Record lineage
        for c in new_chunks:
            conn.execute(
                "INSERT OR REPLACE INTO chunks VALUES (?, ?, ?, ?, ?)",
                (c.id, c.doc_id, c.chunk_index, c.content_hash, "ingested"),
            )
        conn.execute(
            "INSERT OR REPLACE INTO documents VALUES (?, ?, ?, ?, ?)",
            (doc.id, doc.path, time.time(), len(chunks), provider.model_name),
        )
        conn.commit()
        total_chunks += len(new_chunks)
        logger.info(
            "document_ingested",
            doc=doc.metadata["source"],
            chunks=len(new_chunks),
            total_so_far=total_chunks,
        )

    logger.info("ingestion_complete", total_chunks=total_chunks)
    if neo4j_client:
        await neo4j_client.close()


if __name__ == "__main__":
    from observability.logging import setup_logging
    setup_logging(json_output=False)
    asyncio.run(ingest())
