"""
graph_db/neo4j_client.py — Neo4j async driver with:
  - Connection pool + retry logic
  - Cypher helper methods for GraphRAG traversal
  - Full-text + vector index support (Neo4j 5.x)
"""
from __future__ import annotations

import asyncio
from contextlib import asynccontextmanager
from typing import Any, AsyncGenerator

import structlog
from tenacity import (
    retry,
    stop_after_attempt,
    wait_exponential,
    retry_if_exception_type,
)

logger = structlog.get_logger(__name__)


class Neo4jClient:
    """
    Async Neo4j client wrapping the official neo4j driver.
    All queries go through this client; never instantiate drivers directly.
    """

    def __init__(self) -> None:
        from core.config import settings
        self._settings = settings
        self._driver: Any = None

    def _get_driver(self) -> Any:
        if self._driver is None:
            from neo4j import AsyncGraphDatabase  # type: ignore
            self._driver = AsyncGraphDatabase.driver(
                self._settings.neo4j_uri,
                auth=(
                    self._settings.neo4j_user,
                    self._settings.neo4j_password.get_secret_value(),
                ),
                max_connection_pool_size=50,
                connection_acquisition_timeout=30,
            )
        return self._driver

    async def close(self) -> None:
        if self._driver:
            await self._driver.close()
            self._driver = None

    @asynccontextmanager
    async def session(self) -> AsyncGenerator[Any, None]:
        driver = self._get_driver()
        async with driver.session(database=self._settings.neo4j_database) as s:
            yield s

    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=1, max=10),
        retry=retry_if_exception_type(Exception),
        reraise=True,
    )
    async def run_query(
        self, cypher: str, params: dict[str, Any] | None = None
    ) -> list[dict[str, Any]]:
        """Execute a Cypher query and return list of record dicts."""
        async with self.session() as s:
            result = await s.run(cypher, params or {})
            records = await result.data()
            return records

    # ── Schema setup ──────────────────────────────────────────────────────────

    async def setup_schema(self) -> None:
        """Create indexes and constraints for the knowledge graph schema."""
        statements = [
            # Constraints
            "CREATE CONSTRAINT IF NOT EXISTS FOR (d:Document) REQUIRE d.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (c:Chunk) REQUIRE c.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (e:Entity) REQUIRE e.id IS UNIQUE",
            "CREATE CONSTRAINT IF NOT EXISTS FOR (a:Author) REQUIRE a.name IS UNIQUE",
            # Full-text indexes
            """CREATE FULLTEXT INDEX chunk_fulltext IF NOT EXISTS
               FOR (c:Chunk) ON EACH [c.content]
               OPTIONS {indexConfig: {`fulltext.analyzer`: 'english'}}""",
            """CREATE FULLTEXT INDEX entity_fulltext IF NOT EXISTS
               FOR (e:Entity) ON EACH [e.name, e.description]""",
            # Vector index (Neo4j 5.11+)
            """CREATE VECTOR INDEX chunk_vector IF NOT EXISTS
               FOR (c:Chunk) ON c.embedding
               OPTIONS {indexConfig: {
                 `vector.dimensions`: 1536,
                 `vector.similarity_function`: 'cosine'
               }}""",
        ]
        for stmt in statements:
            try:
                await self.run_query(stmt)
            except Exception as exc:
                logger.warning("schema_statement_failed", error=str(exc), stmt=stmt[:80])
        logger.info("neo4j_schema_ready")

    # ── Ingestion helpers ─────────────────────────────────────────────────────

    async def upsert_document(self, doc_id: str, metadata: dict[str, Any]) -> None:
        await self.run_query(
            """
            MERGE (d:Document {id: $id})
            SET d += $props
            """,
            {"id": doc_id, "props": metadata},
        )

    async def upsert_chunk(
        self,
        chunk_id: str,
        doc_id: str,
        content: str,
        embedding: list[float],
        metadata: dict[str, Any] | None = None,
    ) -> None:
        await self.run_query(
            """
            MERGE (c:Chunk {id: $chunk_id})
            SET c.content = $content,
                c.embedding = $embedding,
                c.metadata = $metadata
            WITH c
            MATCH (d:Document {id: $doc_id})
            MERGE (c)-[:PART_OF]->(d)
            """,
            {
                "chunk_id": chunk_id,
                "doc_id": doc_id,
                "content": content,
                "embedding": embedding,
                "metadata": metadata or {},
            },
        )

    async def upsert_entity(
        self,
        entity_id: str,
        name: str,
        entity_type: str,
        chunk_id: str,
        description: str = "",
    ) -> None:
        await self.run_query(
            """
            MERGE (e:Entity {id: $entity_id})
            SET e.name = $name, e.type = $type, e.description = $description
            WITH e
            MATCH (c:Chunk {id: $chunk_id})
            MERGE (c)-[:MENTIONS]->(e)
            """,
            {
                "entity_id": entity_id,
                "name": name,
                "type": entity_type,
                "chunk_id": chunk_id,
                "description": description,
            },
        )

    async def upsert_relation(
        self,
        source_entity_id: str,
        target_entity_id: str,
        relation_type: str,
        weight: float = 1.0,
    ) -> None:
        rel = relation_type.upper().replace(" ", "_")
        await self.run_query(
            f"""
            MATCH (s:Entity {{id: $source}}), (t:Entity {{id: $target}})
            MERGE (s)-[r:{rel}]->(t)
            SET r.weight = $weight
            """,
            {"source": source_entity_id, "target": target_entity_id, "weight": weight},
        )

    # ── Traversal queries ─────────────────────────────────────────────────────

    async def find_entities(self, names: list[str]) -> list[dict[str, Any]]:
        """Find entity nodes by approximate name match."""
        return await self.run_query(
            """
            UNWIND $names AS name
            MATCH (e:Entity)
            WHERE toLower(e.name) CONTAINS toLower(name)
            RETURN e.id AS id, e.name AS name, e.type AS type,
                   e.description AS description
            LIMIT 20
            """,
            {"names": names},
        )

    async def multi_hop_traverse(
        self,
        entity_ids: list[str],
        max_hops: int = 2,
        limit: int = 50,
    ) -> dict[str, Any]:
        """Multi-hop graph traversal from seed entities."""
        records = await self.run_query(
            f"""
            UNWIND $ids AS seed_id
            MATCH path = (start:Entity {{id: seed_id}})-[*1..{max_hops}]-(related)
            WHERE related:Entity OR related:Chunk
            UNWIND nodes(path) AS n
            UNWIND relationships(path) AS r
            RETURN DISTINCT
              n.id AS node_id,
              labels(n)[0] AS node_label,
              n.name AS node_name,
              n.content AS node_content,
              type(r) AS rel_type,
              startNode(r).id AS rel_source,
              endNode(r).id AS rel_target
            LIMIT $limit
            """,
            {"ids": entity_ids, "limit": limit},
        )

        nodes: dict[str, Any] = {}
        edges: list[dict[str, Any]] = []
        for row in records:
            nid = row["node_id"]
            if nid and nid not in nodes:
                nodes[nid] = {
                    "id": nid,
                    "label": row["node_label"],
                    "name": row.get("node_name") or row.get("node_content", "")[:100],
                }
            if row["rel_source"] and row["rel_target"]:
                edges.append({
                    "source": row["rel_source"],
                    "target": row["rel_target"],
                    "type": row["rel_type"],
                })

        return {"nodes": list(nodes.values()), "edges": edges}

    async def get_chunks_for_entities(
        self, entity_ids: list[str], limit: int = 10
    ) -> list[dict[str, Any]]:
        """Fetch text chunks associated with matched entities."""
        return await self.run_query(
            """
            UNWIND $ids AS eid
            MATCH (e:Entity {id: eid})<-[:MENTIONS]-(c:Chunk)
            RETURN DISTINCT c.id AS id, c.content AS content,
                   c.metadata AS metadata
            LIMIT $limit
            """,
            {"ids": entity_ids, "limit": limit},
        )

    async def fulltext_search(
        self, query: str, limit: int = 10
    ) -> list[dict[str, Any]]:
        """Full-text search over chunk content."""
        return await self.run_query(
            """
            CALL db.index.fulltext.queryNodes('chunk_fulltext', $query)
            YIELD node, score
            RETURN node.id AS id, node.content AS content, score
            ORDER BY score DESC
            LIMIT $limit
            """,
            {"query": query, "limit": limit},
        )

    async def vector_search(
        self, embedding: list[float], top_k: int = 10
    ) -> list[dict[str, Any]]:
        """Vector similarity search over chunk embeddings."""
        return await self.run_query(
            """
            CALL db.index.vector.queryNodes('chunk_vector', $k, $embedding)
            YIELD node, score
            RETURN node.id AS id, node.content AS content, score
            ORDER BY score DESC
            """,
            {"embedding": embedding, "k": top_k},
        )
