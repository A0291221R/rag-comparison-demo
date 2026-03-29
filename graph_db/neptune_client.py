"""
graph_db/neptune_client.py — AWS Neptune client (optional).

Supports both Gremlin (property graph) and SPARQL (RDF) query modes.
Only used when NEPTUNE_ENDPOINT is configured.
"""
from __future__ import annotations

from typing import Any

import structlog

logger = structlog.get_logger(__name__)


class NeptuneClient:
    """
    Async AWS Neptune client.
    Neptune does not have an official async Python driver;
    this wraps synchronous calls in asyncio thread pool.
    """

    def __init__(self) -> None:
        from core.config import settings
        self._settings = settings
        self._gremlin_client: Any = None

    def _is_configured(self) -> bool:
        return bool(self._settings.neptune_endpoint)

    def _get_gremlin_client(self) -> Any:
        if not self._is_configured():
            raise RuntimeError("Neptune endpoint not configured")
        if self._gremlin_client is None:
            from gremlin_python.driver import client as gremlin  # type: ignore
            endpoint = (
                f"wss://{self._settings.neptune_endpoint}:"
                f"{self._settings.neptune_port}/gremlin"
            )
            self._gremlin_client = gremlin.Client(endpoint, "g")
        return self._gremlin_client

    async def gremlin_query(self, query: str) -> list[Any]:
        """Execute a Gremlin traversal query."""
        import asyncio
        if not self._is_configured():
            return []
        client = self._get_gremlin_client()
        loop = asyncio.get_event_loop()
        result = await loop.run_in_executor(
            None, lambda: client.submit(query).all().result()
        )
        return result

    async def find_entities(self, names: list[str]) -> list[dict[str, Any]]:
        """Find vertices by name using Gremlin."""
        if not self._is_configured():
            return []
        name_list = str(names).replace("[", "").replace("]", "")
        query = (
            f"g.V().has('name', within({name_list}))"
            ".valueMap('id', 'name', 'type').toList()"
        )
        results = await self.gremlin_query(query)
        return [
            {
                "id": r.get("id", [""])[0],
                "name": r.get("name", [""])[0],
                "type": r.get("type", [""])[0],
            }
            for r in results
        ]

    async def multi_hop_traverse(
        self, entity_ids: list[str], max_hops: int = 2
    ) -> dict[str, Any]:
        """Multi-hop traversal from seed entity IDs."""
        if not self._is_configured() or not entity_ids:
            return {"nodes": [], "edges": []}
        id_list = str(entity_ids).replace("[", "").replace("]", "")
        query = (
            f"g.V().has('id', within({id_list}))"
            f".repeat(both().simplePath()).times({max_hops})"
            ".path().by(valueMap('id', 'name', 'type')).toList()"
        )
        results = await self.gremlin_query(query)
        # Parse path results into nodes/edges
        nodes: dict[str, Any] = {}
        edges: list[dict[str, Any]] = []
        for path in results:
            objects = path.objects if hasattr(path, "objects") else path
            for i, obj in enumerate(objects):
                nid = obj.get("id", [""])[0] if isinstance(obj, dict) else str(obj)
                if nid not in nodes:
                    nodes[nid] = {"id": nid, "name": obj.get("name", [""])[0] if isinstance(obj, dict) else nid}
                if i > 0:
                    prev_id = objects[i - 1].get("id", [""])[0] if isinstance(objects[i - 1], dict) else str(objects[i - 1])
                    edges.append({"source": prev_id, "target": nid, "type": "RELATED_TO"})
        return {"nodes": list(nodes.values()), "edges": edges}

    async def sparql_query(self, sparql: str) -> list[dict[str, Any]]:
        """Execute a SPARQL query (RDF mode) via HTTP."""
        import asyncio
        import httpx
        if not self._is_configured():
            return []
        url = f"https://{self._settings.neptune_endpoint}:{self._settings.neptune_port}/sparql"
        async with httpx.AsyncClient() as client:
            response = await client.post(
                url,
                data={"query": sparql},
                headers={"Accept": "application/sparql-results+json"},
                timeout=30,
            )
            response.raise_for_status()
            data = response.json()
        bindings = data.get("results", {}).get("bindings", [])
        return [{k: v.get("value") for k, v in row.items()} for row in bindings]

    async def close(self) -> None:
        if self._gremlin_client:
            self._gremlin_client.close()
            self._gremlin_client = None
