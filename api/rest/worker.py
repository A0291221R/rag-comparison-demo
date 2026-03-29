"""
api/rest/worker.py — Async Redis Streams worker.

Decouples API from inference for throughput scaling.
Stream: rag.query.requested → [worker] → rag.answer.generated

Run alongside the API: make run-worker
"""
from __future__ import annotations

import asyncio
import json
import time
import uuid

import structlog

logger = structlog.get_logger(__name__)

STREAM_IN = "rag.query.requested"
STREAM_OUT = "rag.answer.generated"
STREAM_DLQ = "rag.query.dead_letter"
CONSUMER_GROUP = "rag-workers"
CONSUMER_NAME = f"worker-{uuid.uuid4().hex[:8]}"
MAX_RETRIES = 3


async def process_message(
    redis: Any,  # type: ignore
    message_id: str,
    data: dict,
) -> None:
    """Process a single queued query message."""
    from orchestrator.graph import get_orchestrator

    query = data.get("query", "")
    pipeline = data.get("pipeline", "auto")
    session_id = data.get("session_id")
    retry_count = int(data.get("retry_count", 0))

    logger.info(
        "worker_processing",
        message_id=message_id,
        query=query[:60],
        pipeline=pipeline,
        retry=retry_count,
    )

    try:
        orchestrator = get_orchestrator()
        result = await orchestrator.run(
            query=query,
            mode=pipeline,  # type: ignore
            session_id=session_id,
        )
        # Publish result to output stream
        await redis.xadd(
            STREAM_OUT,
            {
                "original_message_id": message_id,
                "trace_id": result.get("trace_id", ""),
                "pipeline_used": result.get("pipeline_used", ""),
                "final_answer": result.get("final_answer", ""),
                "token_usage": json.dumps(result.get("token_usage", {})),
                "timestamp": str(time.time()),
            },
        )
        # ACK message
        await redis.xack(STREAM_IN, CONSUMER_GROUP, message_id)
        logger.info("worker_message_acked", message_id=message_id)

    except Exception as exc:
        logger.error("worker_processing_failed", error=str(exc), message_id=message_id)
        if retry_count < MAX_RETRIES:
            # Re-queue with incremented retry count
            await redis.xadd(
                STREAM_IN,
                {
                    **data,
                    "retry_count": str(retry_count + 1),
                    "last_error": str(exc),
                },
            )
            await redis.xack(STREAM_IN, CONSUMER_GROUP, message_id)
        else:
            # Move to dead letter queue
            await redis.xadd(
                STREAM_DLQ,
                {**data, "final_error": str(exc), "message_id": message_id},
            )
            await redis.xack(STREAM_IN, CONSUMER_GROUP, message_id)
            logger.warning("message_sent_to_dlq", message_id=message_id)


async def run_worker() -> None:
    from core.config import settings
    import redis.asyncio as aioredis

    r = aioredis.from_url(settings.redis_url)

    # Create consumer group if not exists
    try:
        await r.xgroup_create(STREAM_IN, CONSUMER_GROUP, id="0", mkstream=True)
        logger.info("consumer_group_created", group=CONSUMER_GROUP)
    except Exception:
        pass  # Group already exists

    logger.info("worker_started", consumer=CONSUMER_NAME, stream=STREAM_IN)

    while True:
        try:
            messages = await r.xreadgroup(
                CONSUMER_GROUP,
                CONSUMER_NAME,
                {STREAM_IN: ">"},
                count=10,
                block=2000,  # 2s block timeout
            )
            if not messages:
                continue

            tasks = []
            for stream_name, stream_messages in messages:
                for message_id, fields in stream_messages:
                    data = {
                        k.decode() if isinstance(k, bytes) else k:
                        v.decode() if isinstance(v, bytes) else v
                        for k, v in fields.items()
                    }
                    tasks.append(process_message(r, message_id.decode(), data))

            if tasks:
                await asyncio.gather(*tasks, return_exceptions=True)

        except asyncio.CancelledError:
            break
        except Exception as exc:
            logger.error("worker_loop_error", error=str(exc))
            await asyncio.sleep(1)

    await r.aclose()


if __name__ == "__main__":
    from typing import Any  # noqa
    from observability.logging import setup_logging
    from observability.tracing import setup_tracing, setup_langsmith
    setup_logging()
    setup_tracing()
    setup_langsmith()
    asyncio.run(run_worker())
