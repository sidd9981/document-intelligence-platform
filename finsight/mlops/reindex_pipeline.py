"""
Blue/green re-indexing pipeline.

When a new embedding model is promoted, this pipeline:
    1. Creates a new Qdrant collection for the new model
    2. Re-embeds all documents from Postgres into the new collection
    3. Runs offline eval against the new collection
    4. If eval passes, atomically swaps the collection alias
    5. Keeps the old collection for ROLLBACK_RETENTION_HOURS as a rollback target
    6. Deletes the old collection after the retention window

The alias (filings_current) always points to the active collection.
All retrieval queries use the alias, never a versioned collection name
directly. This is what makes the swap zero-downtime.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from enum import Enum

from qdrant_client import AsyncQdrantClient
from qdrant_client.models import Distance, VectorParams

from finsight.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

COLLECTION_ALIAS = "filings_current"
ROLLBACK_RETENTION_HOURS = 24
MIN_EVAL_RECALL = 0.70


class ReindexStatus(Enum):
    PENDING = "pending"
    INDEXING = "indexing"
    EVALUATING = "evaluating"
    SWAPPED = "swapped"
    ROLLED_BACK = "rolled_back"
    FAILED = "failed"


@dataclass
class ReindexResult:
    old_collection: str
    new_collection: str
    status: ReindexStatus
    eval_recall: float
    docs_reindexed: int
    message: str


async def create_collection(
    client: AsyncQdrantClient,
    collection_name: str,
    embedding_dim: int,
) -> None:
    """Create a new Qdrant collection for the new model version."""
    await client.create_collection(
        collection_name=collection_name,
        vectors_config=VectorParams(
            size=embedding_dim,
            distance=Distance.COSINE,
        ),
    )
    logger.info("created collection %s dim=%d", collection_name, embedding_dim)


async def swap_alias(
    client: AsyncQdrantClient,
    alias: str,
    new_collection: str,
) -> None:
    """Atomically point the alias to the new collection.

    Qdrant alias updates are atomic — no requests see a gap
    between the old and new collection during the swap.
    """
    await client.update_collection_aliases(
        change_aliases_operations=[
            {
                "create_alias": {
                    "collection_name": new_collection,
                    "alias_name": alias,
                }
            }
        ]
    )
    logger.info("alias %s now points to %s", alias, new_collection)


async def delete_collection(
    client: AsyncQdrantClient,
    collection_name: str,
) -> None:
    """Delete a collection. Called after rollback retention window expires."""
    await client.delete_collection(collection_name)
    logger.info("deleted collection %s", collection_name)


async def run_reindex(
    client: AsyncQdrantClient,
    old_collection: str,
    new_collection: str,
    embedding_dim: int,
    embed_fn,
    get_documents_fn,
    eval_fn,
) -> ReindexResult:
    """Run the full blue/green re-indexing pipeline.

    Args:
        client: Qdrant async client.
        old_collection: Current production collection name.
        new_collection: Name for the new collection to create.
        embedding_dim: Embedding dimension for the new model.
        embed_fn: Async callable (text) -> list[float]. The new model's
                  embed function. Injected for testability.
        get_documents_fn: Async callable () -> list[dict] returning all
                          documents with their text and metadata.
        eval_fn: Async callable (collection_name) -> float returning
                 context recall for the collection. Must exceed
                 MIN_EVAL_RECALL for the swap to proceed.

    Returns:
        ReindexResult with final status and metrics.
    """
    with tracer.start_as_current_span("reindex_pipeline.run") as span:
        span.set_attribute("old_collection", old_collection)
        span.set_attribute("new_collection", new_collection)

        try:
            await create_collection(client, new_collection, embedding_dim)

            documents = await get_documents_fn()
            span.set_attribute("docs_total", len(documents))

            from qdrant_client.models import PointStruct

            points = []
            for doc in documents:
                embedding = await embed_fn(doc["text"])
                points.append(PointStruct(
                    id=doc["id"],
                    vector=embedding,
                    payload=doc.get("metadata", {}),
                ))

            if points:
                await client.upsert(
                    collection_name=new_collection,
                    points=points,
                    wait=True,
                )

            logger.info("indexed %d documents into %s", len(points), new_collection)

            eval_recall = await eval_fn(new_collection)
            span.set_attribute("eval_recall", eval_recall)

            if eval_recall < MIN_EVAL_RECALL:
                logger.error(
                    "reindex eval failed: recall=%.3f threshold=%.3f — aborting swap",
                    eval_recall,
                    MIN_EVAL_RECALL,
                )
                await delete_collection(client, new_collection)
                return ReindexResult(
                    old_collection=old_collection,
                    new_collection=new_collection,
                    status=ReindexStatus.FAILED,
                    eval_recall=eval_recall,
                    docs_reindexed=len(points),
                    message=f"eval recall {eval_recall:.3f} below threshold {MIN_EVAL_RECALL}",
                )

            await swap_alias(client, COLLECTION_ALIAS, new_collection)

            return ReindexResult(
                old_collection=old_collection,
                new_collection=new_collection,
                status=ReindexStatus.SWAPPED,
                eval_recall=eval_recall,
                docs_reindexed=len(points),
                message=f"swapped alias {COLLECTION_ALIAS} to {new_collection}",
            )

        except Exception as e:
            logger.error("reindex pipeline failed: %s", e)
            return ReindexResult(
                old_collection=old_collection,
                new_collection=new_collection,
                status=ReindexStatus.FAILED,
                eval_recall=0.0,
                docs_reindexed=0,
                message=str(e),
            )