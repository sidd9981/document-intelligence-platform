"""
Backfill SPLADE vectors for every chunk already in the dense collection.

Reads each dense point, encodes its stored content with SPLADE, and
upserts it into the sparse collection under the same point id and
payload. Safe to rerun because upserts overwrite. No re-embedding and
no duplicate chunks.

Run from the repo root with Qdrant up:
    python scripts/backfill_sparse.py
"""

from __future__ import annotations

import asyncio
import logging

from finsight.services.sparse_encoder import close_encoder, encode_sparse_batch, init_encoder
from finsight.services.vector_store import (
    close_client,
    ensure_collections_exist,
    init_client,
    scroll_dense,
    upsert_sparse_chunks,
)

logger = logging.getLogger(__name__)

BATCH_SIZE = 16


async def backfill() -> tuple[int, int]:
    """Encode and upsert sparse vectors for all dense points.

    Returns:
        Tuple of (written, skipped). Skipped points had empty content
        or produced an empty sparse vector.
    """
    await init_client()
    await ensure_collections_exist()
    init_encoder()

    written = 0
    skipped = 0
    offset = None

    try:
        while True:
            points, offset = await scroll_dense(BATCH_SIZE, offset)
            if not points:
                break

            payloads = [point.payload or {} for point in points]
            texts = [payload.get("content", "") for payload in payloads]
            vectors = await asyncio.to_thread(encode_sparse_batch, texts)

            chunks = [
                {
                    "chunk_id": str(point.id),
                    "sparse_vector": vector,
                    "content": payload.get("content", ""),
                    "metadata": {k: v for k, v in payload.items() if k != "content"},
                }
                for point, payload, vector in zip(points, payloads, vectors)
                if vector and payload.get("content")
            ]
            skipped += len(points) - len(chunks)

            if chunks:
                await upsert_sparse_chunks(chunks)
            written += len(chunks)
            logger.info("backfilled %d chunks, skipped %d", written, skipped)

            if offset is None:
                break
    finally:
        close_encoder()
        await close_client()

    return written, skipped


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s %(message)s")
    written, skipped = asyncio.run(backfill())
    print(f"sparse backfill complete: {written} written, {skipped} skipped")