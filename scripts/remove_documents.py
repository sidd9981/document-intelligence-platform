"""
Remove whole documents from the corpus by doc_id prefix.

Deletes a document's chunks from the dense and sparse Qdrant collections,
its row in the Postgres documents table, and its Filing node in Neo4j.
Each prefix must match exactly one document. Without --yes the script
only reports what it would delete.

Usage, from the repo root with services up:
    python scripts/remove_documents.py 39c022c9 52f5ce8a
    python scripts/remove_documents.py 39c022c9 52f5ce8a --yes
"""

from __future__ import annotations

import argparse
import asyncio
import uuid

from neo4j import AsyncGraphDatabase

from finsight.gateway.db import close_pool, get_pool, init_pool
from finsight.services.vector_store import (
    close_client,
    count_document_chunks,
    delete_document,
    init_client,
    list_doc_ids,
)

NEO4J_URI = "bolt://localhost:7687"
NEO4J_AUTH = ("neo4j", "changeme")

_DELETE_FILING = """
MATCH (f:Filing {doc_id: $doc_id})
WITH f, f.doc_id AS removed
DETACH DELETE f
RETURN count(removed) AS deleted
"""


def resolve_prefixes(prefixes: list[str], doc_ids: set[str]) -> list[str]:
    """Map each prefix to exactly one full doc_id.

    Raises:
        ValueError: If a prefix matches no document or more than one.
    """
    resolved = []
    for prefix in prefixes:
        matches = sorted(d for d in doc_ids if d.startswith(prefix))
        if len(matches) != 1:
            raise ValueError(f"prefix {prefix} matches {len(matches)} documents: {matches}")
        resolved.append(matches[0])
    return resolved


async def delete_from_postgres(doc_id: str) -> str:
    """Delete the document row and return the command status."""
    async with get_pool().acquire() as conn:
        return await conn.execute("DELETE FROM documents WHERE doc_id = $1", uuid.UUID(doc_id))


async def delete_from_neo4j(driver, doc_id: str) -> int:
    """Delete the Filing node and its edges, returning how many nodes were removed."""
    async with driver.session() as session:
        result = await session.run(_DELETE_FILING, doc_id=doc_id)
        record = await result.single()
        return record["deleted"] if record else 0


async def main(prefixes: list[str], confirm: bool) -> None:
    """Resolve the prefixes, report the targets, and delete them if confirmed."""
    await init_client()
    await init_pool()
    driver = AsyncGraphDatabase.driver(NEO4J_URI, auth=NEO4J_AUTH)

    try:
        doc_ids = resolve_prefixes(prefixes, await list_doc_ids())

        for doc_id in doc_ids:
            print(f"{doc_id}: {await count_document_chunks(doc_id)}")

        if not confirm:
            print("dry run, nothing deleted. rerun with --yes to delete.")
            return

        for doc_id in doc_ids:
            await delete_document(doc_id)
            pg_status = await delete_from_postgres(doc_id)
            neo4j_deleted = await delete_from_neo4j(driver, doc_id)
            remaining = await count_document_chunks(doc_id)
            print(f"{doc_id}: qdrant now {remaining}, postgres {pg_status}, neo4j removed {neo4j_deleted}")
    finally:
        await driver.close()
        await close_pool()
        await close_client()


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Remove documents from the FinSight corpus")
    parser.add_argument("prefixes", nargs="+")
    parser.add_argument("--yes", action="store_true")
    args = parser.parse_args()
    asyncio.run(main(args.prefixes, args.yes))