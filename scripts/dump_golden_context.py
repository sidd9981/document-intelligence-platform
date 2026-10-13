"""
Dump the corpus passages most relevant to each golden query.

For each golden item, embeds the query, searches the dense collection
restricted to the item's ticker, and writes the top passages to a text
file with HTML entities decoded. Used to write reference answers from
text that is actually in the corpus.

Passages chosen by the retriever can bias reference answers toward what
the retriever already finds, so treat the output as a starting point
rather than the full picture.

Usage, from the repo root with services up:
    python scripts/dump_golden_context.py
"""

from __future__ import annotations

import asyncio
import html
import json
from pathlib import Path

from finsight.services.llm import close_client, embed, init_client
from finsight.services.vector_store import close_client as close_qdrant
from finsight.services.vector_store import init_client as init_qdrant
from finsight.services.vector_store import search_dense

ROOT = Path(__file__).parent.parent
GOLDEN_PATH = ROOT / "eval" / "golden_dataset" / "queries.json"
OUTPUT_PATH = ROOT / "eval" / "golden_context.txt"
SEARCH_TEAM = "analysis"
TOP_K = 6
MAX_CHARS = 900


def _clean(text: str) -> str:
    """Decode HTML entities and non-breaking spaces left over from ingestion."""
    return html.unescape(text).replace("\xa0", " ")


async def main() -> None:
    """Write the top passages for every golden query to OUTPUT_PATH."""
    await init_client()
    await init_qdrant()

    items = json.loads(GOLDEN_PATH.read_text())
    lines: list[str] = []

    try:
        for item in items:
            lines.append(f"=== {item['id']} | {item['query']}")
            chunks = await search_dense(
                query_embedding=await embed(item["query"]),
                team_id=SEARCH_TEAM,
                k=TOP_K,
                ticker=item.get("ticker"),
            )
            for chunk in chunks:
                section = _clean(chunk.metadata.section)[:40]
                lines.append(f"--- {chunk.metadata.filing_date} {section} score={chunk.score:.3f}")
                lines.append(_clean(chunk.content)[:MAX_CHARS])
            lines.append("")
    finally:
        await close_qdrant()
        await close_client()

    OUTPUT_PATH.write_text("\n".join(lines))
    print(f"wrote {len(items)} queries to {OUTPUT_PATH}")


if __name__ == "__main__":
    asyncio.run(main())