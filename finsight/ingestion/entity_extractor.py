"""
Entity extractor for the ingestion pipeline.

Consumes raw document events, extracts company entities using pattern
matching, resolves them against the canonical EDGAR registry, and
publishes resolved entity events to extracted:entities.

NER strategy for Phase 2: regex patterns for ticker symbols and CIK
references. These are the highest-confidence signals in financial text.
Transformer-based NER comes in Phase 3 once we can evaluate quality
against real filings.

Entities below the confidence threshold go to provisional_entities in
Postgres for review. They never enter the graph silently.
"""

from __future__ import annotations

import logging
import re
from dataclasses import dataclass, field

import asyncpg

from finsight.ingestion.stream_backend import (
    GROUP_CHUNKERS,
    STREAM_ENTITIES,
    STREAM_FILINGS,
    StreamBackend,
)
from finsight.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

CONFIDENCE_THRESHOLD = 0.85

TICKER_PATTERN = re.compile(r"\$([A-Z]{1,5})\b|(?<!\w)([A-Z]{2,5})(?!\w)(?=\s+(?:Inc\.|Corp\.|Ltd\.|LLC|Company|Co\.))")
CIK_PATTERN = re.compile(r"\bCIK\s*(?:No\.?)?\s*(\d{7,10})\b", re.IGNORECASE)


@dataclass
class ExtractedEntity:
    """An entity pulled from a document before resolution.

    raw_text is what appeared in the filing. canonical_cik and
    canonical_name are populated after successful resolution.
    confidence reflects how certain we are about the match.
    """

    raw_text: str
    entity_type: str
    canonical_cik: str | None = None
    canonical_name: str | None = None
    confidence: float = 0.0
    source_doc_id: str = ""


@dataclass
class CanonicalEntity:
    cik: str
    official_name: str
    tickers: list[str] = field(default_factory=list)


class EntityExtractor:
    """Extracts and resolves entities from raw document events.

    Needs a Postgres connection to look up the canonical entity registry
    and write provisional entities. Pass the connection pool in so this
    is testable with a mock.
    """

    def __init__(self, backend: StreamBackend, db_pool: asyncpg.Pool) -> None:
        self._backend = backend
        self._db = db_pool
        self._ticker_to_canonical: dict[str, CanonicalEntity] = {}
        self._cik_to_canonical: dict[str, CanonicalEntity] = {}
        self._registry_loaded = False

    async def load_registry(self) -> None:
        """Load the canonical entity registry from Postgres into memory.

        Called once at startup. The registry is ~50k rows and fits
        comfortably in memory. Loading it upfront avoids a DB hit on
        every entity lookup during ingestion.
        """
        with tracer.start_as_current_span("entity_extractor.load_registry") as span:
            async with self._db.acquire() as conn:
                rows = await conn.fetch(
                    "SELECT cik, official_name, tickers FROM canonical_entities"
                )

            for row in rows:
                entity = CanonicalEntity(
                    cik=row["cik"],
                    official_name=row["official_name"],
                    tickers=[t.upper() for t in (row["tickers"] or [])],
                )
                self._cik_to_canonical[row["cik"]] = entity
                for ticker in entity.tickers:
                    self._ticker_to_canonical[ticker] = entity

            self._registry_loaded = True
            span.set_attribute("registry.size", len(self._cik_to_canonical))
            logger.info("loaded %d canonical entities", len(self._cik_to_canonical))

    async def process_document(self, doc_id: str, text: str, ticker: str) -> list[ExtractedEntity]:
        """Extract and resolve entities from a document.

        Returns only the resolved entities (confidence >= threshold).
        Low-confidence matches are written to provisional_entities and
        excluded from the returned list.

        Args:
            doc_id: Document identifier for provenance tracking.
            text: Raw document text to extract from.
            ticker: The filing company's ticker. Used to seed the
                    entity list — the filing company itself is always
                    an entity even if not mentioned by name.

        Returns:
            List of resolved ExtractedEntity objects ready to publish.
        """
        if not self._registry_loaded:
            await self.load_registry()

        with tracer.start_as_current_span("entity_extractor.process_document") as span:
            span.set_attribute("doc_id", doc_id)
            span.set_attribute("ticker", ticker)

            raw_entities = _extract_raw_entities(text)

            if ticker.upper() in self._ticker_to_canonical:
                filing_company = ExtractedEntity(
                    raw_text=ticker,
                    entity_type="company",
                    source_doc_id=doc_id,
                )
                raw_entities.insert(0, filing_company)

            resolved = []
            for entity in raw_entities:
                entity.source_doc_id = doc_id
                confidence, canonical = self._resolve_entity(entity)
                entity.confidence = confidence

                if confidence >= CONFIDENCE_THRESHOLD and canonical:
                    entity.canonical_cik = canonical.cik
                    entity.canonical_name = canonical.official_name
                    resolved.append(entity)
                else:
                    await self._write_provisional(entity, doc_id)

            span.set_attribute("entities.extracted", len(raw_entities))
            span.set_attribute("entities.resolved", len(resolved))
            return resolved

    def _resolve_entity(self, entity: ExtractedEntity) -> tuple[float, CanonicalEntity | None]:
        """Try to match an extracted entity to a canonical registry entry.

        Tries exact ticker match first (highest confidence), then CIK
        match, then falls back to returning zero confidence so the
        entity goes to the provisional queue.
        """
        text = entity.raw_text.upper().strip("$")

        if text in self._ticker_to_canonical:
            return 1.0, self._ticker_to_canonical[text]

        if text.isdigit() and text.zfill(10) in self._cik_to_canonical:
            return 1.0, self._cik_to_canonical[text.zfill(10)]

        return 0.0, None

    async def _write_provisional(self, entity: ExtractedEntity, doc_id: str) -> None:
        """Write a low-confidence entity to the provisional queue.

        These are reviewed periodically. They never enter the graph
        until manually resolved.
        """
        try:
            async with self._db.acquire() as conn:
                await conn.execute(
                    """
                    INSERT INTO provisional_entities
                        (extracted_text, best_match_cik, confidence, filing_doc_id)
                    VALUES ($1, $2, $3, $4::uuid)
                    """,
                    entity.raw_text,
                    entity.canonical_cik,
                    entity.confidence,
                    doc_id if _is_valid_uuid(doc_id) else None,
                )
        except Exception as e:
            logger.error("failed to write provisional entity %s: %s", entity.raw_text, e)

    async def publish_entities(
        self,
        doc_id: str,
        ticker: str,
        filing_type: str,
        filing_date: str,
        scopes: list[str],
        entities: list[ExtractedEntity],
    ) -> None:
        """Publish resolved entities to the extracted:entities stream."""
        for entity in entities:
            await self._backend.publish(STREAM_ENTITIES, {
                "doc_id": doc_id,
                "ticker": ticker,
                "filing_type": filing_type,
                "filing_date": filing_date,
                "scopes": scopes,
                "entity": {
                    "raw_text": entity.raw_text,
                    "entity_type": entity.entity_type,
                    "canonical_cik": entity.canonical_cik,
                    "canonical_name": entity.canonical_name,
                    "confidence": entity.confidence,
                },
            })


def _extract_raw_entities(text: str) -> list[ExtractedEntity]:
    """Pull candidate entities from text using pattern matching.

    Returns deduplicated candidates. The same ticker mentioned 40
    times in a filing is one entity, not 40.
    """
    seen: set[str] = set()
    entities: list[ExtractedEntity] = []

    for match in TICKER_PATTERN.finditer(text):
        ticker = (match.group(1) or match.group(2)).upper()
        if ticker not in seen and len(ticker) >= 2:
            seen.add(ticker)
            entities.append(ExtractedEntity(raw_text=ticker, entity_type="company"))

    for match in CIK_PATTERN.finditer(text):
        cik = match.group(1).zfill(10)
        if cik not in seen:
            seen.add(cik)
            entities.append(ExtractedEntity(raw_text=cik, entity_type="company"))

    return entities


def _is_valid_uuid(value: str) -> bool:
    import uuid
    try:
        uuid.UUID(value)
        return True
    except ValueError:
        return False