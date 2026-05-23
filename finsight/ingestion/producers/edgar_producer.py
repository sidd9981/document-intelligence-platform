"""
SEC EDGAR producer.

Fetches filings from the EDGAR full-text search API and publishes
raw document events to the raw:filings stream. Everything downstream
(chunking, embedding, entity extraction) reads from that stream.

One message per filing. The message carries the raw text and all
the metadata the chunker needs so it never has to call EDGAR itself.
"""

from __future__ import annotations

import asyncio
import logging
from datetime import date, datetime

import httpx

from finsight.config.settings import settings
from finsight.ingestion.stream_backend import STREAM_FILINGS, StreamBackend
from finsight.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

EDGAR_BASE = "https://data.sec.gov"
SUBMISSIONS_URL = "{base}/submissions/CIK{cik}.json"
FILING_INDEX_URL = "{base}/Archives/edgar/data/{cik}/{accession_no_dashes}/"

SUPPORTED_FILING_TYPES = {"10-K", "10-Q", "8-K", "DEF 14A"}

# SEC asks for polite crawling. 0.1s between requests keeps us well
# under any rate limit and avoids the IP block that ruins your day.
REQUEST_DELAY_SECONDS = 0.1


class EdgarProducer:
    """Fetches SEC filings and publishes them to the ingestion stream.

    One instance per run. Pass in the stream backend so this is
    testable without a real Redis connection.
    """

    def __init__(self, backend: StreamBackend, http_client: httpx.AsyncClient) -> None:
        self._backend = backend
        self._http = http_client

    async def produce_filings(
        self,
        ticker: str,
        filing_type: str,
        start_year: int,
        end_year: int | None = None,
        scopes: list[str] | None = None,
    ) -> int:
        """Fetch all filings of a given type for a ticker and publish them.

        Args:
            ticker: Company ticker symbol, e.g. 'AAPL'.
            filing_type: One of 10-K, 10-Q, 8-K, DEF 14A.
            start_year: Earliest filing year to include.
            end_year: Latest filing year to include. Defaults to current year.
            scopes: Team IDs that can access these documents. Defaults to
                    ['public'] if not provided.

        Returns:
            Number of filings successfully published to the stream.
        """
        if filing_type not in SUPPORTED_FILING_TYPES:
            raise ValueError(f"{filing_type} is not a supported filing type. choose from {SUPPORTED_FILING_TYPES}")

        if end_year is None:
            end_year = datetime.now().year

        if scopes is None:
            scopes = ["public"]

        with tracer.start_as_current_span("edgar.produce_filings") as span:
            span.set_attribute("ticker", ticker)
            span.set_attribute("filing_type", filing_type)
            span.set_attribute("start_year", start_year)
            span.set_attribute("end_year", end_year)

            cik, company_name = await self._resolve_cik(ticker)
            span.set_attribute("cik", cik)

            filings = await self._list_filings(cik, filing_type, start_year, end_year)
            span.set_attribute("filings.found", len(filings))

            published = 0
            for filing in filings:
                try:
                    text = await self._fetch_filing_text(cik, filing)
                    if not text:
                        logger.warning("empty text for %s %s %s", ticker, filing_type, filing["date"])
                        continue

                    await self._backend.publish(STREAM_FILINGS, {
                        "ticker": ticker,
                        "cik": cik,
                        "company_name": company_name,
                        "filing_type": filing_type,
                        "filing_date": filing["date"],
                        "accession_number": filing["accession_number"],
                        "source_url": filing["source_url"],
                        "text": text,
                        "scopes": scopes,
                    })

                    published += 1
                    logger.info("published %s %s %s", ticker, filing_type, filing["date"])
                    await asyncio.sleep(REQUEST_DELAY_SECONDS)

                except Exception as e:
                    logger.error("failed to publish %s %s: %s", ticker, filing["date"], e)
                    continue

            span.set_attribute("filings.published", published)
            return published

    async def _resolve_cik(self, ticker: str) -> tuple[str, str]:
        """Look up the CIK and company name for a ticker from EDGAR.

        EDGAR's company search endpoint returns a list of matches.
        We take the first exact ticker match. If none, raise so the
        caller knows the ticker isn't in EDGAR rather than silently
        ingesting the wrong company.
        """
        with tracer.start_as_current_span("edgar.resolve_cik") as span:
            span.set_attribute("ticker", ticker)

            await asyncio.sleep(REQUEST_DELAY_SECONDS)
            response = await self._http.get(
                f"{EDGAR_BASE}/submissions/",
                params={"action": "getcompany", "company": ticker, "type": "", "dateb": "", "owner": "include", "count": "10", "search_text": ""},
            )

            # EDGAR's company tickers JSON is the reliable lookup
            tickers_response = await self._http.get(
                "https://www.sec.gov/files/company_tickers.json"
            )
            tickers_response.raise_for_status()
            tickers_data = tickers_response.json()

            for entry in tickers_data.values():
                if entry.get("ticker", "").upper() == ticker.upper():
                    cik = str(entry["cik_str"]).zfill(10)
                    company_name = entry["title"]
                    span.set_attribute("cik", cik)
                    return cik, company_name

            raise ValueError(f"ticker {ticker} not found in EDGAR company registry")

    async def _list_filings(
        self,
        cik: str,
        filing_type: str,
        start_year: int,
        end_year: int,
    ) -> list[dict]:
        """Fetch the submissions JSON for a CIK and filter to the target type and date range."""
        with tracer.start_as_current_span("edgar.list_filings") as span:
            span.set_attribute("cik", cik)
            span.set_attribute("filing_type", filing_type)

            await asyncio.sleep(REQUEST_DELAY_SECONDS)
            url = SUBMISSIONS_URL.format(base=EDGAR_BASE, cik=cik)
            response = await self._http.get(url)
            response.raise_for_status()
            data = response.json()

            filings = []
            recent = data.get("filings", {}).get("recent", {})
            forms = recent.get("form", [])
            dates = recent.get("filingDate", [])
            accessions = recent.get("accessionNumber", [])
            primary_docs = recent.get("primaryDocument", [])

            for form, filing_date_str, accession, primary_doc in zip(forms, dates, accessions, primary_docs):
                if form != filing_type:
                    continue

                filing_date = date.fromisoformat(filing_date_str)
                if not (start_year <= filing_date.year <= end_year):
                    continue

                accession_no_dashes = accession.replace("-", "")
                source_url = (
                    f"https://www.sec.gov/Archives/edgar/data/{int(cik)}/"
                    f"{accession_no_dashes}/{primary_doc}"
                )

                filings.append({
                    "date": filing_date_str,
                    "accession_number": accession,
                    "source_url": source_url,
                    "primary_doc": primary_doc,
                })

            span.set_attribute("filings.matched", len(filings))
            return filings

    async def _fetch_filing_text(self, cik: str, filing: dict) -> str | None:
        """Fetch the raw text of a filing document.

        Returns None if the fetch fails or the response is empty so the
        caller can skip rather than publishing an empty document event.
        """
        with tracer.start_as_current_span("edgar.fetch_text") as span:
            span.set_attribute("source_url", filing["source_url"])

            await asyncio.sleep(REQUEST_DELAY_SECONDS)
            try:
                response = await self._http.get(filing["source_url"])
                response.raise_for_status()
                text = response.text.strip()
                span.set_attribute("text_length", len(text))
                return text if text else None
            except httpx.HTTPError as e:
                logger.error("http error fetching %s: %s", filing["source_url"], e)
                return None


def build_producer(backend: StreamBackend) -> EdgarProducer:
    """Build an EdgarProducer with the correct User-Agent header.

    The SEC requires a descriptive User-Agent. Without it you get 403s
    or get blocked. The value comes from EDGAR_USER_AGENT in .env.
    """
    user_agent = getattr(settings, "edgar_user_agent", "FinSight research@example.com")

    http_client = httpx.AsyncClient(
        headers={"User-Agent": user_agent},
        timeout=30.0,
        follow_redirects=True,
    )

    return EdgarProducer(backend=backend, http_client=http_client)