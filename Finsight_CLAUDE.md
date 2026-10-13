# FinSight — Enterprise Document Intelligence Platform

## What It Is

FinSight makes any document in a financial institution's corpus queryable in plain English. An analyst asks "which portfolio companies disclosed TSMC supply chain dependency in the last two quarters?" and gets a cited, auditable answer in seconds instead of spending half a day manually reading filings.

Every answer is grounded — every claim maps to a specific paragraph in a specific document. Every answer is auditable — a judge pipeline flags any claim not supported by the retrieved context. Every answer is governed — different teams see different documents, consume different compute budgets, and get different model tiers.

**One sentence:** FinSight is an enterprise document intelligence platform — query any document in your corpus in plain English and get a cited, auditable, governed answer in seconds.

---

## Who Uses It

Three teams with meaningfully different profiles. Multi-tenancy is not a nice-to-have — it is a load-bearing architectural concern that drives the token governance, data isolation, priority queuing, and per-tenant observability design.

| Team | Who | What they ask | Stakes |
|---|---|---|---|
| Analysis | Senior analysts, portfolio managers | Multi-document synthesis, year-over-year comparisons, relationship-dependent queries | Investment decisions. Wrong answer costs money. |
| Risk | Risk managers, compliance officers | Structured extraction, compliance checks, portfolio screening | Regulatory consequences. Wrong answer costs licenses. |
| Ops | Junior analysts, operations staff | Fast factual lookups | Operational efficiency. Wrong answer wastes time. |

**Resource governance per team:**

| Team | Priority | Daily token budget | Context window | Model tier |
|---|---|---|---|---|
| Analysis | P1 | 2M tokens | 64k | llama3.1:8b |
| Risk | P2 | 800k tokens | 32k | llama3.1:8b |
| Ops | P3 | 200k tokens | 8k | llama3.2:3b |

---

## Architecture

```
[User query + team_id]
         │
         ▼
[FastAPI Gateway]          ← validates JWT, loads tenant config, runs guardrails
         │
         ▼
[LangGraph Orchestrator]   ← classifies intent, extracts entities, builds plan
         │
    ┌────┴──────────────┐
    ▼                   ▼
[Retrieval Agent]   [Graph Agent]      ← run in parallel
    │                   │
[Qdrant Dense]     [Neo4j Cypher]
[Qdrant Sparse]
[Cross-Encoder]
    │                   │
    └────────┬──────────┘
             ▼
    [Input Harness]       ← context validation, position bias fix, injection detection
             │
             ▼
    [Synthesis Agent]
             │
         [Ollama LLM]
             │
             ▼
    [Output Harness]      ← faithfulness check, hallucination flags, PII scrub
             │
             ▼
    [Meter tokens → Redis]
    [Write cache → Redis]
    [Log trace → Postgres + Langfuse]
             │
             ▼
    [Response to client]
```

---

## Data

**Sources ingested:**
- SEC EDGAR 10-K, 10-Q, 8-K, DEF 14A filings — free public API
- Current corpus: AAPL, MSFT, TSLA — 2022 to 2026
- 1,939 chunks indexed in Qdrant
- 4 company nodes, 10 filing nodes, 11 MENTIONED_IN relationships in Neo4j

**Chunking strategy — one chunker per document type:**
- SEC filings: section-aware splitting at Item boundaries (Item 1, 1A, 7, 8). Tables extracted as single chunks, never split.
- Earnings transcripts: speaker-turn splitting with speaker metadata prepended
- News articles: paragraph-level splitting with source and headline prepended

**Why document-type-specific chunkers matter:** Generic recursive text splitters break on financial documents. A 10-K risk factors section split mid-paragraph loses the regulatory context that makes the chunk meaningful. Section-aware chunking preserves the semantic unit.

---

## Retrieval

Three modalities, fused with Reciprocal Rank Fusion.

**Dense vector search (Qdrant):** Semantic similarity. "Supply chain concentration risk" matches paragraphs about TSMC even if TSMC is not mentioned explicitly. Model: nomic-embed-text (768-dim), upgradeable to BAAI/bge-m3 (1024-dim).

**Sparse vector search (Qdrant SPLADE):** Keyword matching. Catches exact financial terms — ticker symbols, CIK numbers, specific clause references. Model: naver/splade-cocondenser-selfdistil. Replaces Elasticsearch locally with equivalent retrieval quality.

**Graph traversal (Neo4j Cypher):** Relationship-based retrieval. TSMC supplies Apple. Apple competes with Samsung. This is structurally impossible with vector search — you cannot search for a relationship type, only for text similarity.

**RRF fusion:** `score = 1/(rank + 60)`. Rank-based fusion is robust to score distribution differences between dense and sparse methods. Score-based fusion would require calibrating scales across three different models — fragile and expensive to maintain.

**Cross-encoder reranking:** After RRF, top-50 candidates are rescored by a cross-encoder (ms-marco-MiniLM-L-6-v2) that attends to query and chunk jointly. More accurate than retrieval scores alone. Only run on top-50, never the full corpus.

---

## Inference Harness

The component that separates a production system from a demo. Every LLM call passes through three layers.

**Input harness (before every LLM call):**
1. Context quality check — is the max retrieval score above 0.5? If not, flag low-confidence.
2. Token budget enforcement — trim lowest-scoring chunks until the prompt fits the tenant's context window.
3. Position bias reordering — best chunk first, second-best last, rest in the middle. Counteracts the "lost in the middle" attention pattern.
4. Prompt injection detection — scan retrieved chunks for adversarial content attempting to hijack the prompt.
5. Prompt version stamping — every trace records which prompt template was running.

**Output harness (after every LLM response):**
1. Schema validation — does the response parse into SynthesisResult? Retry once with explicit format instruction on failure.
2. Citation extraction — map each claim to a specific chunk ID and document ID.
3. Faithfulness check — LLM-as-judge: "given this context and this answer, which claims are not supported?" Returns score (0-1) and list of specific unsupported claims.
4. Hallucination flagging — any number in the answer not present verbatim in the retrieved chunks is flagged. Financial hallucinations are the highest-risk failure mode.
5. PII scrubbing — strip SSN and email patterns before returning to client.

**Eval harness (async, 10% of live traffic):**
- RAGAS-equivalent metrics logged to Langfuse per query
- Faithfulness score, context recall, answer relevancy
- Prompt version attribution on every trace

---

## Multi-Agent Architecture

One orchestrator, three specialist agents. The orchestrator/specialist pattern maps to real production agentic systems.

**Orchestrator (LangGraph StateGraph):** Classifies intent, extracts entities, fans out to agents in parallel, merges results, runs harnesses, meters tokens, writes cache, logs audit trail, streams response.

**Retrieval Agent:** Owns the full retrieval pipeline — cache lookup, hybrid search, RRF fusion, reranking. Never raises. Returns empty chunks with structured errors on failure.

**Graph Agent:** Queries Neo4j for entity relationships. Runs in parallel with the retrieval agent. Returns fallback=True if Neo4j is unavailable — the orchestrator continues with vector-only retrieval.

**Synthesis Agent:** Builds context, calls LLM, receives response. Harness wraps this agent — the agent itself just does LLM I/O.

**Why LangGraph:** Explicit state machine with typed transitions. Every conditional edge is code you write. Failure modes are designed, not accidental. If Neo4j times out, the conditional edge routes to vector-only, not an undefined crash.

**Agent state (single source of truth):**
```python
class AgentState(TypedDict):
    query: str
    team_id: str
    tenant_config: TenantConfig
    trace_id: str
    intent: str
    entities: list[str]
    retrieval_result: RetrievalResult | None
    graph_result: GraphResult | None
    synthesis_result: SynthesisResult | None
    context_quality_score: float | None
    prompt_version: str | None
    retry_count: int
    tokens_used: int
    errors: list[AgentError]
    cache_hit: bool
    final_response: QueryResponse | None
```

---

## MCP Server Architecture

Each tool is an independently deployable FastAPI service. Agents discover tools at runtime via the registry. Adding a new data source means spinning up a new MCP server — no agent code changes.

| Server | Port | Scope required | Tools |
|---|---|---|---|
| Registry | 8101 | None | Tool discovery, manifest |
| Qdrant | 8102 | read:public_filings or read:all_filings | vector_search, sparse_search |
| Neo4j | 8103 | query:graph | entity_lookup, cypher_query |
| LLM | 8104 | model:small/medium/large | generate |
| Reranker | 8105 | read:public_filings or read:all_filings | rerank |

Every MCP server validates the JWT independently — defense in depth on top of the gateway scope check.

---

## OAuth2 + Multi-Tenant Data Isolation

Machine-to-machine auth. Client credentials grant. No user login flows.

**Three independent data isolation layers:**

1. **OAuth scope** — token without `read:all_filings` cannot request that data at the gateway
2. **Qdrant payload filter** — `must: team_id IN chunk.scopes` on every vector query
3. **Neo4j Cypher** — `WHERE $team_id IN node.scopes OR 'public' IN node.scopes` on every query

All three layers must be bypassed simultaneously for a data leak. This is what you explain to a compliance officer.

**Scope definitions:**
```
read:all_filings     → Analysis team
read:public_filings  → Risk, Ops teams
query:graph          → Analysis, Risk teams
model:large          → Analysis team
model:medium         → Analysis, Risk teams
model:small          → All teams
```

---

## Token Governance

**Budget enforcement (Redis):**
```python
key = f"budget:{team_id}:{date.today().isoformat()}"
current = int(await redis.get(key) or 0)
if current + estimated_tokens > tenant.daily_token_budget:
    raise BudgetExceededError(retry_after=seconds_until_midnight())
```

**Priority queue:** Three Redis sorted sets (queue:p1, queue:p2, queue:p3). Dequeue policy: 8 from P1, 2 from P2, 1 from P3. Weighted fair queuing — Analysis always served first, Ops never starved.

**Semantic cache (Redis):** Key: `cache:{team_id}:{sha256(query)[:16]}`. TTL: 6 hours for news, 24 hours for filings. Cache hit = 0 tokens charged. At steady state eliminates 40-60% of LLM calls.

---

## Ingestion Pipeline

```
SEC EDGAR API
    │
    ▼
Redis Streams producer (partitioned by ticker)
    │
    ▼
EmbeddingWorker (consumer group: embedders)
    ├── Chunks document with type-specific chunker
    ├── Embeds each chunk with nomic-embed-text
    ├── Publishes to embedded:chunks stream
    └── Runs EntityExtractor inline
         └── Publishes to extracted:entities stream
    │
    ├── QdrantWriter (consumer group: qdrant-writers)
    │   └── Upserts embedded chunks to Qdrant
    │
    └── Neo4jWriter (consumer group: neo4j-writers)
        └── MERGEs Company and Filing nodes, MENTIONED_IN relationships
```

**Redis Streams locally, Kafka in production.** The `StreamBackend` protocol abstraction means this is a one-line config change. The same consumer group semantics, offset tracking, and DLQ patterns work identically.

**Entity resolution — three layers:**
1. Canonical EDGAR company registry (10k+ companies, loaded from SEC API)
2. Fuzzy matching with confidence threshold — confidence >= 0.85 merges to canonical node
3. Provisional queue — low-confidence matches go to `provisional_entities` table for review, never silently enter the graph

**Dead letter queue:** Max 3 retries. On 3rd failure, message goes to `ingestion:dlq` stream. Alert fires if DLQ depth > 10. Retained 30 days.

---

## Guardrails

**Query-level (before any LLM call, zero token cost):**
- Prompt injection detection — scans query for `ignore previous instructions`, `jailbreak`, `DAN mode`, etc.
- Investment advice rejection — rejects buy/sell recommendations, price targets, price predictions. Legal exposure in finance domain.

**Document-level (input harness):**
- Indirect injection detection — scans retrieved chunks for adversarial content in financial filings

**Output-level (output harness):**
- Hallucination flagging — numbers in answer not present in context
- PII scrubbing — SSN and email patterns stripped
- Faithfulness threshold — answers below 0.85 get a warning attached

---

## Reliability

**Circuit breakers on every MCP call:**
- Opens after 5 consecutive failures
- Recovery timeout: 30 seconds
- Half-open state: single probe, success closes, failure restarts timeout
- Prometheus metric: `finsight_circuit_breaker_opens_total`

**Graceful degradation matrix:**

| What fails | System behavior |
|---|---|
| Neo4j | Continue with vector+sparse only |
| Qdrant vector search | Fall back to sparse-only |
| Cross-encoder reranker | Use raw RRF scores |
| Redis cache | Bypass cache, proceed to LLM |
| LLM 8b tier | Route to 3b, log downgrade |

**SLOs:**
- P95 query latency (Analysis): < 8 seconds
- P95 query latency (Ops): < 2 seconds
- API gateway availability: 99.5%
- Faithfulness score: >= 0.85 (7-day rolling)

---

## Observability

**OpenTelemetry from Phase 1.** Every external service call is wrapped in a span from day one. Not added after the fact.

**Prometheus metrics:**
- `finsight_query_latency_seconds` — P50/P95/P99 by team and intent
- `finsight_tokens_used_total` — by team and model
- `finsight_cache_hits_total` / `finsight_cache_misses_total` — by team
- `finsight_faithfulness_score` — rolling gauge by team
- `finsight_circuit_breaker_opens_total` — by service name
- `finsight_budget_utilization_pct` — by team
- `finsight_dlq_depth` — ingestion health

**Langfuse RAG traces:** Every eval-sampled query logged with full span tree, retrieved chunks with scores, prompt version, LLM response, and RAGAS-equivalent scores. Faithfulness failures logged as structured data — specific unsupported claim text, not just a score.

**Grafana dashboards:** Query latency P50/P95 by team, cache hit/miss rate, faithfulness score, token usage rate, circuit breaker opens, ingestion lag, DLQ depth, provisional entities.

---

## MLOps

**MLflow model registry:** Every embedding model version logged with context recall and context precision at registration time. Promotion requires >= 2% context recall improvement over current production model.

**Zero-downtime re-indexing:** Create new Qdrant collection, re-embed all documents in background, run offline eval, atomically swap collection alias, keep old collection 24 hours for rollback.

**Drift detection:** Weekly job samples recent queries, reruns retrieval, compares context recall against 4-week baseline. Drop > 10% triggers alert.

**Prompt versioning:** Prompts live as versioned files in `prompts/`. Every Langfuse trace records `prompt_version`. When faithfulness drops you immediately see if a prompt change caused it.

---

## Evaluation

**Offline eval (CI gate):** 15 queries across AAPL, MSFT, TSLA — risk extraction, analytical synthesis, factual lookup, adversarial. Runs on every PR touching retrieval or agent code. Blocks merge on regression.

**Current scores (13 non-adversarial queries):**

| Metric | Score | Threshold |
|---|---|---|
| Context Precision | 0.615 | 0.60 |
| Context Recall | 0.677 | 0.60 |
| Faithfulness | 0.809 | 0.75 |
| Answer Relevancy | 0.900 | 0.60 |

**What these measure:**
- Context Precision: of retrieved chunks, what fraction were relevant?
- Context Recall: of needed information, what fraction was retrieved?
- Faithfulness: of answer claims, what fraction are grounded in context? (LLM-as-judge)
- Answer Relevancy: does the answer address what was asked?

**Known limitations:**
- Corpus is 3 companies (AAPL, MSFT, TSLA). Production would be S&P 500.
- Analytical multi-filing queries underperform because the eval pipeline does not use graph context. Graph retrieval is wired in the production orchestrator but not in the standalone eval script.
- Item 1 business description sections are underrepresented in the ingested corpus relative to Item 1A risk factors.

**Online eval (production monitor):** 10% of live queries sampled async post-response. RAGAS metrics logged to Langfuse. Alert fires if any metric drops below threshold for 2 consecutive days.

---

## CI/CD

**GitHub Actions pipeline:**
- Every PR: unit tests (312 tests, no services needed)
- Every merge to main: integration tests + RAGAS gate against live services
- Every merge to main: Docker image build and push
- Weekly schedule: chaos tests

**Chaos tests verify the degradation matrix works:**
- Kill Qdrant → retrieval agent returns empty chunks, not exception
- Kill Neo4j → graph agent returns fallback=True, orchestrator continues
- Trip circuit breaker → subsequent calls fail fast, probe succeeds after timeout
- Kill Redis → cache miss, retrieval continues normally

---

## Tech Stack

| Concern | Choice | Why not X |
|---|---|---|
| Vector store | Qdrant | Native payload filtering for tenant isolation. Supports sparse vectors — eliminates Elasticsearch locally. |
| Graph DB | Neo4j | Cypher is the most expressive graph query language for relationship traversal. |
| Agent framework | LangGraph | Explicit state machine = debuggable failure modes. Every edge is code you control. |
| Retrieval fusion | RRF | Robust to score scale differences between dense and sparse. No calibration needed. |
| Local LLM | Ollama | OpenAI-compatible API. One env var to swap models. |
| Event streaming | Redis Streams | Kafka-compatible semantics. One config line to swap in production. |
| Auth | OAuth2 client credentials | Correct M2M pattern. Scopes flow through every layer. |
| RAG observability | Langfuse | Purpose-built for RAG traces, chunk-level scoring, prompt attribution. |
| Model registry | MLflow | Experiment tracking, promotion gates, embedding version history. |
| Metrics | Prometheus + Grafana | Industry standard. Scrapes from gateway /metrics endpoint. |
| Tracing | OpenTelemetry | Vendor-neutral. From Phase 1, not retrofitted. |

---

## Repository Structure

```
finsight/
├── agents/               LangGraph orchestrator + 3 specialist agents
├── harness/              Input, output, eval harness
├── mcp_servers/          5 independently deployable MCP servers
├── auth/                 OAuth server, JWT validator, scope definitions
├── ingestion/            Redis Streams pipeline, chunkers, embedder, writers
├── entity_resolution/    3-layer entity resolution against EDGAR registry
├── gateway/              FastAPI app, token broker, tenant config
├── models/               Pydantic contracts — no logic, only types
├── services/             Qdrant, LLM, circuit breaker, guardrails, metrics
├── mlops/                Model registry, drift detector, reindex pipeline, prompt registry
├── eval/                 Golden dataset, offline eval, RAGAS metrics
├── prompts/              Versioned prompt templates
├── telemetry/            OTEL tracer setup
├── tests/                312 unit tests + chaos tests
├── scripts/              Migration, seeding, ingestion runner
├── infra/                Docker Compose, Prometheus, Grafana dashboards
└── frontend/             React + TypeScript + Vite frontend
```

---

## React + TypeScript Frontend

Built with Vite 5.4.0, React, TypeScript, and Tailwind CSS. Proxies `/api/*` to the gateway at `http://localhost:8080`.

**Features:**
- Team login screen with OAuth2 client credentials — Analysis, Risk, Ops
- Auth persisted in localStorage with JWT expiry check — auto-login on refresh
- Multi-conversation support — create, switch, delete conversations
- Conversations persisted per-team in localStorage
- Conversation title auto-derived from first query
- Last 1 message sent as history for conversational follow-ups
- Dynamic model display — shows actual model used from response, not hardcoded
- Budget consumption bar — fetches `GET /budget/{team_id}` after each query, green/yellow/red
- Faithfulness badge per response (green >= 0.85, yellow >= 0.70, red < 0.70)
- Cache hit indicator, latency display, citation cards

**Component structure:**
```
frontend/src/
├── types/index.ts           TypeScript types for API responses
├── api/client.ts            getToken, sendQuery (with history), getBudget
├── hooks/useChat.ts         Multi-conversation state, localStorage, history
├── components/
│   ├── LoginScreen.tsx      Team picker, OAuth2 client credentials
│   ├── Sidebar.tsx          Team badge, conversation list, budget bar, model
│   ├── ChatInput.tsx        Auto-resize textarea, Enter to submit
│   ├── MessageBubble.tsx    Query + answer + faithfulness badge + citations
│   ├── CitationCard.tsx     Citation with confidence score
│   └── FaithfulnessBadge.tsx Colored confidence indicator
└── App.tsx                  Auth persistence, ChatView, conversation routing
```

**Run:**
```bash
cd frontend
npm run dev
# Open http://localhost:5173
```

---

## Known Constraints and Fixes Applied

**asyncpg SSL issue (post Docker restart):**
asyncpg attempts SSL by default which fails against Dockerized Postgres. Fixed with `ssl=False` in `asyncpg.create_pool()` in `finsight/gateway/db.py`. asyncpg pinned to 0.29.0.

**LangGraph pinned to 0.2.72:**
Versions above 0.2.72 introduced a `Reviver(allowed_objects="core")` call incompatible with the installed langchain-core version.

**llama3.1:8b OOM on 18GB unified memory:**
With Docker at 6GB + SPLADE + reranker + all services running, llama3.1:8b crashes at ~8700 prompt tokens. Local dev uses llama3.2:3b. Production uses dedicated GPU with vLLM.

**Analysis team context window reduced for local dev:**
```sql
UPDATE tenant_configs SET max_context_tokens = 8000, max_output_tokens = 500, allowed_models = '{small}' WHERE team_id = 'analysis';
```

**Docker memory:** Reduced to 6GB in Docker Desktop settings.

**Langfuse v4.6.1 API:** Old `.trace()` method removed. Fixed in offline_eval.py using `lf.start_observation()` + `obs.score_trace()`. eval_harness.py in gateway still needs same fix.

**RAGAS eval dataset expanded:** 10 AAPL-only queries → 15 queries across AAPL, MSFT, TSLA. Adversarial queries excluded from metric averages. Current passing scores: cp=0.615, cr=0.677, f=0.809, ar=0.900.

**Budget endpoint added:** `GET /budget/{team_id}` returns `{used, limit, pct}` from Redis. Frontend polls after each query.

**Vite rolldown arm64 issue:** Vite 8 has broken arm64 native binding on macOS. Downgraded to Vite 5.4.0.