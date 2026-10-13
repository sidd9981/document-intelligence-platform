# FinSight — Decisions Log & Progress Tracker

> This document records every significant decision made during the design phase, why it was made, what was considered and rejected, and the current build status. Open this alongside `finsight_project_scope.md` when starting a new session.

---

## Current Build Status

| Phase | Status | Notes |
|---|---|---|
| 1 — Local skeleton | **Not started** | Start here — see scope doc section 0 |
| 2 — Ingestion pipeline | Not started | |
| 3 — Multi-modal retrieval | Not started | |
| 4 — Multi-agent orchestration | Not started | |
| 5 — Serving + MCP + OAuth | Not started | |
| 6 — Reliability + observability | Not started | |
| 6.5 — MLOps | Not started | |
| 7 — CI/CD + production hardening | Not started | |

---

## What We Have Done So Far

As of this session: **full design and scoping only**. Zero code written. The value of this phase is that every decision below has a reason — we won't re-debate them mid-build, which is where most projects die.

---

## Decision Log

Each entry answers: what did we decide, what did we consider, why did we pick this, and what does it cost us.

---

### D-01 — Project domain: Finance

**Decision:** Build in the financial sector (SEC filings, earnings calls, news).

**Why:** Finance has hard requirements that force you to actually learn each system design concept rather than getting away with shortcuts. Compliance needs audit trails → forces proper distributed logging. Latency SLAs → forces caching design. Regulatory documents + structured data → forces multi-modal RAG. Corpus size → forces real scaling decisions. A generic "document Q&A" project lets you skip all of this.

**What we rejected:** Healthcare (HIPAA compliance complexity is real and adds legal risk to a portfolio project), Legal (similar), generic enterprise docs (too vague, no forcing function for the hard engineering).

**Cost:** None — finance data is free via SEC EDGAR.

---

### D-02 — Multi-tenant: Three teams

**Decision:** Three teams (Analysis, Risk, Ops) with different token budgets, model tiers, context windows, data scopes, and priority levels.

**Why:** Multi-tenancy is a senior engineering concern that most RAG portfolio projects skip entirely. It forces you to design token governance, data isolation, priority queuing, and per-tenant observability — all of which come up in senior interviews. Having three teams with meaningfully different profiles (64k context vs 8k, 70B model vs 8B) makes the design non-trivial.

**What we rejected:** Single tenant (too simple, misses the governance story), more than 3 teams (adds config complexity without adding new design patterns).

**Cost:** Adds the token broker, Redis budget counters, priority queue, and OAuth scope enforcement. Worth it.

---

### D-03 — Agent architecture: Orchestrator + specialists

**Decision:** One orchestrator agent (LangGraph StateGraph) dispatching to three specialist agents (Retrieval, Graph, Synthesis), each owning their own tools via MCP.

**Why:** The orchestrator/specialist pattern maps directly to real production agentic systems. It demonstrates: state machine design, parallel fan-out, typed inter-agent contracts, and graceful degradation when one agent fails. A single monolithic agent would be simpler but demonstrates nothing about distributed agent coordination.

**What we rejected:** Single agent with all tools (too simple), debate-style agents like PipelineAgents (doesn't fit the RAG query pattern naturally), CrewAI/AutoGen (less control over failure handling, less production-credible).

**Cost:** More boilerplate. Each agent needs its own file, MCP client setup, and typed return model. Worth it — this is exactly the boilerplate a senior engineer writes.

---

### D-04 — Three retrieval modalities

**Decision:** Vector search (Qdrant dense) + sparse vector search (Qdrant SPLADE) + Graph traversal (Neo4j Cypher), fused with RRF.

**Why:** The canonical demo query — "how has Apple's TSMC supply chain exposure changed across 6 10-K filings and which competitors mention the same risk?" — cannot be answered by any single modality. It requires graph traversal (TSMC → Apple → competitor relationships), semantic search (supply chain risk language), and potentially keyword matching (exact ticker symbols). This is the core technical claim of the project.

**What we rejected:** Vector-only (can't do multi-hop graph queries), graph-only (can't do semantic similarity), BM25-only (misses semantic paraphrasing).

**Cost:** Neo4j adds ~500 MB locally, requires entity extraction during ingestion, and Cypher query design. The graph retriever is the hardest component to build correctly.

---

### D-05 — MCP for tool serving

**Decision:** Each agent tool is an independently deployable MCP server. Agents discover tools at runtime via the MCP registry.

**Why:** MCP makes the system a platform, not a monolith. Adding a new data source (Bloomberg, Fed data) means spinning up a new MCP server and updating tenant scope config — no agent code changes. It also demonstrates protocol-level thinking, which very few portfolio projects do. Each MCP server validates OAuth tokens independently, giving us defense-in-depth security.

**What we rejected:** Hardcoded tool functions in agent code (simpler but tightly coupled, not a platform), LangChain tools (framework-specific, less transferable to interviews).

**Cost:** 5 custom FastAPI services to write and maintain. Each needs a manifest endpoint, invoke endpoint, health check, and OAuth middleware. Significant Phase 5 work.

---

### D-06 — OAuth2 client credentials

**Decision:** Machine-to-machine OAuth2 with client credentials grant. Scopes carried in JWT, validated at gateway and at every MCP server.

**Why:** Client credentials is the correct enterprise pattern for service-to-service auth. The scope system (`read:all_filings`, `query:graph`, `model:large`) flowing through every layer gives us data isolation enforced at multiple independent layers — the definition of defense in depth. This is what a bank would actually deploy.

**What we rejected:** API keys only (can't carry structured scopes natively), session/user auth (wrong pattern for machine-to-machine), no auth in dev (bad habit that creates prod gaps).

**Cost:** Requires an OAuth server, JWT validation middleware, and scope enforcement in every MCP server. Phase 5 work.

---

### D-07 — Vector store: Qdrant

**Decision:** Qdrant for both dense vector search and sparse vector search (SPLADE).

**Why:** Qdrant supports native payload filtering — every chunk has `scopes: list[str]` in its payload, and every search query includes a filter on scopes. This gives us tenant data isolation at the vector index level with zero application-layer workaround. It also supports sparse vectors natively, eliminating the need for a separate Elasticsearch instance locally.

**What we rejected:** Pinecone (managed/costly, no self-hosting, harder to demo), FAISS (no payload filtering, no persistence, no sparse vectors), Weaviate (heavier footprint, more complex ops).

**Cost:** None meaningful. Qdrant is the right tool for this use case.

---

### D-08 — Elasticsearch replaced by Qdrant sparse vectors locally

**Decision:** Use Qdrant's SPLADE sparse vector support instead of running Elasticsearch locally. Elasticsearch documented as production horizontal scaling option.

**Why:** Elasticsearch uses ~1.2 GB of JVM memory locally on a machine with 18 GB unified memory shared with macOS, IDE, and Claude. Qdrant sparse vectors (SPLADE model) give equivalent BM25-style retrieval quality. This is actually a more modern approach than traditional BM25. The MCP server abstraction means swapping back to ES in production is one config change.

**What we rejected:** Keeping ES locally (too much RAM pressure), removing BM25 entirely (weakens hybrid retrieval story, context recall drops for exact-match financial terms like ticker symbols).

**Cost:** The "Elasticsearch" brand name is absent from locally running services. Mitigated by documenting it explicitly as a production swap and explaining why Qdrant sparse vectors are equivalent or better.

---

### D-09 — Kafka replaced by Redis Streams locally

**Decision:** Use Redis Streams for the ingestion event bus locally. Kafka/Redpanda documented as production swap.

**Why:** Redpanda (the lightest Kafka-compatible option) still uses ~600 MB of JVM memory. Redis Streams has identical consumer group semantics: persistent logs, consumer groups, per-consumer offsets, dead-letter patterns. The ingestion codebase uses a `StreamBackend` abstraction — `RedisStreamBackend` vs `KafkaBackend` — swappable via config. You learn the same concepts either way.

**What we rejected:** Keeping Redpanda locally (RAM pressure), removing event streaming entirely (loses the distributed ingestion story, backpressure patterns, consumer group isolation).

**Cost:** Must clearly communicate Redis Streams → Kafka equivalence in the README and in interviews. The concepts are identical; the brand name is different.

---

### D-10 — LLM strategy: Ollama locally, vLLM in production

**Decision:** Ollama with `llama3.2:3b` as dev default, `llama3.1:8b` for on-demand synthesis quality testing. vLLM documented as the production serving layer.

**Why:** 18 GB unified memory with macOS + IDE + all services running leaves ~5 GB for the LLM. llama3.2:3b fits in ~2 GB. llama3.1:8b requires ~6.5 GB which is fine on-demand (close Claude, close browser, run tests) but not as a persistent service. `OLLAMA_KEEP_ALIVE=0` unloads the model immediately after each response, reclaiming RAM.

**3b for plumbing, 8b for quality:** Phases 1–4 are about pipeline correctness (does ingestion work, do agents wire correctly, does retrieval return chunks). The model quality is irrelevant for these tests — a 3b model producing a mediocre answer still proves the plumbing works. Phase 5+ is when synthesis quality matters, and that's when you swap to 8b on-demand.

**What we rejected:** Always running 8b (too much RAM pressure for persistent service), Anthropic API as synthesis LLM (costs money, even if tiny), llama3.2:1b (too degraded for any meaningful quality testing).

**Cost:** Phase 5 synthesis testing requires closing other apps and swapping models. Acceptable tradeoff.

---

### D-11 — Graph RAG with Neo4j

**Decision:** Neo4j as the knowledge graph, heap-capped at 512 MB for local dev.

**Why:** Graph RAG is the biggest differentiator in this project. The supply chain traversal query (TSMC → Apple → competitors) is impossible with vector search alone. Neo4j's Cypher is the most expressive graph query language for this type of relationship traversal. Heap capping to 512 MB reduces local memory footprint from ~1 GB to ~500 MB with acceptable performance degradation for dev workloads.

**What we rejected:** ArangoDB (less mature Cypher equivalent), Amazon Neptune (managed, costs money), no graph store (loses the Graph RAG story entirely, which is a core differentiator).

**Cost:** Entity extraction during ingestion (NER pipeline), entity deduplication (Apple Inc. = Apple = AAPL), and Cypher query design. The hardest component to build correctly.

---

### D-12 — LangGraph for agent orchestration

**Decision:** LangGraph StateGraph as the orchestrator framework.

**Why:** LangGraph gives explicit, inspectable state machine control. Every node transition is defined, every conditional edge is code you write. This means you can reason about failure modes precisely — if Neo4j times out, the conditional edge routes to vector-only, not an undefined crash. This is what you want in a production system and what you can explain in an interview.

**What we rejected:** LangChain (higher level, less control), CrewAI (opinionated about agent roles, harder to customize failure handling), AutoGen (Microsoft-specific patterns, less production-credible in most interviews), raw asyncio (too much boilerplate, reinventing the wheel).

**Cost:** LangGraph has a learning curve. The StateGraph model requires upfront thinking about state shape and transitions. Pays off in debuggability.

---

### D-13 — RAGAS + LLM-as-judge for eval

**Decision:** RAGAS for offline and online eval. LLM-as-judge for faithfulness with structured unsupported claim logging.

**Why:** Context eval is the part of RAG systems most commonly done wrong or skipped. Having a golden dataset of 100 hand-curated (query, answer, source) triples that runs in CI and blocks merges on regression is what separates a production system from a demo. The LLM-as-judge pipeline logging specific unsupported claims to Postgres (not just a score) is what you show in interviews as evidence you take correctness seriously.

**What we rejected:** Manual testing only (doesn't scale, can't gate PRs), BLEU/ROUGE scores (wrong metric for RAG — measures n-gram overlap not factual grounding), no eval (the demo approach, not acceptable).

**Cost:** Building the golden dataset is labor-intensive (not a coding task — a curation task). Budget significant time in Phase 3 before tuning anything.

---

### D-14 — Observability: OTEL from Phase 1

**Decision:** OpenTelemetry instrumentation on every external service call, from the very first line of Phase 1 code. Not added later.

**Why:** You cannot debug what you didn't instrument. Adding observability after the fact means retrofitting every function, missing spans, and having incomplete traces for the bugs that matter most. Starting with OTEL from Phase 1 means every bug you hit during development has a trace. It also means the observability story in interviews is "instrumented from day one" not "added at the end."

**What we rejected:** Adding OTEL in Phase 6 (the common mistake), logging-only observability (insufficient for distributed tracing), no observability (unacceptable for a senior portfolio project).

**Cost:** ~5 extra lines per function that calls an external service. Negligible.

---

### D-15 — No feature store, no LLM fine-tuning in v1

**Decision:** No Feast/Tecton feature store. No LLM fine-tuning pipeline in v1. Embedding model fine-tuning documented as Phase 8 stretch goal.

**Why:** FinSight doesn't train models on features — it serves pre-trained models. Feature stores are for training pipelines, not RAG serving pipelines. Adding one would add operational overhead with zero benefit. LLM fine-tuning adds training infrastructure complexity (data pipeline, training loop, RLHF/DPO, champion/challenger serving) that is a separate project in itself.

**What we rejected:** Including Feast for completeness (adds complexity with no use case), fine-tuning the synthesis LLM (out of scope for v1, documented as future work).

**Cost:** The project doesn't demonstrate LLM fine-tuning. Acceptable — the project demonstrates RAG, Graph RAG, multi-agent orchestration, multi-tenancy, and production serving, which is more than enough.

---

### D-16 — No real-time price feeds

**Decision:** Market data ingested as daily ticker metadata only (company name, sector, SIC code). No real-time prices, no options data, no derivatives.

**Why:** FinSight is an analyst intelligence platform for querying documents — not a trading system. Price feeds add compliance complexity (data licensing), regulatory concerns (are we giving investment advice?), and infrastructure complexity (tick data requires different ingestion architecture) without improving document Q&A quality.

**What we rejected:** Including price feeds for completeness (scope creep with real legal/compliance implications for a finance-domain project).

**Cost:** Can't answer "what is Apple's stock price right now?" — but that was never a use case.

---

### D-17 — Docker Compose profiles for phased startup

**Decision:** Docker Compose profiles (`core`, `graph`, `observability`, `mcp`, `full`) so only the services needed for the current phase run simultaneously.

**Why:** Running all 17 services from Phase 1 creates unnecessary complexity and RAM pressure before the core loop is even working. Profile-based startup means Phase 1 runs 6 services, Phase 3 adds 2 more, Phase 5 adds the rest. Each phase has a clear, minimal footprint.

**What we rejected:** Single `docker compose up` that starts everything from day one (too much RAM, too many failure modes in Phase 1), manual service startup (error-prone, not reproducible).

**Cost:** Slightly more complex `docker compose up` commands. Worth it.

---

## Non-Negotiable Rules (Agreed Before Phase 1)

These are the rules we established before writing any code. They exist because violating them is how projects fail in production.

1. **Every component has a health check and a fallback.** If Neo4j is slow, the graph agent returns empty and logs the event. The orchestrator continues with vector-only. Degradation is designed, not accidental.

2. **Every agent communicates through typed Pydantic contracts.** No raw strings between agents. Type boundary violations are caught at the contract layer, not in the middle of synthesis.

3. **OTEL instrumentation is built in Phase 1, not Phase 6.** Every external service call gets a span from day one.

4. **All infra is Docker Compose locally, Kubernetes-compatible in structure.** No "it works on my machine."

5. **Tests before the component, not after.** Unit test with mock, integration test with real service, eval test for retrieval quality. The eval test is the one that matters.

6. **No hardcoded secrets, ever.** `.env` locally. If you commit a secret, rotate the whole secret.

---

## Open Questions (Resolved Before Starting Each Phase)

| Question | Resolution | Phase |
|---|---|---|
| Which SPLADE model for sparse vectors? | `naver/splade-cocondenser-selfdistil` — strong retrieval, reasonable size | Phase 3 |
| Golden dataset construction process? | Manual curation of 100 queries before retrieval tuning begins | Phase 3 |
| Langfuse: self-hosted or cloud free tier? | Cloud free tier (10k traces/month) — saves ~300 MB RAM | Phase 6 |
| Redis Streams consumer group retry logic? | Max 3 retries, then write to `dlq` stream key, alert on dlq depth > 10 | Phase 2 |

---

## What We Explicitly Did NOT Do (And Why)

| Skipped | Reason |
|---|---|
| GitHub Codespaces / cloud VM | Cost. Not willing to pay for infra before knowing job outcome. Local-first is the constraint. |
| Elasticsearch locally | ~1.2 GB JVM on 18 GB machine. Qdrant sparse vectors are equivalent. |
| Kafka/Redpanda locally | ~600 MB JVM. Redis Streams is semantically identical. |
| Vault locally | 50 MB saved, zero learning payoff vs dotenv for local dev. |
| LLM fine-tuning | Different project. Documented as Phase 8 stretch goal. |
| Real-time price feeds | Legal/compliance complexity, no use case benefit. |
| Feature store | No training pipeline, no benefit. |
| Social media data | Noise-to-signal too high for v1. |
| Always-on llama3.1:8b | 6.5 GB persistent RAM usage not viable. On-demand swap is the right pattern. |

---

### D-18 — Inference harness as a first-class component

**Decision:** Every LLM call passes through a named, explicit inference harness — input harness before the call, output harness after, eval harness async on sampled traffic. These live in `harness/` as a separate module, not scattered across agent code.

**Why:** "Prompt engineering" as a term and concept is already dated. What makes a production LLM system reliable isn't the prompt — it's everything around it. Input validation, output schema enforcement, faithfulness checking, hallucination flagging, PII scrubbing, citation extraction. Naming this explicitly as a harness and making it a first-class component means it's testable, versioned, and explainable.

**What we rejected:** Scattering validation logic across agent files (untestable, inconsistent), calling it "prompt engineering" (2 years behind), no output validation (acceptable for demos, not for a bank).

**Cost:** One additional module directory. Worth it for the interview story alone.

---

### D-19 — Entity resolution as a three-layer system

**Decision:** Entity resolution uses three layers: (1) canonical EDGAR company registry as ground truth, (2) fuzzy matching with confidence threshold during NER, (3) provisional entity queue for low-confidence matches.

**Why:** Financial documents refer to the same company in dozens of ways. A naive NER pipeline creates garbage in the knowledge graph. The three-layer approach is honest about uncertainty — low-confidence matches don't silently enter the graph, they go to a review queue. This is the difference between a graph that works and a graph that looks like it works.

**What we rejected:** Single-pass NER with no deduplication (graph garbage from day one), string equality matching only (too brittle for financial text), external entity resolution API (cost + latency + dependency).

**Cost:** `entity_resolution/` module with canonical loader, fuzzy matcher, and provisional queue. One Postgres table for provisional entities. Worth it — this is the thing that makes Graph RAG actually work.

---

### D-20 — Problem statement: enterprise document intelligence

**Decision:** FinSight is an enterprise document intelligence platform for financial and insurance institutions, not "a RAG system for SEC filings."

**Why:** The broader framing covers both the bank (State Street — ESG extraction, corporate actions, compliance Q&A) and the insurer (Plymouth Rock — policy Q&A, claims research, regulatory compliance). The technical depth is the same. The framing is more defensible to HR at a large institution and more honest about the actual use case.

**Reference companies:**
- State Street: ESG disclosure extraction, corporate actions processing, reference data entity resolution, compliance reporting
- Plymouth Rock: Policy Q&A for adjusters, regulatory filing compliance, claims document analysis

**Cost:** None. Better framing costs nothing.