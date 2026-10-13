# FinSight Interview Prep

> Private study doc. Not for the repo. Add `INTERVIEW.md` to `.gitignore`.

How to use this: read the question, try to answer out loud before reading the model answer, then compare. The answers are written to be spoken, not recited. Difficulty tags: `[warm-up]`, `[core]`, `[hard]`, `[curveball]`. The sections marked **Defend it** are the ones an interviewer will actually push on, because they are the honest soft spots in the project.

---

## 0. The opening pitch

You will be asked "tell me about this project" in the first two minutes. Have three lengths ready.

**10 seconds:** FinSight is a multi-agent platform that lets financial analysts query a corpus of SEC filings in plain English and get cited, auditable answers, with per-team access control and token budgets.

**30 seconds:** It is built around a LangGraph orchestrator that fans out to specialist retrieval and graph agents, fuses dense, sparse, and graph results, and runs every LLM call through an inference harness that scores faithfulness and flags hallucinated numbers. Three tenant teams each have their own data scope, token budget, and model tier, enforced at three independent layers. Full observability with OpenTelemetry, Prometheus, and Langfuse from day one.

**Two minutes:** The above, plus the why. Financial analysts make high-stakes decisions out of unstructured documents and spend hours manually reading filings. The hard part is not generating an answer, it is generating one you can trust and audit in a regulated setting. So the design decisions all serve trust: grounding (every claim maps to a chunk), governance (teams cannot see each other's data), and verifiability (a judge pipeline flags unsupported claims, logged as structured data). I made deliberate cost-aware substitutions for local development, Redis Streams for Kafka, Qdrant sparse vectors for Elasticsearch, Ollama for vLLM, each documented as a one-config production swap.

The one thing to land: **the orchestration is a designed state machine, not model-driven planning, because in compliance you want the failure modes you specified, not the ones a model improvises.**

---

## 1. LLM fundamentals

**[warm-up] What is a token?**
The unit an LLM actually processes. Not a word, a subword piece produced by a tokenizer (BPE or similar). "unbelievable" might be three tokens. Roughly 0.75 words per token in English. It matters because context windows, pricing, and latency are all measured in tokens, and your budget enforcement counts them. In FinSight I count tokens with tiktoken's cl100k_base as an approximation for the Llama models, close enough for budgeting even though the exact tokenizer differs.

**[warm-up] What is the context window and why does it matter?**
The maximum number of tokens a model can attend to in one call, prompt plus completion. Exceed it and you truncate or error. It matters here because each tenant has a different context budget (Analysis 64k, Ops 8k), and the input harness trims lowest-scoring chunks until the prompt fits. Bigger is not free: attention is roughly O(n²) in sequence length, so long contexts cost latency and memory.

**[core] What does temperature do?**
It scales the logits before the softmax. Lower temperature sharpens the distribution toward the highest-probability token, making output more deterministic; higher flattens it, making output more diverse and random. FinSight runs at 0.1 because in a factual, grounded-answer setting you want consistency and low creativity, not variety. Temperature 0 is near-greedy.

**[core] Top-k vs top-p (nucleus) sampling?**
Both restrict the candidate set before sampling. Top-k keeps the k highest-probability tokens. Top-p keeps the smallest set of tokens whose cumulative probability exceeds p, so the candidate set size adapts to how confident the model is. Top-p is usually preferred because it is adaptive. With temperature near 0 these matter less because the distribution is already peaked.

**[core] Why do LLMs hallucinate, and how do you reduce it?**
A language model predicts plausible next tokens; it has no built-in notion of truth, so when its parametric knowledge is thin or the prompt invites it, it produces fluent but ungrounded text. Reductions, in order of strength: ground it with retrieval so the answer comes from supplied context not memory; instruct it to answer only from context and say "not in the documents" otherwise; lower temperature; verify after generation. FinSight does all four, and the output harness adds a number-level check: any figure in the answer not appearing verbatim in the retrieved chunks gets flagged, because a fabricated revenue number is the highest-risk failure in finance.

**[hard] What is the KV cache and why does it matter for serving?**
During autoregressive generation, the attention keys and values for past tokens are cached so each new token does not recompute attention over the whole sequence. It is what makes generation roughly linear per token instead of quadratic. It matters for serving because the KV cache consumes GPU memory proportional to context length times batch size, and it is the main thing vLLM optimizes with PagedAttention, treating the cache like virtual memory pages to pack more concurrent requests. This is exactly why FinSight's production serving layer is vLLM and not Ollama: throughput under concurrency.

**[hard] Greedy vs beam search vs sampling for decoding?**
Greedy takes the argmax each step, fast but can be locally myopic. Beam search keeps the top-b partial sequences, better for tasks with a single correct output like translation, but it is slower and tends to produce bland text for open generation. Sampling (with temperature and top-p) is standard for chat. For grounded factual answering you lean toward low-temperature near-greedy, which is what FinSight does.

**[curveball] cl100k_base is the GPT-4 tokenizer but you serve Llama. Is your token counting wrong?**
Slightly, yes, and I know it. Different tokenizers segment text differently, so my counts are approximate. For budget enforcement that is acceptable: I am gating spend, not billing a customer to the token, and a few percent error does not change whether someone blows a 2M daily budget. If this were a paid metering system I would use the actual model tokenizer. It is a deliberate approximation, documented, not an oversight.

---

## 2. RAG

**[warm-up] What is RAG and why use it instead of just a bigger model or fine-tuning?**
Retrieval-augmented generation: fetch relevant documents at query time and put them in the prompt so the model answers from them. You use it over fine-tuning when knowledge changes often (new filings every quarter), when you need citations and auditability, and when you cannot put the whole corpus in context. Fine-tuning bakes knowledge into weights, is expensive to update, and gives you no source attribution. RAG keeps the knowledge external, fresh, and traceable, which is the entire point in a compliance setting.

**[core] Walk me through your chunking strategy.**
Phase 1 was paragraph-aware splitting at a 400-token target with 50-token overlap. The real version is document-type-specific: SEC filings split at Item boundaries (Item 1, 1A, 7, 8) with tables kept as single chunks never split mid-table, transcripts split by speaker turn with the speaker prepended, news by paragraph with headline prepended. The reasoning: a generic recursive splitter breaks a risk-factors section mid-paragraph and the chunk loses the regulatory context that made it meaningful. Overlap preserves context at boundaries so a sentence spanning two chunks is not orphaned.

**[core] Why overlap between chunks? What is the tradeoff?**
Overlap means a boundary-spanning sentence appears in both neighboring chunks, so neither loses it. The tradeoff is storage and redundancy: overlapping content is embedded and stored twice, and you can retrieve near-duplicate chunks. You tune overlap against chunk size; too much overlap wastes the context budget on repetition.

**[core] Dense vs sparse retrieval. Why do you use both?**
Dense embeddings capture semantic similarity: "supply chain concentration risk" matches a paragraph about TSMC even if TSMC is never named. Sparse retrieval (SPLADE here, BM25-style) captures exact lexical matches: ticker symbols, CIK numbers, specific clause references that dense vectors can blur. Financial text needs both, you want the semantic recall and the exact-term precision. Using only dense, you miss exact ticker matches; using only sparse, you miss paraphrased concepts.

**[hard] What is SPLADE and how is it different from BM25?**
SPLADE is a learned sparse retrieval model. It runs text through a masked-language-model head and produces a sparse vector over the vocabulary, where each dimension is a term weight, including terms that did not literally appear (learned term expansion). BM25 is a fixed statistical scoring function over exact term frequency and inverse document frequency, no learning, no expansion. SPLADE gives you BM25-like exact-match behavior plus some learned semantic expansion, while staying sparse and efficient, and Qdrant supports sparse vectors natively so I get it without running Elasticsearch.

**[core] What is Reciprocal Rank Fusion and why use it over score-based fusion?**
RRF merges multiple ranked lists by summing 1/(k + rank) for each item across lists, with k around 60. It uses ranks, not raw scores. That is the key advantage: dense cosine scores and sparse term-weight scores live on completely different scales, so naively adding them is meaningless and normalizing them is fragile and needs constant recalibration. RRF sidesteps that entirely by only caring about position. An item ranked highly by multiple methods rises to the top. Simple, robust, no tuning.

**[core] What is a cross-encoder reranker and why not just use it for everything?**
A cross-encoder takes the query and a document together in one forward pass and outputs a relevance score, so it can model fine-grained interaction between them. That makes it much more accurate than the bi-encoder embeddings used for first-stage retrieval, which encode query and document separately and only compare vectors. You cannot use it for everything because it cannot precompute: every query-document pair is a fresh forward pass, so scoring the whole corpus per query is infeasible. The standard pattern, which FinSight uses, is retrieve cheaply to a top-50 with bi-encoders plus sparse, then rerank those 50 with the cross-encoder.

**[hard] Lost in the middle. What is it and what did you do about it?**
LLMs attend most strongly to the beginning and end of their context and can neglect information buried in the middle. So if you stuff your best chunk in position 25 of 50, the model may underweight it. The input harness reorders chunks best-first and second-best-last, pushing the most relevant content to the two positions the model attends to most. It is a cheap, deterministic mitigation grounded in the attention-position research.

**[core] How do you stop the system from answering when it has no good context?**
Two gates. The orchestrator checks the retrieval result before calling the LLM: empty chunk list routes straight to an error response, no LLM call. The input harness checks the top retrieval score against a minimum threshold (0.5) and flags low confidence. If there is nothing to ground on, the system returns "no relevant context found" rather than letting the model freelance from parametric memory. Not calling the LLM is the correct behavior, not a failure.

**Defend it: [hard] Your context recall is 0.677. That means a third of needed information is missing. Is this system actually usable?**
At three companies and with the standalone eval not using graph context, yes those numbers are modest, and I report them honestly rather than rounding up. Two things. First, the eval pipeline is a deliberately stripped path: it runs dense retrieval only, so it understates the production orchestrator which adds sparse, graph, and reranking. Second, recall is the metric I would invest in next: better chunking on the underrepresented Item 1 sections, and wiring graph context into the eval so multi-filing queries stop being penalized. The point of the eval gate is not that the numbers are perfect, it is that they are measured, thresholded, and block regressions in CI. A system with honest 0.68 recall and a gate is more trustworthy than one claiming 0.95 with no methodology.

---

## 3. Querying and Graph RAG

**[core] Why a graph database at all? What can it do that vector search cannot?**
Vector search finds text that is semantically similar. It structurally cannot answer relationship questions, because a relationship is not a similarity. "TSMC supplies Apple, Apple competes with Samsung, which competitors also depend on TSMC" is a multi-hop traversal over typed edges. There is no embedding for "two hops away via a SUPPLIES_TO then COMPETITOR_OF edge." Neo4j with Cypher expresses that directly. The canonical FinSight demo query needs graph traversal plus semantic search plus exact ticker matching, which is why all three modalities exist.

**[core] Give me a Cypher query for "who supplies Apple."**
Something like:
```cypher
MATCH (s:Company)-[:SUPPLIES_TO]->(a:Company {ticker: 'AAPL'})
WHERE 'public' IN a.scopes OR $team_id IN a.scopes
RETURN s.name, s.cik
```
Note the scope predicate. Tenant isolation is enforced in the query itself, not after, so a team never even retrieves nodes outside its scope.

**[hard] How do you build the graph from raw filings? What is the hard part?**
Named-entity extraction during ingestion produces candidate entities, then they have to be resolved to canonical companies. That is the hard part. Financial text refers to one company dozens of ways: "Apple," "Apple Inc.," "AAPL," "the Company." Naive NER puts all of those in as separate nodes and the graph becomes garbage. FinSight uses three layers: a canonical EDGAR registry as ground truth (CIK is the primary key, not the name), fuzzy matching with a confidence threshold (0.85 merges to the canonical node), and a provisional-entity queue where low-confidence matches go for review instead of silently entering the graph. The principle: be honest about uncertainty rather than polluting the graph.

**[core] Why CIK as the entity key instead of name or ticker?**
CIK is the SEC Central Index Key, the canonical, stable, unique identifier for a registered filer. Names are ambiguous and change; tickers can be reused or change; a company can have multiple tickers. CIK does not. Keying on CIK is what prevents two spellings of the same company from becoming two nodes.

**Defend it: [hard] Your graph has 4 company nodes and 11 relationships. That is tiny. Is the Graph RAG real or decorative?**
The graph is small because the corpus is three companies; the machinery is real. The ingestion path does NER, three-layer resolution against the full EDGAR registry, and MERGEs nodes and typed edges. The graph agent runs real Cypher with scope filtering and returns to the orchestrator, with a fallback to vector-only if Neo4j is down. What is honestly underdeveloped is edge richness, with three companies there are few real relationships to traverse, and the standalone eval does not exercise graph context. At S&P 500 scale the same code produces a dense, useful graph. So: real implementation, small data. I would not oversell the current graph's analytical power.

---

## 4. Multi-agent orchestration

**[core] What is an agent, and what makes your system multi-agent?**
A raw LLM generates text. An agent is a model plus the infrastructure that lets it act: tool use, state, a loop, error handling. FinSight is multi-agent because work is split across specialists, an orchestrator that owns the request lifecycle, a retrieval agent that owns the hybrid search and rerank pipeline, a graph agent that owns Neo4j, and a synthesis agent that owns the LLM call. Each has a single responsibility and a typed contract. They never talk to each other directly; they read and write a shared state object the orchestrator controls.

**[core] Why LangGraph over LangChain, CrewAI, or AutoGen?**
LangGraph models the workflow as an explicit state graph: nodes mutate shared state, conditional edges define routing, and every transition is code I wrote. That gives me debuggable, designed failure modes. If Neo4j times out, a conditional edge routes to vector-only retrieval, not an undefined crash. LangChain's older AgentExecutor was a higher-level ReAct loop with less control. CrewAI and AutoGen are opinionated about agent roles and conversation patterns, which makes custom failure handling harder and is less production-credible for a compliance system. The theme is control: I want to specify what happens when things break.

**[hard] Designed state machine vs model-driven planning (ReAct). When is each right?**
A ReAct-style agent lets the model decide its own next action in a loop, flexible, good for open-ended tasks where you cannot enumerate the steps. A designed state machine fixes the control flow up front. For an open research assistant, model-driven planning is reasonable. For a regulated financial Q&A pipeline, I deliberately chose the state machine, because I want the failure modes I specified, not ones the model improvises, and I want every run to be inspectable and reproducible. The tradeoff is flexibility: my pipeline does the same shape of work every time. That is a feature here, not a limitation.

**[core] How does state flow through your orchestrator?**
There is a single `AgentState` TypedDict that is the source of truth. The graph runs: classify intent, extract entities, invoke retrieval and graph agents, merge, run input harness, synthesize, run output harness, finalize. Each node returns a partial update merged into state. Conditional edges check state, for example, after merge, if there are no chunks, route to the error node and never call the LLM. Agents communicate only through this state, never directly, so there are no hidden side channels.

**[hard] Walk me through your graceful degradation matrix.**
Every dependency has a defined fallback. Neo4j down: graph agent returns fallback=true, orchestrator continues with vector and sparse only. Qdrant vector search fails: fall back to sparse-only. Reranker fails: use raw RRF scores. Redis cache down: bypass cache, go straight to the LLM. LLM 8b out of memory: route to 3b and log the downgrade. The principle from day one: every component has a health check and a fallback, degradation is designed, not accidental. Circuit breakers wrap the MCP calls, opening after five consecutive failures with a 30-second recovery probe, so a dead dependency fails fast instead of hanging every request.

**[core] You recently fixed conversation history handling. What was the bug and the fix?**
The gateway built a history-wrapped prompt but then passed the raw query to the orchestrator, so follow-up context was computed and thrown away. The naive fix is to pass the wrapped blob everywhere, but that creates two problems: intent classification and entity extraction would run over old turns, leaking a stale ticker into the current retrieval filter, and client-supplied history would bypass the prompt-injection guardrail that only scanned the current query. So the clean fix separates concerns: the raw query drives retrieval, entities, audit, and the faithfulness judge; the history is length-bounded, injection-scanned at the gateway, and passed as a synthesis-only context. History can never pollute retrieval or scoring, and a malicious history turn fails the request closed.

---

## 5. The harness

**[core] What do you mean by "inference harness," and why call it that?**
The harness is everything wrapping the LLM call that makes it production-safe, the validation, scoring, and safety layers around generation. I name it explicitly because "prompt engineering" frames the prompt as the thing that makes an LLM reliable, and it is not. Reliability comes from the input validation, output schema enforcement, faithfulness checking, hallucination flagging, and PII scrubbing around the call. Making it a named, first-class module means it is testable and versioned rather than scattered across agent code.

**[hard] How does this relate to the broader "agent harness" idea people talk about with Codex?**
Two senses of the word. The broad one, popularized recently around Codex and Claude Code, is the whole infrastructure that turns a model into an agent: model interface, tool registry, context management, orchestration loop, memory, sandbox, serving layer. By that definition my harness is basically the whole `finsight/` package. My `harness/` directory is the narrower, classic sense: the inference-time control layer immediately around each LLM call. In the agent-harness framing that maps to the control layer, computational checks like my regex hallucination and PII scan, plus inferential checks like the LLM-as-judge faithfulness scorer. What FinSight is not is an autonomous coding agent: there is no sandbox executing model-generated code in a self-directed loop. It is a governed retrieval-and-synthesis harness, which is the honest and correct description.

**[core] How does your faithfulness check work, and what is the failure mode of using an LLM to judge an LLM?**
The output harness sends the context, question, and answer to the model with a structured judge prompt asking for a score and a list of specific unsupported claims, then parses the response. The risk of LLM-as-judge is obvious: the judge can be wrong, biased toward fluent answers, or inconsistent. Mitigations: I log the specific unsupported claims as structured data, not just a score, so failures are inspectable; I pair it with a deterministic computational check (numbers not in context get flagged regardless of what the judge says); and on a judge parse failure I default to not penalizing the answer rather than silently scoring it zero. At scale you would also calibrate the judge against human labels.

**[core] What does the input harness actually do before the LLM call?**
Five things: context-quality check (top score above threshold, else flag low confidence), token-budget trim (drop lowest-scoring chunks until the prompt fits the tenant's window), position-bias reorder (best chunk first, second-best last), prompt-injection scan over the retrieved chunks (filings are public documents an adversary could craft), and prompt-version stamping so every trace records which template ran. All deterministic, all cheap, all before any token is spent.

---

## 6. MCP

**[core] What is MCP and why did you use it?**
Model Context Protocol is a standard for exposing tools, resources, and prompts to an LLM application over a uniform interface, so tools become discoverable, swappable services rather than hardcoded functions. In FinSight each tool, vector search, sparse search, graph query, generation, rerank, is an independently deployable MCP server. Agents discover tools at runtime via a registry. The payoff: adding a new data source like Bloomberg means standing up a new MCP server and updating tenant scope config, with no agent code change. It turns the system into a platform rather than a monolith, and it demonstrates protocol-level thinking.

**[hard] What is the cost of the MCP abstraction? Was it worth it here?**
The cost is real: five FastAPI services to write and maintain, each with a manifest endpoint, invoke endpoint, health check, and independent JWT validation. For a single-machine project that is overhead you do not strictly need; you could call the functions directly. Was it worth it? For the learning and the interview story, yes, it demonstrates the platform pattern and defense-in-depth, since each server validates the token independently on top of the gateway. If I were optimizing purely for shipping the demo fastest, in-process tool calls would have been simpler. I made the call deliberately, knowing the tradeoff.

**[curveball] If each MCP server validates the JWT independently, isn't that redundant with the gateway check?**
That redundancy is the point, it is defense in depth. The gateway check is the first line, but if an attacker reached an MCP server directly, or a bug let an unauthorized request past the gateway, the server-level check still stops it. Independent validation at every layer means a single bypassed check does not leak data. In a bank this is exactly the posture you want, no single point of trust.

---

## 7. Multi-tenancy, auth, governance

**[core] Explain your three-layer data isolation.**
Three independent layers, all of which must be bypassed simultaneously for a leak. One, OAuth scope at the gateway: a token without `read:all_filings` cannot request that data. Two, Qdrant payload filter: every vector query carries a mandatory filter that the chunk's scopes include the team or "public," enforced at the index level so a caller cannot forget it. Three, Neo4j Cypher: every graph query has a scope predicate in the WHERE clause. Isolation is enforced at the data layer, not just the application layer. That is the difference between "we check permissions" and "the data is unreachable."

**[core] Why OAuth2 client credentials and not API keys or user login?**
Client credentials is the correct machine-to-machine pattern: services authenticate with a client id and secret and receive a JWT carrying scopes. API keys cannot natively carry structured, layered scopes. User login is the wrong model, there is no human in a service-to-service call. The scopes (`read:all_filings`, `query:graph`, `model:large`) flow through every layer in the JWT, which is what enables the three-layer isolation.

**[hard] What is in your JWT and how is it validated? What would you change for production?**
Subject, team id, scopes, issued-at, expiry, signed with HS256. The gateway and every MCP server decode and verify the signature, check expiry, and enforce required scopes. For production I would change three things, all of which I know are dev shortcuts: HS256 with a shared secret becomes RS256 with asymmetric keys so services verify with a public key and cannot mint tokens; the secret moves from code into a secrets manager like Vault; and I would add token revocation, since stateless JWTs cannot be revoked before expiry without a denylist.

**[core] How does token budgeting work?**
Each team has a daily token budget. Before a request, the broker checks a Redis counter keyed by team and date; if the estimate would exceed the budget, the request is rejected with a retry-after until midnight. Usage is metered into the same counter. The counter has a TTL so budgets reset automatically. Priority queuing uses three Redis sorted sets with weighted fair dequeue (8 from P1, 2 from P2, 1 from P3) so the high-priority Analysis team is served first but Ops is never starved.

---

## 8. Observability and evaluation

**[core] What do you trace, and why from day one?**
OpenTelemetry spans on every external service call, Qdrant, Neo4j, the LLM, embeddings, from the first phase. The reason is simple: you cannot debug what you did not instrument. Adding tracing later means retrofitting every function and missing spans for exactly the hard bugs. Spans carry attributes like token counts, latencies, and result counts, so every bug I hit during development already had a trace. Prometheus carries the aggregate metrics (latency percentiles, token usage, cache hit rate, faithfulness gauge, circuit breaker opens), Langfuse carries the RAG-specific traces with chunk scores and prompt versions.

**[core] Explain the four RAG eval metrics you use.**
Context precision: of the chunks retrieved, what fraction were actually relevant, a ranking-quality measure. Context recall: of the information needed to answer, what fraction was retrieved. Faithfulness: of the claims in the answer, what fraction are grounded in the context, the anti-hallucination metric, judged by LLM. Answer relevancy: does the answer actually address the question asked. Precision and recall measure retrieval; faithfulness and relevancy measure generation. Together they cover both halves of RAG.

**[hard] You abandoned the RAGAS library and implemented the metrics yourself. Why, and is that a red flag?**
RAGAS had intractable dependency conflicts with my langchain-core and LangGraph versions, broken imports across multiple RAGAS versions, the kind of dependency hell that does not hold even with surgical pins. Rather than fight it, I implemented the four metrics directly as structured LLM-judge prompts with regex score parsing. Not a red flag, the opposite: the metrics are well-defined, I understand exactly what each computes because I wrote it, and I removed a fragile dependency from my CI gate. Reimplementing a clearly-specified metric to escape dependency hell is a sound engineering call. What I would add at scale is calibration of my judge against human labels.

**[core] How does the eval gate work in CI?**
A 15-query golden dataset runs offline, computes the metrics, and fails the build if any metric drops below its threshold in `thresholds.yaml`. It runs on every PR touching retrieval or agent code and blocks merge on regression. That is what separates a production system from a demo: retrieval quality is a tested, gated property, not a vibe.

**Defend it: [hard] Your golden dataset is 15 queries. Real eval sets are thousands. Is 15 meaningful?**
Fifteen hand-curated queries across three tenant types is enough to catch regressions, which is the gate's job, but not enough to make strong absolute quality claims, and I would not pretend otherwise. The honest framing: it is a smoke-test gate, not a statistical benchmark. Curating a golden set is labor, not code, so I built a small high-quality one rather than a large noisy one. Scaling it is straightforward and is the obvious next investment; the harness already supports it.

---

## 9. Systems design and production

**[core] Walk me through the full request lifecycle.**
Client sends a query with a JWT. Gateway validates the token, loads tenant config, runs the query guardrail (injection and investment-advice rejection), sanitizes any conversation history. Orchestrator classifies intent, extracts entities, fans out to retrieval and graph agents in parallel. Retrieval does cache lookup, then hybrid dense-plus-sparse search with scope filtering, RRF fusion, cross-encoder rerank. Results merge; if empty, return error without calling the LLM. Input harness validates and prepares context. Synthesis agent calls the LLM. Output harness scores faithfulness, flags hallucinated numbers, scrubs PII. Tokens metered to Redis, cache written, trace logged to Postgres and Langfuse, cited response returned.

**[core] How does your caching work and what does it save?**
A semantic cache in Redis keyed by team and a hash of the query, with a TTL (shorter for news, longer for filings). A cache hit returns the stored result and charges zero tokens. At steady state it eliminates a large fraction of repeated LLM calls, which is the single biggest cost lever in a RAG system, generation dominates cost and latency. Cache is scoped per team so isolation holds even in the cache.

**[hard] Your demo answers take 240 seconds. How is this production-ready?**
It is not, locally, and the architecture is the reason it does not need to be. That latency is a 3-billion-parameter model running on an 18GB unified-memory Mac with every service resident and competing for RAM, plus model load time because I unload between calls to reclaim memory. None of that is the system design; it is the hardware constraint. The LLM interface is OpenAI-compatible, so production swaps one environment variable to point at vLLM on a dedicated GPU, where the same 8b model serves in low single-digit seconds with batching and a persistent KV cache. The SLOs I designed for are P95 under 8 seconds for Analysis and under 2 for Ops. The whole model-agnostic serving layer exists precisely so the local constraint never touches the design.

**[hard] How would you scale this from 3 companies to the S&P 500?**
Several axes. Ingestion: the Redis Streams pipeline becomes Kafka for real partitioning and throughput, with consumer groups for embedding, Qdrant writing, and Neo4j writing scaling independently. Vector store: Qdrant sharded, with the sparse side possibly moving back to a dedicated Elasticsearch as documented. Serving: vLLM on GPUs behind the model interface, horizontal gateway replicas behind a load balancer. The graph gets genuinely useful at that scale. Eval set grows from 15 to thousands. The substitutions I made for local dev are all documented as exactly these production swaps, which is the point of the abstraction layers.

**[curveball] What breaks first under 100x load?**
Honestly, the LLM serving tier, generation is the bottleneck in any RAG system. Retrieval scales horizontally cheaply; generation needs GPUs and batching. After that, the synchronous parts of ingestion and any unbatched embedding calls. The caching layer is what buys headroom, so I would watch cache hit rate as the leading indicator and the LLM queue depth as the lagging one.

**Defend it: [hard] This all runs on one laptop. Have you actually built a distributed system or just drawn one?**
Fair challenge. What is real and running: the full multi-agent pipeline, three-layer auth, hybrid retrieval, the harnesses, observability, the eval gate, real EDGAR data. What is structurally distributed-ready but not actually distributed: it runs in Docker Compose on one machine, not Kubernetes across nodes. I was deliberate about the constraint, local-first, no cloud spend before knowing the job outcome, and I made the substitutions that preserve the concepts (Redis Streams has Kafka's consumer-group semantics; the code uses a swap interface). So: I built the architecture and the concepts correctly, on a single machine, with documented paths to true horizontal scale. I would rather say that plainly than claim I ran a 50-node cluster I did not.

---

## 10. The substitution questions (you will get at least one)

**[core] Redis Streams instead of Kafka. Justify it.**
Redpanda, the lightest Kafka-compatible option, still uses around 600MB of JVM memory I did not have on an 18GB machine. Redis Streams has the semantics that matter: persistent append-only logs, consumer groups, per-consumer offsets, dead-letter patterns. The ingestion code uses a `StreamBackend` abstraction so swapping to Kafka in production is a config change, not a rewrite. You learn the same concepts, consumer-group isolation, offset tracking, backpressure, DLQ, either way. The brand differs; the distributed-log concept does not.

**[core] Qdrant sparse vectors instead of Elasticsearch. Justify it.**
Elasticsearch is around 1.2GB of JVM locally. Qdrant supports SPLADE sparse vectors natively, giving me BM25-style exact-match retrieval in the same store as my dense vectors, with one fewer service to run. It is arguably more modern than classic BM25. Documented as a production swap if a team already runs an ES cluster.

**[core] Ollama instead of vLLM. Justify it.**
Ollama is an OpenAI-compatible local runtime, perfect for development on constrained hardware, one command to pull and run a model. vLLM is the production serving layer for throughput, PagedAttention, continuous batching, high concurrency. Because both speak the OpenAI API, the swap is one environment variable. Dev convenience locally, production throughput in prod, zero application code change.

---

## 11. Behavioral, tied to real things you did

**[core] Tell me about a hard bug you debugged.**
The conversation-history bug is a good one: a feature that looked done but silently did nothing, the wrapped context was computed and discarded. The interesting part was not the one-line fix, it was recognizing that the obvious fix introduced two security holes, retrieval pollution from stale entities and an injection bypass of the guardrail. So the real fix was an architectural separation: raw query for retrieval and scoring, sanitized history for synthesis only. The lesson: the naive fix and the correct fix are often different, and security-relevant code rewards thinking past the first solution.

**[core] Tell me about a time you chose not to use a popular tool.**
RAGAS. I needed real RAG eval metrics, tried to install the library, and hit cascading dependency conflicts with my langchain-core and LangGraph versions across multiple RAGAS releases. Rather than pin-and-pray, I judged the metrics were well-specified enough to implement directly as LLM-judge prompts. That removed a fragile dependency from my CI gate and meant I fully understood my own metrics. Knowing when to stop fighting a dependency is a skill.

**[core] What would you do differently if you started over?**
Wire graph context into the eval pipeline from the start, so analytical multi-filing queries are not understated. Ingest a broader corpus earlier so the graph is meaningful sooner. And I would set up the GPU serving path earlier even if just rented briefly, so I had real latency numbers instead of having to explain the local constraint. The architecture I would keep, the designed-state-machine and three-layer-isolation decisions have held up.

**[curveball] What is the weakest part of this project?**
The eval surface: 15 queries, 3 companies, graph context not in the eval path, so the reported metrics understate the design and are too small for strong absolute claims. I am upfront about that because the gate's value is regression-catching, and the fix is curation and corpus expansion, not redesign. I would rather name the real weak spot than have you find it.

---

## 12. Rapid-fire fundamentals (know these cold)

- **Embedding dimension here:** 768 (nomic-embed-text), upgradeable to 1024 (bge-m3).
- **Why cosine similarity:** for normalized vectors cosine equals dot product; it measures direction not magnitude, which is what semantic similarity needs.
- **Bi-encoder vs cross-encoder:** bi encodes separately and is precomputable and fast; cross encodes jointly and is accurate but per-pair. Retrieve with bi, rerank with cross.
- **Why not fine-tune the LLM:** knowledge changes quarterly, you lose citations, training infra is a separate project. RAG keeps knowledge external and auditable.
- **What a vector index does:** approximate nearest neighbor (HNSW in Qdrant) to avoid scanning every vector; trades a little recall for large speed gains.
- **Idempotent ingestion:** deterministic chunk IDs via SHA256 of doc id plus index, so re-running upserts overwrite instead of duplicating.
- **Why `query_points` not `search` in Qdrant:** the older `search` API is deprecated; `query_points` is current.
- **Faithfulness threshold:** 0.85; below it the answer gets a warning and the unsupported claims are logged to Postgres.
- **Circuit breaker:** opens after 5 consecutive failures, 30s recovery, single half-open probe.
- **RRF constant k:** 60, standard, dampens the influence of very top ranks so lower ranks still contribute.

---

## 13. Questions to ask them (have 3 ready)

- How do you handle eval and regression testing for your LLM systems today, and where does it hurt?
- Where on the spectrum from designed pipelines to autonomous agents do your production systems sit, and why?
- What does your model serving and observability stack look like in production?

---

*Study the Defend-it sections hardest. Those are where a sharp interviewer goes, and a calm, honest answer about a real tradeoff beats a polished claim every time.*