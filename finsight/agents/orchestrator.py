"""
LangGraph orchestrator.

A designed state machine for one request: classify intent, extract
entities, run retrieval and the graph lookup in parallel, join, prepare
context in the input harness, synthesize, validate in the output harness,
and build the response. Every edge is explicit. Failure paths are part
of the graph rather than improvised at runtime.

The state object is the single source of truth. Nodes read from it and
write back to it. Components never call each other directly. errors uses
an additive reducer so both parallel branches can record failures
without overwriting each other.

Degradation is designed in. If Neo4j fails, the graph branch returns a
fallback result, the join still fires, and synthesis runs on vector
retrieval alone with the failure recorded in errors. If retrieval returns
nothing, the graph routes to an explicit no-context response instead of
calling the model.

The raw current query drives intent classification, entity extraction,
retrieval, caching and the faithfulness judge. Prior conversation turns,
validated and injection-scanned by the gateway, reach only the synthesis
node through conversation_context, so history can never affect the
ticker filter, the embedding or the faithfulness score.
"""

from __future__ import annotations

import asyncio
import logging
import operator
import re
import uuid
from typing import Annotated, TypedDict

from langgraph.graph import END, START, StateGraph

from finsight.agents.graph_agent import GraphAgent
from finsight.agents.retrieval_agent import RetrievalAgent
from finsight.agents.synthesis_agent import PROMPT_VERSION, SynthesisAgent
from finsight.harness.eval_harness import maybe_run_eval
from finsight.harness.input_harness import DEFAULT_RESERVED_TOKENS, run_input_harness
from finsight.harness.output_harness import run_output_harness
from finsight.models.base import AgentError
from finsight.models.graph import GraphResult
from finsight.models.retrieval import RetrievalResult
from finsight.models.synthesis import QueryResponse, SynthesisResult
from finsight.models.tenant import TenantConfig
from finsight.services.reranker import relevance_probability
from finsight.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

LOW_FAITHFULNESS_THRESHOLD = 0.85

_TICKER_PATTERN = re.compile(r"(?<![\w$])\$?([A-Z]{2,5})\b")
_NON_TICKERS = frozenset({
    "AI", "CEO", "CFO", "COO", "CTO", "EU", "ESG", "FY", "GAAP", "IPO",
    "SEC", "UK", "US", "USA", "USD", "ETF", "LLC", "INC", "API", "Q1", "Q2", "Q3", "Q4",
})


class AgentState(TypedDict):
    query: str
    team_id: str
    tenant_config: TenantConfig
    trace_id: str

    intent: str
    entities: list[str]
    conversation_context: str | None

    retrieval_result: RetrievalResult | None
    graph_result: GraphResult | None
    synthesis_result: SynthesisResult | None

    context_quality_score: float | None
    low_confidence: bool
    graph_used: bool
    prompt_version: str | None
    retry_count: int

    tokens_used: int
    errors: Annotated[list[AgentError], operator.add]
    cache_hit: bool
    final_response: QueryResponse | None


def _initial_state(
    query: str,
    tenant_config: TenantConfig,
    trace_id: str,
    conversation_context: str | None = None,
) -> AgentState:
    return AgentState(
        query=query,
        team_id=tenant_config.team_id,
        tenant_config=tenant_config,
        trace_id=trace_id,
        intent="factual",
        entities=[],
        conversation_context=conversation_context,
        retrieval_result=None,
        graph_result=None,
        synthesis_result=None,
        context_quality_score=None,
        low_confidence=False,
        graph_used=False,
        prompt_version=PROMPT_VERSION,
        retry_count=0,
        tokens_used=0,
        errors=[],
        cache_hit=False,
        final_response=None,
    )


class Orchestrator:
    """LangGraph state machine that coordinates the pipeline stages.

    Build once per application lifetime. The graph is compiled once at
    init and reused across requests, since compilation is expensive and
    invocation is cheap.
    """

    def __init__(
        self,
        retrieval_agent: RetrievalAgent,
        graph_agent: GraphAgent,
        synthesis_agent: SynthesisAgent,
    ) -> None:
        self._retrieval_agent = retrieval_agent
        self._graph_agent = graph_agent
        self._synthesis_agent = synthesis_agent
        self._background_tasks: set[asyncio.Task] = set()
        self._graph = self._build_graph()

    def _build_graph(self):
        builder = StateGraph(AgentState)

        builder.add_node("classify_intent", self._classify_intent)
        builder.add_node("extract_entities", self._extract_entities)
        builder.add_node("invoke_retrieval_agent", self._invoke_retrieval_agent)
        builder.add_node("invoke_graph_agent", self._invoke_graph_agent)
        builder.add_node("merge_results", self._merge_results)
        builder.add_node("run_input_harness", self._run_input_harness)
        builder.add_node("invoke_synthesis_agent", self._invoke_synthesis_agent)
        builder.add_node("run_output_harness", self._run_output_harness)
        builder.add_node("finalize", self._finalize)
        builder.add_node("return_error", self._return_error)

        builder.add_edge(START, "classify_intent")
        builder.add_edge("classify_intent", "extract_entities")
        builder.add_edge("extract_entities", "invoke_retrieval_agent")
        builder.add_edge("extract_entities", "invoke_graph_agent")
        builder.add_edge(["invoke_retrieval_agent", "invoke_graph_agent"], "merge_results")
        builder.add_conditional_edges(
            "merge_results",
            self._route_after_merge,
            {"run_input_harness": "run_input_harness", "return_error": "return_error"},
        )
        builder.add_edge("run_input_harness", "invoke_synthesis_agent")
        builder.add_conditional_edges(
            "invoke_synthesis_agent",
            self._route_after_synthesis,
            {
                "run_output_harness": "run_output_harness",
                "return_error": "return_error",
            },
        )
        builder.add_edge("run_output_harness", "finalize")
        builder.add_edge("finalize", END)
        builder.add_edge("return_error", END)

        return builder.compile()

    async def run(
        self,
        query: str,
        tenant_config: TenantConfig,
        conversation_context: str | None = None,
    ) -> QueryResponse:
        """Run the full request lifecycle and return a response.

        After the graph finishes, writes the retrieval cache and schedules
        the online eval as a background task so sampling and Langfuse
        logging never delay the response.

        Args:
            query: The raw current question. Drives intent, entities,
                   retrieval, the cache key and the faithfulness judge.
            tenant_config: Controls model tier, context window and scopes.
            conversation_context: The validated, injection-scanned history
                   prompt from the gateway, or None for a single-turn request.
                   Consumed only by the synthesis node.

        Returns:
            QueryResponse ready to return to the client.
        """
        trace_id = str(uuid.uuid4())

        with tracer.start_as_current_span("orchestrator.run") as span:
            span.set_attribute("team_id", tenant_config.team_id)
            span.set_attribute("trace_id", trace_id)

            state = _initial_state(query, tenant_config, trace_id, conversation_context)
            final_state = await self._graph.ainvoke(state)

            span.set_attribute("intent", final_state["intent"])
            span.set_attribute("graph_used", final_state["graph_used"])
            span.set_attribute("errors.count", len(final_state["errors"]))

            synthesis = final_state.get("synthesis_result")
            retrieval = final_state.get("retrieval_result")

            if synthesis and retrieval and retrieval.chunks and not retrieval.cache_hit:
                await self._retrieval_agent.write_cache(
                    query=query,
                    team_id=tenant_config.team_id,
                    chunks=retrieval.chunks,
                )

            if synthesis and retrieval:
                self._schedule_eval(query, synthesis, retrieval, trace_id, tenant_config.team_id)

            if final_state["final_response"]:
                return final_state["final_response"]

            return QueryResponse(
                trace_id=trace_id,
                answer="An error occurred processing your request.",
                citations=[],
                faithfulness_score=0.0,
                model_used="unknown",
                latency_ms=0.0,
                cache_hit=False,
                warning="Internal error",
            )

    async def run_state(
        self,
        query: str,
        tenant_config: TenantConfig,
        conversation_context: str | None = None,
    ) -> AgentState:
        """Run the graph and return the final state with no side effects.

        Used by the offline eval, which needs the exact chunks synthesis
        received. Skips the cache write and the online eval that run() does.
        """
        state = _initial_state(query, tenant_config, str(uuid.uuid4()), conversation_context)
        return await self._graph.ainvoke(state)

    def _schedule_eval(
        self,
        query: str,
        synthesis: SynthesisResult,
        retrieval: RetrievalResult,
        trace_id: str,
        team_id: str,
    ) -> None:
        """Start the online eval in the background and keep a reference until it finishes.

        The reference stops the task from being garbage collected mid-flight.
        """
        task = asyncio.create_task(
            maybe_run_eval(
                query=query,
                result=synthesis,
                chunks=retrieval.chunks,
                trace_id=trace_id,
                team_id=team_id,
            )
        )
        self._background_tasks.add(task)
        task.add_done_callback(self._background_tasks.discard)

    async def _classify_intent(self, state: AgentState) -> dict:
        """Classify the query intent by keyword: factual, analytical or graph.

        Recorded in state and on the trace for observability. Routing does
        not depend on it yet.
        """
        query = state["query"].lower()

        if any(w in query for w in ("compare", "change", "trend", "over time", "years")):
            intent = "analytical"
        elif any(w in query for w in ("relationship", "supply", "competitor", "connected")):
            intent = "graph"
        else:
            intent = "factual"

        return {"intent": intent}

    async def _extract_entities(self, state: AgentState) -> dict:
        """Extract ticker-like symbols from the raw query.

        Matches 2 to 5 uppercase letters, optionally prefixed with $, at any
        position including before punctuation or at the end of the query.
        Common uppercase abbreviations such as SEC or CEO are excluded.
        Runs on the current question only, so a ticker from an earlier turn
        cannot leak into this turn.
        """
        found = {m.group(1) for m in _TICKER_PATTERN.finditer(state["query"])}
        return {"entities": sorted(found - _NON_TICKERS)}

    async def _invoke_retrieval_agent(self, state: AgentState) -> dict:
        result = await self._retrieval_agent.retrieve(
            query=state["query"],
            tenant_config=state["tenant_config"],
            trace_id=state["trace_id"],
        )
        logger.debug("retrieval complete chunks=%d errors=%d", len(result.chunks), len(result.errors))
        return {
            "retrieval_result": result,
            "cache_hit": result.cache_hit,
            "errors": list(result.errors),
        }

    async def _invoke_graph_agent(self, state: AgentState) -> dict:
        """Look up the query entities in Neo4j, or skip when there are none.

        Runs in parallel with retrieval. A Neo4j failure comes back as a
        fallback result with its error recorded, so the join still fires
        and synthesis continues on vector retrieval alone.
        """
        if not state["entities"]:
            return {"graph_result": GraphResult(), "graph_used": False}

        result = await self._graph_agent.query(
            entities=state["entities"],
            tenant_config=state["tenant_config"],
            trace_id=state["trace_id"],
        )
        return {
            "graph_result": result,
            "graph_used": not result.fallback,
            "errors": list(result.errors),
        }

    async def _merge_results(self, state: AgentState) -> dict:
        """Join point for the parallel retrieval and graph branches.

        Records the best chunk's relevance probability as the context
        quality score. Graph context passes to synthesis unchanged.
        """
        retrieval = state["retrieval_result"]
        if not retrieval or not retrieval.chunks:
            return {}
        best = max(relevance_probability(c.score) for c in retrieval.chunks)
        return {"context_quality_score": best}

    def _route_after_merge(self, state: AgentState) -> str:
        retrieval = state["retrieval_result"]
        if not retrieval or not retrieval.chunks:
            return "return_error"
        return "run_input_harness"

    async def _run_input_harness(self, state: AgentState) -> dict:
        """Prepare context, reserving room for the model's output and prompt overhead."""
        retrieval = state["retrieval_result"]
        tenant_config = state["tenant_config"]

        harness_result = run_input_harness(
            chunks=retrieval.chunks,
            max_context_tokens=tenant_config.max_context_tokens,
            prompt_version=state["prompt_version"] or PROMPT_VERSION,
            reserved_tokens=DEFAULT_RESERVED_TOKENS + tenant_config.max_output_tokens,
        )
        updated_retrieval = RetrievalResult(
            chunks=harness_result.chunks,
            cache_hit=retrieval.cache_hit,
            retrieval_method=retrieval.retrieval_method,
            total_tokens=harness_result.tokens_in_context,
            latency_ms=retrieval.latency_ms,
            errors=retrieval.errors,
        )
        return {
            "retrieval_result": updated_retrieval,
            "prompt_version": harness_result.prompt_version,
            "low_confidence": harness_result.low_confidence,
        }

    async def _invoke_synthesis_agent(self, state: AgentState) -> dict:
        """Synthesize the answer.

        Uses conversation_context when present so the model sees prior turns,
        otherwise the raw query. This is the only node that reads history.
        """
        retrieval = state["retrieval_result"]
        synthesis_query = state["conversation_context"] or state["query"]
        result = await self._synthesis_agent.synthesize(
            query=synthesis_query,
            chunks=retrieval.chunks if retrieval else [],
            graph_result=state["graph_result"],
            tenant_config=state["tenant_config"],
            trace_id=state["trace_id"],
        )
        return {
            "synthesis_result": result,
            "tokens_used": result.tokens_used,
        }

    def _route_after_synthesis(self, state: AgentState) -> str:
        result = state["synthesis_result"]
        if not result or not result.answer:
            return "return_error"
        return "run_output_harness"

    async def _run_output_harness(self, state: AgentState) -> dict:
        """Validate and score the answer against the raw question.

        Faithfulness is judged against state["query"], never the wrapped
        history, so the score reflects whether the answer is grounded in
        the retrieved chunks for the question actually asked.
        """
        synthesis = state["synthesis_result"]
        retrieval = state["retrieval_result"]

        harness_result = await run_output_harness(
            result=synthesis,
            chunks=retrieval.chunks if retrieval else [],
            query=state["query"],
        )

        updated_synthesis = SynthesisResult(
            answer=harness_result.answer,
            citations=harness_result.citations,
            faithfulness_score=harness_result.faithfulness_score,
            unsupported_claims=harness_result.unsupported_claims,
            hallucination_flags=harness_result.hallucination_flags,
            tokens_used=synthesis.tokens_used,
            model_used=synthesis.model_used,
            prompt_version=synthesis.prompt_version,
            latency_ms=synthesis.latency_ms,
        )
        return {"synthesis_result": updated_synthesis}

    async def _finalize(self, state: AgentState) -> dict:
        """Build the response, warning on low retrieval confidence or low faithfulness."""
        synthesis = state["synthesis_result"]
        warnings: list[str] = []

        if state["low_confidence"]:
            warnings.append("Retrieved context had low relevance to the question.")

        if synthesis and synthesis.faithfulness_score < LOW_FAITHFULNESS_THRESHOLD:
            claims = [c for c in synthesis.unsupported_claims if c.strip()]
            if claims:
                warnings.append(f"Low confidence answer. Unsupported claims: {', '.join(claims[:3])}")
            else:
                warnings.append("Low confidence answer, please verify with source documents.")

        response = QueryResponse(
            trace_id=state["trace_id"],
            answer=synthesis.answer if synthesis else "",
            citations=synthesis.citations if synthesis else [],
            faithfulness_score=synthesis.faithfulness_score if synthesis else 0.0,
            model_used=synthesis.model_used if synthesis else "unknown",
            latency_ms=synthesis.latency_ms if synthesis else 0.0,
            cache_hit=state["cache_hit"],
            warning=" ".join(warnings) or None,
        )
        return {"final_response": response}

    async def _return_error(self, state: AgentState) -> dict:
        """Return an explicit failure response that names which stage failed."""
        retrieval = state.get("retrieval_result")

        if retrieval and retrieval.chunks:
            answer = "An answer could not be generated for this question. Please try again."
            warning = "Synthesis failed after context was retrieved."
        else:
            answer = "No relevant context found. Cannot generate a grounded answer."
            warning = "No context retrieved for this query."

        response = QueryResponse(
            trace_id=state["trace_id"],
            answer=answer,
            citations=[],
            faithfulness_score=0.0,
            model_used="none",
            latency_ms=0.0,
            cache_hit=state["cache_hit"],
            warning=warning,
        )
        return {"final_response": response}