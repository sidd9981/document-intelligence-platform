"""
LangGraph orchestrator.

Owns the full request lifecycle: intent classification, entity extraction,
parallel agent fan-out, harness execution, token metering, caching, and
audit logging. Every request flows through this state machine.

The state object is the single source of truth. Nodes read from it and
write back to it. Agents never communicate directly with each other.

The raw current query drives intent classification, entity extraction,
retrieval, caching, audit, and the faithfulness judge. When a request
carries prior conversation turns, the gateway has already validated and
injection-scanned them and hands us a separate conversation_context. That
context is used only by the synthesis agent, so prior turns can never
pollute the ticker filter, the embedding, or the faithfulness score.
"""

from __future__ import annotations

import logging
import re
import uuid
from typing import TypedDict

from langgraph.graph import END, START, StateGraph

from finsight.agents.graph_agent import GraphAgent
from finsight.agents.retrieval_agent import RetrievalAgent
from finsight.agents.synthesis_agent import SynthesisAgent
from finsight.harness.eval_harness import maybe_run_eval
from finsight.harness.input_harness import run_input_harness
from finsight.harness.output_harness import run_output_harness
from finsight.models.base import AgentError
from finsight.models.graph import GraphResult
from finsight.models.retrieval import RetrievalResult
from finsight.models.synthesis import QueryResponse, SynthesisResult
from finsight.models.tenant import TenantConfig
from finsight.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

PROMPT_VERSION = "synthesis_v1"


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
    prompt_version: str | None
    retry_count: int

    tokens_used: int
    errors: list[AgentError]
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
        prompt_version=PROMPT_VERSION,
        retry_count=0,
        tokens_used=0,
        errors=[],
        cache_hit=False,
        final_response=None,
    )


class Orchestrator:
    """LangGraph state machine that coordinates all agents.

    Build once per application lifetime. The graph is compiled once
    at init and reused across requests — compilation is expensive,
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
        builder.add_edge("invoke_retrieval_agent", "invoke_graph_agent")
        builder.add_edge("invoke_graph_agent", "merge_results")
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

        Args:
            query: The raw current question. Drives intent classification,
                   entity extraction, retrieval, the cache key, audit
                   logging, and the faithfulness judge.
            tenant_config: Controls model tier, context window, and scopes.
            conversation_context: The history-wrapped prompt the gateway
                   has already validated and injection-scanned, or None for
                   a single-turn request. Consumed only by the synthesis
                   agent so prior turns never affect retrieval or scoring.

        Returns:
            QueryResponse ready to return to the client.
        """
        trace_id = str(uuid.uuid4())

        with tracer.start_as_current_span("orchestrator.run") as span:
            span.set_attribute("team_id", tenant_config.team_id)
            span.set_attribute("trace_id", trace_id)

            state = _initial_state(query, tenant_config, trace_id, conversation_context)
            final_state = await self._graph.ainvoke(state)

            synthesis = final_state.get("synthesis_result")
            retrieval = final_state.get("retrieval_result")

            if synthesis and retrieval and retrieval.chunks and not retrieval.cache_hit:
                await self._retrieval_agent.write_cache(
                    query=query,
                    team_id=tenant_config.team_id,
                    chunks=retrieval.chunks,
                )

            if synthesis and retrieval:
                await maybe_run_eval(
                    query=query,
                    result=synthesis,
                    chunks=retrieval.chunks,
                    trace_id=trace_id,
                    team_id=tenant_config.team_id,
                )

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

    async def _classify_intent(self, state: AgentState) -> dict:
        """Classify the query intent: factual, analytical, graph, or hybrid."""
        query = state["query"].lower()

        if any(w in query for w in ("compare", "change", "trend", "over time", "years")):
            intent = "analytical"
        elif any(w in query for w in ("relationship", "supply", "competitor", "connected")):
            intent = "graph"
        else:
            intent = "factual"

        return {"intent": intent}

    async def _extract_entities(self, state: AgentState) -> dict:
        """Extract ticker symbols and company names from the raw query.

        Runs against the current question only. History never reaches this
        node, so a ticker mentioned three turns ago cannot leak into the
        current turn's retrieval filter.
        """
        query = state["query"]
        tickers = re.findall(r"\$([A-Z]{1,5})\b|(?<!\w)([A-Z]{2,5})(?=\s)", query)
        entities = list({t[0] or t[1] for t in tickers if any(t)})
        return {"entities": entities}

    async def _invoke_retrieval_agent(self, state: AgentState) -> dict:
        result = await self._retrieval_agent.retrieve(
            query=state["query"],
            tenant_config=state["tenant_config"],
            trace_id=state["trace_id"],
        )
        logger.debug(
            "retrieval complete chunks=%d errors=%d",
            len(result.chunks),
            len(result.errors),
        )
        return {"retrieval_result": result, "cache_hit": result.cache_hit}

    async def _invoke_graph_agent(self, state: AgentState) -> dict:
        if not state["entities"]:
            return {"graph_result": GraphResult()}

        result = await self._graph_agent.query(
            entities=state["entities"],
            tenant_config=state["tenant_config"],
            trace_id=state["trace_id"],
        )
        return {"graph_result": result}

    async def _merge_results(self, state: AgentState) -> dict:
        retrieval = state["retrieval_result"]
        if not retrieval or not retrieval.chunks:
            return {}
        top_score = max(c.score for c in retrieval.chunks) if retrieval.chunks else 0.0
        return {"context_quality_score": top_score}

    def _route_after_merge(self, state: AgentState) -> str:
        retrieval = state["retrieval_result"]
        if not retrieval or not retrieval.chunks:
            return "return_error"
        return "run_input_harness"

    async def _run_input_harness(self, state: AgentState) -> dict:
        retrieval = state["retrieval_result"]
        harness_result = run_input_harness(
            chunks=retrieval.chunks,
            max_context_tokens=state["tenant_config"].max_context_tokens,
            prompt_version=state["prompt_version"] or PROMPT_VERSION,
        )
        updated_retrieval = RetrievalResult(
            chunks=harness_result.chunks,
            cache_hit=retrieval.cache_hit,
            retrieval_method=retrieval.retrieval_method,
            total_tokens=harness_result.tokens_in_context,
            latency_ms=retrieval.latency_ms,
        )
        return {
            "retrieval_result": updated_retrieval,
            "prompt_version": harness_result.prompt_version,
        }

    async def _invoke_synthesis_agent(self, state: AgentState) -> dict:
        """Synthesize the answer.

        Uses conversation_context when present so the model sees prior turns,
        otherwise the raw query. This is the only node that reads the
        history-wrapped prompt.
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
        context, so the score reflects whether the answer is grounded in the
        retrieved chunks for the actual question asked.
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
        synthesis = state["synthesis_result"]
        warning = None

        if synthesis and synthesis.faithfulness_score < 0.85:
            claims = [c for c in synthesis.unsupported_claims if c.strip()]
            if claims:
                warning = f"Low confidence answer. Unsupported claims: {', '.join(claims[:3])}"
            else:
                warning = "Low confidence answer — please verify with source documents."

        response = QueryResponse(
            trace_id=state["trace_id"],
            answer=synthesis.answer if synthesis else "",
            citations=synthesis.citations if synthesis else [],
            faithfulness_score=synthesis.faithfulness_score if synthesis else 0.0,
            model_used=synthesis.model_used if synthesis else "unknown",
            latency_ms=synthesis.latency_ms if synthesis else 0.0,
            cache_hit=state["cache_hit"],
            warning=warning,
        )
        return {"final_response": response}

    async def _return_error(self, state: AgentState) -> dict:
        response = QueryResponse(
            trace_id=state["trace_id"],
            answer="No relevant context found. Cannot generate a grounded answer.",
            citations=[],
            faithfulness_score=0.0,
            model_used="none",
            latency_ms=0.0,
            cache_hit=False,
            warning="No context retrieved for this query.",
        )
        return {"final_response": response}