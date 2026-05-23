"""
Eval harness stub.

Async evaluation on sampled live traffic. Computes RAGAS metrics
and logs to Langfuse. Filled in during Phase 6 when Langfuse
is wired up. Stubbed now so the orchestrator can call it without
Phase 6 being complete.
"""

from __future__ import annotations

import logging
import random

from finsight.models.synthesis import SynthesisResult
from finsight.telemetry.tracing import get_tracer

logger = logging.getLogger(__name__)
tracer = get_tracer(__name__)

EVAL_SAMPLE_RATE = 0.10


async def maybe_run_eval(
    query: str,
    result: SynthesisResult,
    trace_id: str,
) -> None:
    """Asynchronously evaluate a query-result pair on sampled traffic.

    Called by the orchestrator after the response is streamed so it
    never blocks the user. At EVAL_SAMPLE_RATE (10%) this computes
    RAGAS metrics and logs them. The other 90% returns immediately.

    Filled in Phase 6 with real RAGAS computation and Langfuse logging.
    """
    if random.random() > EVAL_SAMPLE_RATE:
        return

    with tracer.start_as_current_span("eval_harness.sample") as span:
        span.set_attribute("trace_id", trace_id)
        span.set_attribute("sampled", True)
        logger.debug("eval harness sampled trace %s — Phase 6 will compute RAGAS here", trace_id)