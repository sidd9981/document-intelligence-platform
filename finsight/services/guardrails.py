"""
Input guardrails.

Runs on every incoming query before it reaches the orchestrator.
Two concerns:

1. Prompt injection in the query itself — adversarial users trying
   to hijack the LLM via the query string. Separate from the chunk
   injection check in the input harness which scans retrieved documents.

2. Investment advice detection — FinSight is a document intelligence
   platform, not a financial advisor. Queries asking for buy/sell
   recommendations or price predictions have legal exposure and must
   be rejected before any LLM call is made.

Both checks are fast and synchronous — no LLM calls, no external
services. They run at the gateway layer before the orchestrator
so rejected queries consume zero tokens.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field

INVESTMENT_ADVICE_PATTERNS = [
    re.compile(r"\bshould\s+i\s+(buy|sell|invest|short|hold)\b", re.IGNORECASE),
    re.compile(r"\b(buy|sell|short|long)\s+(signal|recommendation|advice)\b", re.IGNORECASE),
    re.compile(r"\bwill\s+.{0,30}(stock|price|share).{0,20}(go up|go down|rise|fall|increase|decrease)\b", re.IGNORECASE),
    re.compile(r"\b(price target|fair value|intrinsic value|upside potential)\b", re.IGNORECASE),
    re.compile(r"\bshould\s+i\s+(invest|put my money|allocate)\b", re.IGNORECASE),
    re.compile(r"\bis\s+.{0,30}(stock|share|equity)\s+(worth|a good|undervalued|overvalued)\b", re.IGNORECASE),
]

QUERY_INJECTION_PATTERNS = [
    re.compile(r"ignore\s+(all\s+)?(previous|prior|above)\s+instructions", re.IGNORECASE),
    re.compile(r"disregard\s+(all\s+)?(previous|prior|above)", re.IGNORECASE),
    re.compile(r"you\s+are\s+now\s+a", re.IGNORECASE),
    re.compile(r"new\s+instruction[s]?\s*:", re.IGNORECASE),
    re.compile(r"system\s*:\s*you\s+are", re.IGNORECASE),
    re.compile(r"<\s*system\s*>", re.IGNORECASE),
    re.compile(r"forget\s+(everything|all)\s+(you|i)\s+(know|said|told)", re.IGNORECASE),
    re.compile(r"(pretend|act|behave)\s+(you\s+are|as\s+if|like\s+you)", re.IGNORECASE),
    re.compile(r"jailbreak", re.IGNORECASE),
    re.compile(r"dan\s+mode", re.IGNORECASE),
]


@dataclass
class GuardrailResult:
    allowed: bool
    reason: str | None = None
    violation_type: str | None = None


def check_query(query: str) -> GuardrailResult:
    """Run all guardrail checks on an incoming query.

    Called at the gateway before the query reaches the orchestrator.
    Returns immediately on first violation — no need to check all
    patterns once one fires.

    Args:
        query: The raw query string from the user.

    Returns:
        GuardrailResult with allowed=True if the query passes all
        checks, or allowed=False with a reason and violation type.
    """
    if not query or not query.strip():
        return GuardrailResult(
            allowed=False,
            reason="Query must not be empty.",
            violation_type="empty_query",
        )

    injection_result = _check_injection(query)
    if not injection_result.allowed:
        return injection_result

    advice_result = _check_investment_advice(query)
    if not advice_result.allowed:
        return advice_result

    return GuardrailResult(allowed=True)


def _check_injection(query: str) -> GuardrailResult:
    """Scan the query for prompt injection attempts."""
    for pattern in QUERY_INJECTION_PATTERNS:
        if pattern.search(query):
            return GuardrailResult(
                allowed=False,
                reason="Query contains content that cannot be processed.",
                violation_type="prompt_injection",
            )
    return GuardrailResult(allowed=True)


def _check_investment_advice(query: str) -> GuardrailResult:
    """Reject queries asking for investment advice or price predictions.

    FinSight is a document intelligence platform. Giving investment
    advice creates legal exposure. These queries are rejected before
    any LLM call so there is zero chance of the model complying.
    """
    for pattern in INVESTMENT_ADVICE_PATTERNS:
        if pattern.search(query):
            return GuardrailResult(
                allowed=False,
                reason=(
                    "FinSight provides document intelligence, not investment advice. "
                    "For investment decisions, consult a licensed financial advisor."
                ),
                violation_type="investment_advice",
            )
    return GuardrailResult(allowed=True)