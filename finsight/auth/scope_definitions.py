"""
Scope definitions and team-to-scope mapping.

Single source of truth for authorization. Import from here everywhere
rather than hardcoding scope strings.
"""

from __future__ import annotations

SCOPE_DEFINITIONS = {
    "read:all_filings":     "Access all document collections",
    "read:public_filings":  "Access public filings only",
    "query:graph":          "Execute Neo4j graph queries",
    "model:large":          "Use 70B model tier",
    "model:medium":         "Use 22B model tier",
    "model:small":          "Use 8B model tier",
    "admin:config":         "Modify tenant configurations",
}

TEAM_SCOPES: dict[str, list[str]] = {
    "analysis": [
        "read:all_filings",
        "query:graph",
        "model:large",
        "model:medium",
    ],
    "risk": [
        "read:public_filings",
        "query:graph",
        "model:medium",
        "model:small",
    ],
    "ops": [
        "read:public_filings",
        "model:small",
    ],
}

DEV_CLIENTS: dict[str, str] = {
    "analysis": "dev-secret-analysis",
    "risk":     "dev-secret-risk",
    "ops":      "dev-secret-ops",
}