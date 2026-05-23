"""
Seeds the tenant_configs table with the three default teams.

Safe to run multiple times — uses INSERT ... ON CONFLICT DO UPDATE
so existing rows are updated rather than causing a duplicate key error.

Usage:
    python scripts/seed_tenants.py
"""

import asyncio
import sys
from pathlib import Path

import asyncpg

sys.path.insert(0, str(Path(__file__).parent.parent))

from finsight.config.settings import settings

TENANTS = [
    {
        "team_id": "analysis",
        "daily_token_budget": 2_000_000,
        "max_context_tokens": 64_000,
        "max_output_tokens": 2_000,
        "requests_per_minute": 60,
        "priority": 1,
        "allowed_models": ["large", "medium"],
        "retrieval_k": 30,
        "data_scopes": ["public", "analysis"],
    },
    {
        "team_id": "risk",
        "daily_token_budget": 800_000,
        "max_context_tokens": 32_000,
        "max_output_tokens": 1_500,
        "requests_per_minute": 30,
        "priority": 2,
        "allowed_models": ["medium", "small"],
        "retrieval_k": 20,
        "data_scopes": ["public", "risk"],
    },
    {
        "team_id": "ops",
        "daily_token_budget": 200_000,
        "max_context_tokens": 8_000,
        "max_output_tokens": 500,
        "requests_per_minute": 20,
        "priority": 3,
        "allowed_models": ["small"],
        "retrieval_k": 5,
        "data_scopes": ["public"],
    },
]


async def seed() -> None:
    """Insert or update tenant configurations."""
    conn = await asyncpg.connect(settings.postgres.dsn)

    try:
        for tenant in TENANTS:
            await conn.execute(
                """
                INSERT INTO tenant_configs (
                    team_id,
                    daily_token_budget,
                    max_context_tokens,
                    max_output_tokens,
                    requests_per_minute,
                    priority,
                    allowed_models,
                    retrieval_k,
                    data_scopes
                ) VALUES (
                    $1, $2, $3, $4, $5, $6, $7, $8, $9
                )
                ON CONFLICT (team_id) DO UPDATE SET
                    daily_token_budget = EXCLUDED.daily_token_budget,
                    max_context_tokens = EXCLUDED.max_context_tokens,
                    max_output_tokens = EXCLUDED.max_output_tokens,
                    requests_per_minute = EXCLUDED.requests_per_minute,
                    priority = EXCLUDED.priority,
                    allowed_models = EXCLUDED.allowed_models,
                    retrieval_k = EXCLUDED.retrieval_k,
                    data_scopes = EXCLUDED.data_scopes,
                    updated_at = NOW()
                """,
                tenant["team_id"],
                tenant["daily_token_budget"],
                tenant["max_context_tokens"],
                tenant["max_output_tokens"],
                tenant["requests_per_minute"],
                tenant["priority"],
                tenant["allowed_models"],
                tenant["retrieval_k"],
                tenant["data_scopes"],
            )
            print(f"seeded tenant: {tenant['team_id']}")
    finally:
        await conn.close()


if __name__ == "__main__":
    asyncio.run(seed())