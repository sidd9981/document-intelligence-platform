"""
Tenant configuration and context models.

TenantConfig is loaded from Postgres at gateway startup and cached
in Redis. It defines what each team is allowed to do and how many
resources they can consume.

TenantContext is the runtime object created per request. It combines
the static config with request-specific data like the current token
usage and the active trace ID.
"""

from typing import Literal

from pydantic import BaseModel, Field


class TenantConfig(BaseModel):
    """Static configuration for a tenant team.

    Loaded once at startup from the tenant_configs table in Postgres.
    Cached in Redis with a 5 minute TTL so config changes propagate
    without requiring a restart.

    data_scopes controls which documents this team can retrieve.
    Every Qdrant query and every Neo4j Cypher query filters on this
    field. It is enforced at the data layer, not just the application
    layer.
    """

    team_id: str
    daily_token_budget: int = Field(gt=0)
    max_context_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    requests_per_minute: int = Field(gt=0)
    priority: Literal[1, 2, 3]
    allowed_models: list[str]
    retrieval_k: int = Field(gt=0)
    data_scopes: list[str]


class TenantContext(BaseModel):
    """Runtime context for a single request from a tenant.

    Created by the gateway for each incoming request and injected
    into the LangGraph agent state. Carried through every agent
    and tool call so resource usage can be metered accurately.
    """

    team_id: str
    config: TenantConfig
    trace_id: str
    tokens_used_today: int = Field(default=0, ge=0)
    request_tokens_estimate: int = Field(default=0, ge=0)