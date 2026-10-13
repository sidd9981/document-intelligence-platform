"""
Tenant configuration and context models.

TenantConfig is loaded from the tenant_configs table in Postgres by the
gateway on every request. It defines what each team is allowed to do and
how many resources it can consume.

TenantContext is a planned per-request object combining the static config
with live usage. It is defined here but not used by the pipeline yet.
"""

from typing import Literal

from pydantic import BaseModel, Field


class TenantConfig(BaseModel):
    """Static configuration for a tenant team.

    allowed_models lists the model tiers this team may use, in preference
    order, for example ["medium", "small"]. The first entry is the team's
    synthesis tier. finsight.services.llm.resolve_model maps a tier to an
    Ollama model name.

    readable_scopes is the set of document scopes this team may read: its
    own team scope, public documents, and any extra scopes in data_scopes.
    """

    team_id: str
    daily_token_budget: int = Field(gt=0)
    max_context_tokens: int = Field(gt=0)
    max_output_tokens: int = Field(gt=0)
    requests_per_minute: int = Field(gt=0)
    priority: Literal[1, 2, 3]
    allowed_models: list[str] = Field(min_length=1)
    retrieval_k: int = Field(gt=0)
    data_scopes: list[str]

    @property
    def default_model(self) -> str:
        """The team's synthesis tier, the first entry in allowed_models."""
        return self.allowed_models[0]

    @property
    def readable_scopes(self) -> list[str]:
        """Scopes this team may read, used by every data-layer filter."""
        return sorted({self.team_id, "public", *self.data_scopes})


class TenantContext(BaseModel):
    """Planned runtime context for a single request from a tenant.

    Not used by the pipeline yet.
    """

    team_id: str
    config: TenantConfig
    trace_id: str
    tokens_used_today: int = Field(default=0, ge=0)
    request_tokens_estimate: int = Field(default=0, ge=0)