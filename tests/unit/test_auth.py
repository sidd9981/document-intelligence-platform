"""
Unit tests for auth components.

No running services needed. JWT encode/decode is pure crypto.
"""

from __future__ import annotations

import time

import jwt
import pytest
from fastapi import HTTPException

from finsight.auth.token_validator import (
    JWT_ALGORITHM,
    JWT_SECRET,
    decode_token,
    get_team_id,
    require_scope,
)
from finsight.auth.scope_definitions import DEV_CLIENTS, TEAM_SCOPES


def make_token(team_id: str = "ops", scopes: list[str] | None = None, expired: bool = False) -> str:
    now = int(time.time())
    payload = {
        "sub": f"team_{team_id}",
        "team_id": team_id,
        "scopes": scopes or TEAM_SCOPES.get(team_id, []),
        "iat": now,
        "exp": now - 10 if expired else now + 3600,
    }
    return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)


def test_decode_token_valid():
    token = make_token("ops")
    payload = decode_token(token)
    assert payload["team_id"] == "ops"


def test_decode_token_expired_raises_401():
    token = make_token(expired=True)
    with pytest.raises(HTTPException) as exc:
        decode_token(token)
    assert exc.value.status_code == 401
    assert "expired" in exc.value.detail


def test_decode_token_invalid_raises_401():
    with pytest.raises(HTTPException) as exc:
        decode_token("not.a.valid.token")
    assert exc.value.status_code == 401


def test_decode_token_wrong_secret_raises_401():
    payload = {"team_id": "ops", "scopes": [], "exp": int(time.time()) + 3600}
    token = jwt.encode(payload, "wrong-secret", algorithm=JWT_ALGORITHM)
    with pytest.raises(HTTPException) as exc:
        decode_token(token)
    assert exc.value.status_code == 401


def test_require_scope_passes_when_scope_present():
    payload = {"scopes": ["read:public_filings", "model:small"]}
    require_scope(payload, "model:small")


def test_require_scope_raises_403_when_missing():
    payload = {"scopes": ["model:small"]}
    with pytest.raises(HTTPException) as exc:
        require_scope(payload, "query:graph")
    assert exc.value.status_code == 403


def test_get_team_id_extracts_correctly():
    payload = {"team_id": "analysis", "scopes": []}
    assert get_team_id(payload) == "analysis"


def test_get_team_id_raises_401_when_missing():
    with pytest.raises(HTTPException) as exc:
        get_team_id({})
    assert exc.value.status_code == 401


def test_ops_team_cannot_access_graph():
    ops_scopes = TEAM_SCOPES["ops"]
    assert "query:graph" not in ops_scopes


def test_analysis_team_has_all_filings_access():
    analysis_scopes = TEAM_SCOPES["analysis"]
    assert "read:all_filings" in analysis_scopes


def test_dev_clients_exist_for_all_teams():
    for team_id in TEAM_SCOPES:
        assert team_id in DEV_CLIENTS