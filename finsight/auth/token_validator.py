"""
JWT token validator.

Used by the gateway and by each MCP server independently.
Defense in depth — every layer validates, not just the gateway.
"""

from __future__ import annotations

import jwt
from fastapi import HTTPException

JWT_SECRET = "changeme-use-vault-in-prod"
JWT_ALGORITHM = "HS256"


def decode_token(token: str) -> dict:
    """Decode and validate a JWT.

    Raises HTTPException on any validation failure so callers
    never see raw jwt exceptions.
    """
    try:
        payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
        return payload
    except jwt.ExpiredSignatureError:
        raise HTTPException(status_code=401, detail="token expired")
    except jwt.InvalidTokenError as e:
        raise HTTPException(status_code=401, detail=f"invalid token: {e}")


def require_scope(payload: dict, required_scope: str) -> None:
    """Raise 403 if the token does not carry the required scope."""
    scopes = payload.get("scopes", [])
    if required_scope not in scopes:
        raise HTTPException(
            status_code=403,
            detail=f"missing required scope: {required_scope}",
        )


def get_team_id(payload: dict) -> str:
    """Extract team_id from a decoded JWT payload."""
    team_id = payload.get("team_id")
    if not team_id:
        raise HTTPException(status_code=401, detail="token missing team_id claim")
    return team_id