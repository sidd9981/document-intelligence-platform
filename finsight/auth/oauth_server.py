"""
OAuth2 client credentials server.

Issues JWTs to team services that present a valid client_id and secret.
Machine-to-machine auth only — no user login flows.
"""

from __future__ import annotations

import time

import jwt
from fastapi import FastAPI, HTTPException, Form
from fastapi.responses import JSONResponse

from finsight.auth.scope_definitions import DEV_CLIENTS, TEAM_SCOPES
from finsight.auth.token_validator import JWT_SECRET, JWT_ALGORITHM

app = FastAPI(title="FinSight OAuth Server")

JWT_EXPIRY_SECONDS = 3600


@app.post("/oauth/token")
async def issue_token(
    grant_type: str = Form(...),
    client_id: str = Form(...),
    client_secret: str = Form(...),
) -> JSONResponse:
    """Issue a JWT for a valid client credentials request."""
    if grant_type != "client_credentials":
        raise HTTPException(status_code=400, detail="unsupported_grant_type")

    expected_secret = DEV_CLIENTS.get(client_id)
    if not expected_secret or expected_secret != client_secret:
        raise HTTPException(status_code=401, detail="invalid_client")

    scopes = TEAM_SCOPES.get(client_id, [])
    now = int(time.time())

    payload = {
        "sub": f"team_{client_id}",
        "team_id": client_id,
        "scopes": scopes,
        "iat": now,
        "exp": now + JWT_EXPIRY_SECONDS,
    }

    token = jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)

    return JSONResponse({
        "access_token": token,
        "token_type": "bearer",
        "expires_in": JWT_EXPIRY_SECONDS,
        "scope": " ".join(scopes),
    })


@app.get("/health")
async def health() -> dict:
    return {"status": "ok"}