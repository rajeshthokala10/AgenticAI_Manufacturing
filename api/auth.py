"""Auth router for the FastAPI server.

Wires :mod:`core.auth_store` to four small endpoints and exposes two
dependency helpers used by the rest of the API:

* :func:`get_optional_user` — reads the Bearer token from the ``Authorization``
  header and returns the :class:`UserRecord` if valid; ``None`` otherwise.
  Used by ``/api/chat`` so anonymous chat keeps working when the gate is off.
* :func:`require_user` — same as above but raises 401 when there is no
  valid token. Used by ``/api/approvals/*`` so every approval is signed.

The router is mounted at ``/api/auth`` from ``api/server.py``.
"""

from __future__ import annotations

import logging
from typing import Optional

from fastapi import APIRouter, Depends, Header, HTTPException
from pydantic import BaseModel, Field

from core.auth_store import AuthError, UserRecord, get_default_store
from core.rbac import list_roles_public

logger = logging.getLogger("api.auth")

router = APIRouter(prefix="/api/auth", tags=["auth"])


# ─── Pydantic payloads ──────────────────────────────────────────────────────

class SignupRequest(BaseModel):
    user_id: str = Field(..., description="Email-style identifier (case-insensitive).")
    password: str = Field(..., min_length=6, description="Plaintext password (PBKDF2-hashed before storage).")
    role: str = Field(..., description="One of the role-ids returned by /api/auth/roles.")
    display_name: str = Field(default="", description="Optional human-friendly name.")


class LoginRequest(BaseModel):
    user_id: str
    password: str


class TokenResponse(BaseModel):
    token: str
    expires_at: float
    user: dict


# ─── Dependencies ───────────────────────────────────────────────────────────

def _token_from_header(authorization: Optional[str]) -> Optional[str]:
    if not authorization:
        return None
    parts = authorization.split(None, 1)
    if len(parts) != 2 or parts[0].lower() != "bearer":
        return None
    return parts[1].strip() or None


def get_optional_user(
    authorization: Optional[str] = Header(default=None),
) -> Optional[UserRecord]:
    """Return the authenticated user, or ``None`` for anonymous requests."""
    token = _token_from_header(authorization)
    if not token:
        return None
    return get_default_store().user_for_token(token)


def require_user(
    authorization: Optional[str] = Header(default=None),
) -> UserRecord:
    """Reject the request with 401 if no valid bearer token is present."""
    user = get_optional_user(authorization)
    if user is None:
        raise HTTPException(401, detail="Authentication required. POST /api/auth/login first.")
    return user


# ─── Routes ─────────────────────────────────────────────────────────────────

def _user_dict(user: UserRecord) -> dict:
    return {
        "user_id": user.user_id,
        "role": user.role,
        "display_name": user.display_name,
        "created_at": user.created_at,
    }


@router.get("/roles")
def list_roles() -> dict:
    """Public catalogue of roles for the signup screen / role badges."""
    return {"roles": list_roles_public()}


@router.post("/signup", response_model=TokenResponse)
def signup(req: SignupRequest) -> TokenResponse:
    store = get_default_store()
    try:
        user = store.signup(req.user_id, req.password, req.role, req.display_name)
        # Auto-login: signup returns a fresh token so the UI flows straight in.
        user, token, expires_at = store.login(req.user_id, req.password)
    except AuthError as exc:
        raise HTTPException(400, detail=str(exc)) from exc
    return TokenResponse(token=token, expires_at=expires_at, user=_user_dict(user))


@router.post("/login", response_model=TokenResponse)
def login(req: LoginRequest) -> TokenResponse:
    try:
        user, token, expires_at = get_default_store().login(req.user_id, req.password)
    except AuthError as exc:
        raise HTTPException(401, detail=str(exc)) from exc
    return TokenResponse(token=token, expires_at=expires_at, user=_user_dict(user))


@router.post("/logout")
def logout(authorization: Optional[str] = Header(default=None)) -> dict:
    token = _token_from_header(authorization)
    revoked = get_default_store().logout(token or "")
    return {"ok": True, "revoked": revoked}


@router.get("/me")
def me(user: UserRecord = Depends(require_user)) -> dict:
    return _user_dict(user)
