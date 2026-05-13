"""Demo-grade user / token store for the HITL approval gate.

This module is **deliberately self-contained**: it uses only Python stdlib
(``sqlite3`` + ``hashlib.pbkdf2_hmac`` + ``secrets``) so it does not add any
new dependency to ``requirements.txt``. It exists to make the role-based
approval flow demoable end-to-end *without* wiring real OIDC/SAML — the
README explicitly lists "wire OIDC / SAML in your deployment infra" as the
production swap.

Schema
------
``users``     (user_id PRIMARY KEY, password_hash, salt, role, display_name,
                created_at)
``tokens``    (token PRIMARY KEY, user_id, expires_at)

Public surface
--------------
* :class:`UserRecord`     — dataclass returned by lookups.
* :class:`AuthError`      — raised on bad credentials / unknown user / role.
* :class:`UserStore`      — open/close, sign-up, login, logout, who-is-token.
* :func:`get_default_store`  — process-wide singleton.

Security notes
--------------
* Passwords stored as ``pbkdf2_hmac("sha256", pw, salt, 200_000)`` with a
  16-byte random salt per user. Not as strong as Argon2 / bcrypt but it's
  stdlib-only and adequate for a demo. Production: swap in ``passlib``.
* Tokens are 32 bytes of CSPRNG output (``secrets.token_urlsafe(32)``) with
  a 24-hour TTL. We expire on read, not via a background sweeper, so old
  rows linger in the DB — fine at demo scale.
* User-IDs are case-folded to lowercase to match the maker-lock check in
  ``core/rbac.py:is_maker_locked``.
"""

from __future__ import annotations

import hashlib
import logging
import secrets
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Optional

from config import PROCESSED_DIR
from core.rbac import ROLE_IDS, get_role

logger = logging.getLogger("core.auth")

# Default location. The HITL audit log uses ``HITL_DB_PATH``; auth gets its
# own file so the two SQLite stores stay independently swappable.
DEFAULT_AUTH_DB: Path = PROCESSED_DIR / "auth.sqlite"
TOKEN_TTL_SECONDS: int = 24 * 60 * 60  # 24 hours

_SCHEMA = """
CREATE TABLE IF NOT EXISTS users (
    user_id        TEXT PRIMARY KEY,
    password_hash  BLOB NOT NULL,
    salt           BLOB NOT NULL,
    role           TEXT NOT NULL,
    display_name   TEXT NOT NULL DEFAULT '',
    created_at     REAL NOT NULL
);
CREATE TABLE IF NOT EXISTS tokens (
    token       TEXT PRIMARY KEY,
    user_id     TEXT NOT NULL,
    expires_at  REAL NOT NULL,
    created_at  REAL NOT NULL
);
CREATE INDEX IF NOT EXISTS idx_tokens_user ON tokens(user_id);
"""

# Demo seed accounts — one per role. Passwords are intentionally weak so the
# demo flow is fast; rotate them or disable seeding in any real deployment.
DEMO_USERS: tuple[tuple[str, str, str, str], ...] = (
    ("alice@plant.local",     "operator123",     "operator",            "Alice Operator"),
    ("bob.supervisor@plant.local", "supervisor123", "shift_supervisor", "Bob Supervisor"),
    ("priya.planner@plant.local",  "planner123",   "maintenance_planner", "Priya Planner"),
    ("carol.eng@plant.local", "engineer123",     "maintenance_engineer", "Carol Engineer"),
    ("dave.ehs@plant.local",  "ehs123",          "ehs_officer",         "Dave EHS"),
    ("grace.qa@plant.local",  "quality123",      "quality_engineer",    "Grace QA"),
    ("eve.buyer@plant.local", "buyer123",        "buyer",               "Eve Buyer"),
    ("frank.proc@plant.local", "procurement123", "procurement_manager", "Frank Procurement"),
    ("henry.pm@plant.local",  "plant123",        "plant_manager",       "Henry Plant Manager"),
)


@dataclass(frozen=True)
class UserRecord:
    user_id: str
    role: str
    display_name: str
    created_at: float


class AuthError(Exception):
    """Raised on any signup/login/token failure. Callers should map to HTTP 401."""


# ─── Hash helpers ───────────────────────────────────────────────────────────

def _hash_password(password: str, salt: bytes) -> bytes:
    return hashlib.pbkdf2_hmac("sha256", password.encode("utf-8"), salt, 200_000)


def _new_salt() -> bytes:
    return secrets.token_bytes(16)


def _new_token() -> str:
    return secrets.token_urlsafe(32)


# ─── Store ──────────────────────────────────────────────────────────────────

class UserStore:
    """Thread-safe SQLite-backed user + token store."""

    def __init__(self, db_path: Path | str = DEFAULT_AUTH_DB, seed_demo: bool = True):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()
        if seed_demo:
            self._seed_demo_users()

    @contextmanager
    def _conn(self):
        with self._lock:
            conn = sqlite3.connect(str(self.db_path))
            conn.row_factory = sqlite3.Row
            conn.execute("PRAGMA journal_mode=WAL")
            conn.execute("PRAGMA synchronous=NORMAL")
            try:
                yield conn
                conn.commit()
            finally:
                conn.close()

    def _init_schema(self) -> None:
        with self._conn() as conn:
            conn.executescript(_SCHEMA)

    def _seed_demo_users(self) -> None:
        """Insert demo accounts on a fresh DB. Idempotent: skips if a user_id
        already exists, so re-running the API server never overwrites a
        password the operator may have changed."""
        for user_id, password, role, display in DEMO_USERS:
            try:
                self.signup(user_id, password, role, display_name=display)
            except AuthError as exc:
                # "already exists" is expected on a populated DB.
                logger.debug("seed skipped for %s: %s", user_id, exc)

    # ─── Sign-up / login ───────────────────────────────────────────────

    def signup(
        self,
        user_id: str,
        password: str,
        role: str,
        display_name: str = "",
    ) -> UserRecord:
        uid = (user_id or "").strip().lower()
        if not uid or "@" not in uid:
            raise AuthError("user_id must be a non-empty email-style identifier")
        if len(password) < 6:
            raise AuthError("password must be at least 6 characters")
        if role not in ROLE_IDS:
            raise AuthError(
                f"unknown role {role!r}. Valid roles: {', '.join(ROLE_IDS)}"
            )

        salt = _new_salt()
        ph = _hash_password(password, salt)
        now = time.time()
        try:
            with self._conn() as conn:
                conn.execute(
                    "INSERT INTO users (user_id, password_hash, salt, role, display_name, created_at) "
                    "VALUES (?, ?, ?, ?, ?, ?)",
                    (uid, ph, salt, role, display_name or uid.split("@")[0], now),
                )
        except sqlite3.IntegrityError:
            raise AuthError(f"user {uid!r} already exists")

        return UserRecord(user_id=uid, role=role, display_name=display_name, created_at=now)

    def login(self, user_id: str, password: str) -> tuple[UserRecord, str, float]:
        """Verify credentials and issue a bearer token (returns user + token + expiry)."""
        uid = (user_id or "").strip().lower()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT user_id, password_hash, salt, role, display_name, created_at "
                "FROM users WHERE user_id = ?",
                (uid,),
            ).fetchone()
            if row is None:
                raise AuthError("invalid credentials")
            expected = bytes(row["password_hash"])
            got = _hash_password(password, bytes(row["salt"]))
            if not secrets.compare_digest(expected, got):
                raise AuthError("invalid credentials")

            token = _new_token()
            now = time.time()
            expires_at = now + TOKEN_TTL_SECONDS
            conn.execute(
                "INSERT INTO tokens (token, user_id, expires_at, created_at) VALUES (?, ?, ?, ?)",
                (token, uid, expires_at, now),
            )

        user = UserRecord(
            user_id=row["user_id"],
            role=row["role"],
            display_name=row["display_name"] or uid.split("@")[0],
            created_at=float(row["created_at"]),
        )
        return user, token, expires_at

    def logout(self, token: str) -> bool:
        """Revoke a token. Returns True if a row was deleted."""
        if not token:
            return False
        with self._conn() as conn:
            cur = conn.execute("DELETE FROM tokens WHERE token = ?", (token,))
            return cur.rowcount > 0

    def user_for_token(self, token: Optional[str]) -> Optional[UserRecord]:
        """Resolve a bearer token to its user, or ``None`` if invalid/expired."""
        if not token:
            return None
        now = time.time()
        with self._conn() as conn:
            row = conn.execute(
                "SELECT t.user_id, t.expires_at, u.role, u.display_name, u.created_at "
                "FROM tokens t JOIN users u ON u.user_id = t.user_id "
                "WHERE t.token = ?",
                (token,),
            ).fetchone()
            if row is None:
                return None
            if float(row["expires_at"]) < now:
                # Lazy expiry — clean it up so we don't grow the table forever.
                conn.execute("DELETE FROM tokens WHERE token = ?", (token,))
                return None
        # Round-trip the role through the catalogue so we ignore stale roles
        # that were removed between sessions (forward-compat).
        if get_role(row["role"]) is None:
            return None
        return UserRecord(
            user_id=row["user_id"],
            role=row["role"],
            display_name=row["display_name"] or row["user_id"].split("@")[0],
            created_at=float(row["created_at"]),
        )

    def get_user(self, user_id: str) -> Optional[UserRecord]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT user_id, role, display_name, created_at FROM users WHERE user_id = ?",
                ((user_id or "").strip().lower(),),
            ).fetchone()
        if row is None:
            return None
        return UserRecord(
            user_id=row["user_id"],
            role=row["role"],
            display_name=row["display_name"] or row["user_id"].split("@")[0],
            created_at=float(row["created_at"]),
        )


# Process-wide singleton.
_default_store: Optional[UserStore] = None
_default_lock = threading.Lock()


def get_default_store() -> UserStore:
    global _default_store
    with _default_lock:
        if _default_store is None:
            _default_store = UserStore()
        return _default_store
