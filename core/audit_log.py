"""Append-only audit log for HITL approvals (Phase B).

Single SQLite database (default: ``data/processed/audit.sqlite``) recording
every approval decision so we have a durable record of *who* approved *what*
and *why*. Used by:

* ``api/server.py`` — writes a row at every ``/api/approvals/{id}/resume``.
* ``app.py`` (Streamlit "📋 Approvals" tab) — reads the most recent N rows
  to surface as a recent-decisions panel.

The log is intentionally simple — one table, one writer-per-process — because
the deployment story is "one FastAPI server in front of one SQLite file" for
this iteration. Multi-process deployments should swap this for Postgres
(same schema). The writer uses WAL mode + ``PRAGMA synchronous=NORMAL`` for
decent throughput while remaining crash-safe.
"""

from __future__ import annotations

import json
import logging
import sqlite3
import threading
import time
from contextlib import contextmanager
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional

from config import HITL_DB_PATH

logger = logging.getLogger("core.audit")


_SCHEMA = """
CREATE TABLE IF NOT EXISTS approvals (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    ts              REAL    NOT NULL,             -- unix epoch seconds
    thread_id       TEXT    NOT NULL,
    decision        TEXT    NOT NULL,             -- 'approved' | 'rejected'
    approver        TEXT    NOT NULL DEFAULT 'unknown',
    risk_score      REAL    NOT NULL DEFAULT 0.0,
    drivers_json    TEXT    NOT NULL DEFAULT '[]',
    domain          TEXT    NOT NULL DEFAULT 'diagnostic',  -- 'diagnostic' | 'purchase_request' | …
    query           TEXT    NOT NULL DEFAULT '',
    proposed_answer TEXT    NOT NULL DEFAULT '',
    edited_answer   TEXT,
    comments        TEXT
);
CREATE INDEX IF NOT EXISTS idx_approvals_thread ON approvals(thread_id);
CREATE INDEX IF NOT EXISTS idx_approvals_ts     ON approvals(ts DESC);
"""


@dataclass
class AuditEntry:
    ts: float
    thread_id: str
    decision: str
    approver: str
    risk_score: float
    drivers: List[str]
    domain: str
    query: str
    proposed_answer: str
    edited_answer: Optional[str] = None
    comments: Optional[str] = None
    id: Optional[int] = field(default=None)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "id": self.id,
            "ts": self.ts,
            "ts_iso": time.strftime("%Y-%m-%d %H:%M:%S", time.localtime(self.ts)),
            "thread_id": self.thread_id,
            "decision": self.decision,
            "approver": self.approver,
            "risk_score": round(float(self.risk_score), 4),
            "drivers": self.drivers,
            "domain": self.domain,
            "query": self.query,
            "proposed_answer": self.proposed_answer,
            "edited_answer": self.edited_answer,
            "comments": self.comments,
        }


class AuditLog:
    """Thread-safe SQLite-backed audit log."""

    def __init__(self, db_path: Path | str = HITL_DB_PATH):
        self.db_path = Path(db_path)
        self.db_path.parent.mkdir(parents=True, exist_ok=True)
        self._lock = threading.Lock()
        self._init_schema()

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

    # ─── Writers ──────────────────────────────────────────────────────────

    def record(
        self,
        *,
        thread_id: str,
        decision: str,
        approver: str = "unknown",
        risk_score: float = 0.0,
        drivers: Optional[Iterable[str]] = None,
        domain: str = "diagnostic",
        query: str = "",
        proposed_answer: str = "",
        edited_answer: Optional[str] = None,
        comments: Optional[str] = None,
        ts: Optional[float] = None,
    ) -> int:
        """Insert a decision row. Returns the new row id."""
        if decision not in ("approved", "rejected"):
            raise ValueError(f"decision must be 'approved' or 'rejected', got {decision!r}")
        drivers_list = list(drivers or [])
        with self._conn() as conn:
            cur = conn.execute(
                """
                INSERT INTO approvals
                  (ts, thread_id, decision, approver, risk_score,
                   drivers_json, domain, query, proposed_answer,
                   edited_answer, comments)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    ts if ts is not None else time.time(),
                    thread_id,
                    decision,
                    approver,
                    float(risk_score),
                    json.dumps(drivers_list),
                    domain,
                    query,
                    proposed_answer,
                    edited_answer,
                    comments,
                ),
            )
            return int(cur.lastrowid or 0)

    # ─── Readers ──────────────────────────────────────────────────────────

    def recent(self, limit: int = 50, offset: int = 0) -> List[AuditEntry]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM approvals ORDER BY ts DESC LIMIT ? OFFSET ?",
                (int(limit), int(offset)),
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def for_thread(self, thread_id: str) -> List[AuditEntry]:
        with self._conn() as conn:
            rows = conn.execute(
                "SELECT * FROM approvals WHERE thread_id = ? ORDER BY ts ASC",
                (thread_id,),
            ).fetchall()
        return [self._row_to_entry(r) for r in rows]

    def stats(self) -> Dict[str, Any]:
        with self._conn() as conn:
            row = conn.execute(
                "SELECT COUNT(*) AS total, "
                "SUM(CASE WHEN decision='approved' THEN 1 ELSE 0 END) AS approved, "
                "SUM(CASE WHEN decision='rejected' THEN 1 ELSE 0 END) AS rejected "
                "FROM approvals"
            ).fetchone()
        total = int(row["total"] or 0)
        approved = int(row["approved"] or 0)
        rejected = int(row["rejected"] or 0)
        return {
            "total": total,
            "approved": approved,
            "rejected": rejected,
            "approval_rate": round(approved / total, 3) if total else 0.0,
        }

    @staticmethod
    def _row_to_entry(row: sqlite3.Row) -> AuditEntry:
        try:
            drivers = json.loads(row["drivers_json"] or "[]")
            if not isinstance(drivers, list):
                drivers = []
        except json.JSONDecodeError:
            drivers = []
        return AuditEntry(
            id=int(row["id"]),
            ts=float(row["ts"]),
            thread_id=row["thread_id"],
            decision=row["decision"],
            approver=row["approver"],
            risk_score=float(row["risk_score"]),
            drivers=drivers,
            domain=row["domain"],
            query=row["query"],
            proposed_answer=row["proposed_answer"],
            edited_answer=row["edited_answer"],
            comments=row["comments"],
        )


# Process-wide singleton for convenience.
_default_log: Optional[AuditLog] = None
_default_lock = threading.Lock()


def get_default_log() -> AuditLog:
    global _default_log
    with _default_lock:
        if _default_log is None:
            _default_log = AuditLog()
        return _default_log
