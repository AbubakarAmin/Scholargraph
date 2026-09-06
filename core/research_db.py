"""Durable SQLite ledger for runs, provenance, and evidence-backed claims."""
from __future__ import annotations

import json
import sqlite3
import threading
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import config


class ResearchDatabase:
    """Small transactional store; JSONL remains an export, not the source of truth."""
    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or config.research_db_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.lock = threading.Lock()
        self._migrate()

    def _connect(self):
        con = sqlite3.connect(str(self.path), timeout=15)
        con.row_factory = sqlite3.Row
        return con

    def _migrate(self):
        with self._connect() as con:
            con.executescript("""
            PRAGMA journal_mode=WAL;
            CREATE TABLE IF NOT EXISTS run_events (
              id INTEGER PRIMARY KEY, ts TEXT NOT NULL, run_id TEXT, agent TEXT,
              event_type TEXT NOT NULL, payload_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_events_run_ts ON run_events(run_id, id DESC);
            CREATE TABLE IF NOT EXISTS evidence_claims (
              id INTEGER PRIMARY KEY, run_id TEXT NOT NULL, section_name TEXT NOT NULL,
              claim_text TEXT NOT NULL, claim_type TEXT NOT NULL, status TEXT NOT NULL,
              evidence_json TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_claims_run ON evidence_claims(run_id, id DESC);
            CREATE TABLE IF NOT EXISTS research_artifacts (
              id INTEGER PRIMARY KEY, run_id TEXT, artifact_type TEXT NOT NULL,
              location TEXT, metadata_json TEXT NOT NULL, created_at TEXT NOT NULL
            );
            CREATE TABLE IF NOT EXISTS research_runs (
              run_id TEXT PRIMARY KEY, started_at TEXT NOT NULL, ended_at TEXT,
              status TEXT NOT NULL, phase TEXT, summary_json TEXT NOT NULL DEFAULT '{}'
            );
            CREATE TABLE IF NOT EXISTS run_scratchpad (
              id INTEGER PRIMARY KEY, ts TEXT NOT NULL, run_id TEXT NOT NULL,
              agent TEXT NOT NULL, kind TEXT NOT NULL, content_json TEXT NOT NULL,
              metadata_json TEXT NOT NULL
            );
            CREATE INDEX IF NOT EXISTS idx_scratch_run ON run_scratchpad(run_id, id DESC);
            """)

    @staticmethod
    def _now(): return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")

    def record_event(self, event_type: str, payload: Dict[str, Any], run_id: Optional[str] = None, agent: Optional[str] = None):
        with self.lock, self._connect() as con:
            con.execute("INSERT INTO run_events(ts,run_id,agent,event_type,payload_json) VALUES(?,?,?,?,?)",
                        (self._now(), run_id, agent, event_type, json.dumps(payload, default=str)))

    def create_run(self, run_id: str, started_at: str):
        with self.lock, self._connect() as con:
            con.execute("INSERT OR REPLACE INTO research_runs(run_id,started_at,status,summary_json) VALUES(?,?,?,?)", (run_id, started_at, "running", "{}"))

    def finish_run(self, run_id: str, status: str, phase: str, summary: Dict[str, Any]):
        with self.lock, self._connect() as con:
            con.execute("UPDATE research_runs SET ended_at=?, status=?, phase=?, summary_json=? WHERE run_id=?", (self._now(), status, phase, json.dumps(summary, default=str), run_id))
    def update_run_summary(self, run_id: str, updates: Dict[str, Any]):
        """Merge additional JSON-safe details into a completed run summary."""
        with self.lock, self._connect() as con:
            row = con.execute("SELECT summary_json FROM research_runs WHERE run_id=?", (run_id,)).fetchone()
            if not row:
                return
            try:
                summary = json.loads(row["summary_json"] or "{}")
            except json.JSONDecodeError:
                summary = {}
            summary.update(updates)
            con.execute(
                "UPDATE research_runs SET summary_json=? WHERE run_id=?",
                (json.dumps(summary, default=str), run_id),
            )

    def record_scratch(self, run_id: str, agent: str, kind: str, content: Any, metadata: Dict[str, Any]):
        with self.lock, self._connect() as con:
            con.execute("INSERT INTO run_scratchpad(ts,run_id,agent,kind,content_json,metadata_json) VALUES(?,?,?,?,?,?)", (self._now(), run_id, agent, kind, json.dumps(content, default=str), json.dumps(metadata, default=str)))

    def record_artifact(self, run_id: Optional[str], artifact_type: str, location: Optional[str], metadata: Dict[str, Any]):
        with self.lock, self._connect() as con:
            con.execute("INSERT INTO research_artifacts(run_id,artifact_type,location,metadata_json,created_at) VALUES(?,?,?,?,?)", (run_id, artifact_type, location, json.dumps(metadata, default=str), self._now()))

    def record_claim(self, run_id: str, section: str, text: str, claim_type: str, status: str, evidence: Dict[str, Any]):
        with self.lock, self._connect() as con:
            con.execute("INSERT INTO evidence_claims(run_id,section_name,claim_text,claim_type,status,evidence_json,created_at) VALUES(?,?,?,?,?,?,?)",
                        (run_id, section, text, claim_type, status, json.dumps(evidence, default=str), self._now()))

    def recent_events(self, run_id: Optional[str], limit: int = 100) -> List[Dict[str, Any]]:
        query, args = ("SELECT * FROM run_events WHERE run_id=? ORDER BY id DESC LIMIT ?", (run_id, limit)) if run_id else ("SELECT * FROM run_events ORDER BY id DESC LIMIT ?", (limit,))
        with self._connect() as con:
            rows = con.execute(query, args).fetchall()
        return [{"ts": r["ts"], "type": r["event_type"], "run_id": r["run_id"], "agent": r["agent"], "data": json.loads(r["payload_json"])} for r in reversed(rows)]

    def claims(self, run_id: Optional[str], limit: int = 200) -> List[Dict[str, Any]]:
        if not run_id: return []
        with self._connect() as con:
            rows = con.execute("SELECT * FROM evidence_claims WHERE run_id=? ORDER BY id DESC LIMIT ?", (run_id, limit)).fetchall()
        return [{"section": r["section_name"], "claim": r["claim_text"], "type": r["claim_type"], "status": r["status"], "evidence": json.loads(r["evidence_json"])} for r in reversed(rows)]

    def artifacts(self, run_id: Optional[str], limit: int = 200) -> List[Dict[str, Any]]:
        if not run_id:
            return []
        with self._connect() as con:
            rows = con.execute(
                "SELECT * FROM research_artifacts WHERE run_id=? ORDER BY id DESC LIMIT ?",
                (run_id, limit),
            ).fetchall()
        return [
            {
                "artifact_type": row["artifact_type"],
                "location": row["location"],
                "metadata": json.loads(row["metadata_json"]),
                "created_at": row["created_at"],
            }
            for row in reversed(rows)
        ]


    def list_runs(self, limit: int = 50) -> List[Dict[str, Any]]:
        with self._connect() as con:
            rows = con.execute("SELECT * FROM research_runs ORDER BY started_at DESC LIMIT ?", (limit,)).fetchall()
        return [dict(r) for r in rows]

    def delete_run(self, run_id: str):
        with self.lock, self._connect() as con:
            con.execute("DELETE FROM research_runs WHERE run_id=?", (run_id,))
            con.execute("DELETE FROM run_events WHERE run_id=?", (run_id,))
            con.execute("DELETE FROM run_scratchpad WHERE run_id=?", (run_id,))
            con.execute("DELETE FROM evidence_claims WHERE run_id=?", (run_id,))
            con.execute("DELETE FROM research_artifacts WHERE run_id=?", (run_id,))

    def clear_all(self):
        with self.lock, self._connect() as con:
            con.execute("DELETE FROM research_runs")
            con.execute("DELETE FROM run_events")
            con.execute("DELETE FROM run_scratchpad")
            con.execute("DELETE FROM evidence_claims")
            con.execute("DELETE FROM research_artifacts")


research_db = ResearchDatabase()
