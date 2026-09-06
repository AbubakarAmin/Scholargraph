"""
Append-only run scratchpad, progress events (for UI), and cross-run memory.
Replaces lossy JSON handoffs with durable source-of-truth logs.
"""

from __future__ import annotations

import json
import threading
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import config
from .research_db import research_db

_lock = threading.Lock()


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _append_jsonl(path: str, record: Dict[str, Any]) -> None:
    p = Path(path)
    p.parent.mkdir(parents=True, exist_ok=True)
    with _lock:
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(record, default=str) + "\n")


class RunTracker:
    """Per-run observability + raw scratchpad shared by all agents."""

    def __init__(self, run_id: Optional[str] = None):
        self.run_id = run_id or str(uuid.uuid4())[:8]
        self.started_at = _now()
        self.stats: Dict[str, Any] = {
            "rejected_topics": 0,
            "debate_rounds": 0,
            "sections_bounced": 0,
            "pivots": 0,
            "refines": 0,
            "plan_revisions": 0,
            "hard_check_fails": 0,
            "llm_calls": 0,
            "cost_estimate_usd": 0.0,
        }
        self.phase = "idle"
        self.status = "idle"  # idle | running | completed | failed
        self.messages: List[str] = []
        research_db.create_run(self.run_id, self.started_at)
        emit_event(
            "run_created",
            {"run_id": self.run_id, "started_at": self.started_at},
            run_id=self.run_id,
        )

    def set_phase(self, phase: str, detail: Optional[Dict[str, Any]] = None):
        self.phase = phase
        self.status = "running"
        emit_event(
            "phase_change",
            {"phase": phase, **(detail or {})},
            run_id=self.run_id,
            agent="Orchestrator",
        )

    def scratch(self, agent: str, kind: str, content: Any, meta: Optional[Dict] = None):
        """Append raw reasoning/code/results — not a compressed summary."""
        record = {
            "ts": _now(),
            "run_id": self.run_id,
            "agent": agent,
            "kind": kind,
            "content": content,
            "meta": meta or {},
        }
        _append_jsonl(config.run_log_path, record)
        research_db.record_scratch(self.run_id, agent, kind, content, meta or {})
        emit_event(
            "scratchpad",
            {
                "agent": agent,
                "kind": kind,
                "preview": str(content)[:240],
                "meta": meta or {},
            },
            run_id=self.run_id,
            agent=agent,
        )

    def bump(self, key: str, amount: int = 1):
        if key in self.stats:
            self.stats[key] += amount
        emit_event("stat", {"key": key, "value": self.stats.get(key)}, run_id=self.run_id)

    def message(self, text: str, level: str = "info"):
        self.messages.append(text)
        emit_event("message", {"text": text, "level": level}, run_id=self.run_id)

    def complete(self, success: bool = True):
        self.status = "completed" if success else "failed"
        emit_event(
            "run_complete",
            {"success": success, "stats": self.stats},
            run_id=self.run_id,
        )
        CrossRunMemory().record_run(
            {
                "run_id": self.run_id,
                "started_at": self.started_at,
                "ended_at": _now(),
                "success": success,
                "stats": self.stats,
                "phase": self.phase,
            }
        )
        research_db.finish_run(self.run_id, self.status, self.phase, {"success": success, "stats": self.stats})

    def dashboard(self) -> Dict[str, Any]:
        return {
            "run_id": self.run_id,
            "started_at": self.started_at,
            "phase": self.phase,
            "status": self.status,
            "stats": self.stats,
            "recent_messages": self.messages[-20:],
        }


_active_tracker: Optional[RunTracker] = None


def get_tracker() -> Optional[RunTracker]:
    return _active_tracker


def start_run(run_id: Optional[str] = None) -> RunTracker:
    global _active_tracker
    _active_tracker = RunTracker(run_id=run_id)
    return _active_tracker


def emit_event(
    event_type: str,
    data: Dict[str, Any],
    run_id: Optional[str] = None,
    agent: Optional[str] = None,
):
    if run_id is None:
        from .context import get_active_context

        active_context = get_active_context()
        if active_context is not None:
            run_id = active_context.run_id or (
                active_context.tracker.run_id if active_context.tracker else None
            )
    record = {
        "ts": _now(),
        "type": event_type,
        "run_id": run_id,
        "agent": agent,
        "data": data,
    }
    _append_jsonl(config.run_events_path, record)
    research_db.record_event(event_type, data, run_id=run_id, agent=agent)


def read_events(limit: int = 200, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
    db_rows = research_db.recent_events(run_id, limit)
    if db_rows:
        return db_rows
    path = Path(config.run_events_path)
    if not path.exists():
        return []

    events = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
                if run_id and ev.get("run_id") != run_id:
                    continue
                events.append(ev)
            except json.JSONDecodeError:
                continue
    return events[-limit:]


def read_scratchpad(limit: int = 100, run_id: Optional[str] = None) -> List[Dict[str, Any]]:
    path = Path(config.run_log_path)
    if not path.exists():
        return []
    rows = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                row = json.loads(line)
                if run_id and row.get("run_id") != run_id:
                    continue
                rows.append(row)
            except json.JSONDecodeError:
                continue
    return rows[-limit:]


def build_run_summary(
    state: Optional[Dict[str, Any]] = None,
    error: Optional[str] = None,
) -> Dict[str, Any]:
    """Operator-facing summary of what was tried, where we got, and why we stopped."""
    state = state or {}
    tracker = get_tracker()
    run_id = state.get("run_id") or (tracker.run_id if tracker else None)
    events = read_events(limit=80, run_id=run_id)
    scratch = read_scratchpad(limit=40, run_id=run_id)
    topic = state.get("selected_topic") or {}
    plan = state.get("plan") or {}
    eng = state.get("engineer_outputs") or {}
    exp_rows = []
    for name, out in eng.items():
        if not isinstance(out, dict):
            continue
        exp_rows.append({
            "name": name,
            "success": bool(out.get("success")),
            "error": (out.get("error") or "")[:300],
            "metrics": out.get("aggregate_metrics") or out.get("results", {}).get("metrics"),
            "decision_log": out.get("decision_log") or [],
            "raw_results_path": out.get("raw_results_path"),
            "branch_search": out.get("branch_search"),
        })
    phase = state.get("current_phase") or (tracker.phase if tracker else "idle")
    terminal = error or state.get("terminal_error")
    stopped_because = (
        terminal
        or (state.get("meta_feedback") or [None])[-1]
        or ("completed with paper" if state.get("latex_output") else None)
        or "run ended"
    )
    summary = {
        "run_id": run_id,
        "phase": phase,
        "status": "failed" if terminal or (tracker and tracker.status == "failed") else (
            "completed" if state.get("latex_output") else (tracker.status if tracker else "idle")
        ),
        "stopped_because": stopped_because,
        "topic": {
            "title": topic.get("title") or topic.get("name"),
            "gap": topic.get("gap") or topic.get("research_gap"),
            "feasibility": topic.get("feasibility"),
        },
        "plan": {
            "contributions": (plan.get("contributions") or plan.get("expected_contributions") or [])[:5],
            "experiments_planned": [
                (e.get("name") if isinstance(e, dict) else str(e))
                for e in (plan.get("experiments") or [])
            ],
        },
        "what_we_tried": {
            "iterations": state.get("iteration", 0),
            "debates": len(state.get("debate_results") or []),
            "sections_drafted": list((state.get("draft_sections") or {}).keys()),
            "experiments": exp_rows,
            "plan_revisions": state.get("plan_revision_requests") or [],
            "stats": tracker.stats if tracker else {},
        },
        "where_we_got": {
            "supervisor_scores": state.get("supervisor_scores") or {},
            "results_verification": state.get("results_verification") or {},
            "reproducibility": state.get("reproducibility") or {},
            "has_latex": bool(state.get("latex_output")),
            "has_paper": bool(state.get("final_paper") or state.get("draft_sections")),
        },
        "recent_events": [
            {
                "ts": e.get("ts"),
                "agent": e.get("agent"),
                "type": e.get("type"),
                "data": e.get("data"),
            }
            for e in events[-15:]
        ],
        "recent_scratch": [
            {"agent": s.get("agent"), "kind": s.get("kind"), "content": s.get("content")}
            for s in scratch[-10:]
        ],
        "meta_feedback": state.get("meta_feedback") or [],
        "error": terminal,
    }
    # Human paragraph for chat / notice banner
    lines = [
        f"Run `{run_id or '?'}` stopped in phase **{phase}**.",
        f"Reason: {stopped_because}",
    ]
    if summary["topic"].get("title"):
        lines.append(f"Topic: {summary['topic']['title']}")
    if exp_rows:
        ok_n = sum(1 for x in exp_rows if x["success"])
        lines.append(f"Experiments: {ok_n}/{len(exp_rows)} succeeded.")
        for x in exp_rows[:4]:
            if x["success"]:
                lines.append(f"  ✓ {x['name']}: {x.get('metrics')}")
            else:
                lines.append(f"  ✕ {x['name']}: {x.get('error') or 'failed'}")
    elif summary["plan"]["experiments_planned"]:
        lines.append("Experiments were planned but none completed successfully.")
    sections = summary["what_we_tried"]["sections_drafted"]
    if sections:
        lines.append(f"Draft sections: {', '.join(sections)}")
    summary["narrative"] = "\n".join(lines)
    return summary


class CrossRunMemory:
    """Persists failures/rejections across runs for Topic Hunter / Planner."""

    def __init__(self, path: Optional[str] = None):
        self.path = Path(path or config.cross_run_memory_path)
        self.path.parent.mkdir(parents=True, exist_ok=True)

    def record(self, category: str, payload: Dict[str, Any]):
        entry = {
            "ts": _now(),
            "category": category,
            **payload,
        }
        _append_jsonl(str(self.path), entry)

    def record_rejection(self, kind: str, item: str, reason: str, meta: Optional[Dict] = None):
        self.record(
            "rejection",
            {"kind": kind, "item": item, "reason": reason, "meta": meta or {}},
        )

    def record_pivot(self, experiment: str, reason: str, meta: Optional[Dict] = None):
        self.record(
            "pivot",
            {"experiment": experiment, "reason": reason, "meta": meta or {}},
        )

    def record_plan_revision(self, reason: str, meta: Optional[Dict] = None):
        self.record("plan_revision", {"reason": reason, "meta": meta or {}})

    def record_run(self, summary: Dict[str, Any]):
        self.record("run_summary", summary)

    def load(self, category: Optional[str] = None, limit: int = 50) -> List[Dict[str, Any]]:
        if not self.path.exists():
            return []
        rows = []
        with open(self.path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    row = json.loads(line)
                    if category and row.get("category") != category:
                        continue
                    rows.append(row)
                except json.JSONDecodeError:
                    continue
        return rows[-limit:]

    def lessons_for_prompt(self, limit: int = 15) -> str:
        """Compact text block for Topic Hunter / Planner system context."""
        rejections = self.load("rejection", limit=limit)
        pivots = self.load("pivot", limit=8)
        revisions = self.load("plan_revision", limit=8)
        lines = ["## Lessons from prior runs (avoid repeating these failures)"]
        for r in rejections[-10:]:
            lines.append(f"- REJECTED {r.get('kind')}: {r.get('item')} — {r.get('reason')}")
        for p in pivots[-5:]:
            lines.append(f"- PIVOT on {p.get('experiment')}: {p.get('reason')}")
        for rev in revisions[-5:]:
            lines.append(f"- PLAN REVISION: {rev.get('reason')}")
        if len(lines) == 1:
            return "No prior-run lessons yet."
        return "\n".join(lines)

    def clear(self):
        with _lock:
            if self.path.exists():
                self.path.write_text("", encoding="utf-8")

