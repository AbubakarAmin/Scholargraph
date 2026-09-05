"""
ScholarGraph Control Deck — FastAPI backend.
God's-eye view of runs, keys, events, scratchpad, cross-run memory.
"""

from __future__ import annotations

import json
import os
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from core.config import config, apply_runtime_keys, validate_config, sync_env_file
from core.llm import reset_llm_client, get_llm_client, call_llm
from core.run_log import (
    read_events,
    read_scratchpad,
    get_tracker,
    start_run,
    CrossRunMemory,
)
from core.memory import memory
from core.research_db import research_db

app = FastAPI(title="ScholarGraph Control Deck", version="2.0")
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

STATIC = Path(__file__).parent / "static"
STATIC.mkdir(exist_ok=True)

_run_thread: Optional[threading.Thread] = None
_run_error: Optional[str] = None
_run_lock = threading.Lock()
_latest_state: Dict[str, Any] = {}


class KeysPayload(BaseModel):
    keys: Dict[str, Any]


class RunPayload(BaseModel):
    domain: Optional[str] = None
    provider: Optional[str] = None


def _keys_path() -> Path:
    # Same resolution as core.config so UI and CLI share one keys file
    from core.config import _resolve_keys_path

    return _resolve_keys_path()


def load_keys() -> Dict[str, Any]:
    path = _keys_path()
    if path.exists():
        try:
            return json.loads(path.read_text(encoding="utf-8"))
        except Exception:
            return {}
    return {}


def _is_masked_secret(value: Any) -> bool:
    s = str(value or "")
    return ("…" in s) or s.startswith("••") or ("..." in s and len(s) <= 16)


def save_keys(keys: Dict[str, Any]) -> None:
    path = _keys_path()
    path.parent.mkdir(parents=True, exist_ok=True)
    # Mask nothing on disk — user asked to store keys; file is local
    path.write_text(json.dumps(keys, indent=2), encoding="utf-8")
    apply_runtime_keys(keys)
    sync_env_file(keys)  # keep .env in sync with UI changes
    reset_llm_client()


@app.on_event("startup")
def _startup():
    keys = load_keys()
    if keys:
        apply_runtime_keys(keys)
        reset_llm_client()


@app.get("/")
def index():
    index_path = STATIC / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>ScholarGraph</h1><p>static/index.html missing</p>")


@app.get("/api/health")
def health():
    return {
        "ok": True,
        "provider": config.llm_provider,
        "model": config.resolve_model("default"),
        "has_google_key": bool(config.google_api_key),
        "has_openai_key": bool(config.openai_api_key),
    }


@app.get("/api/config")
def get_config():
    return {
        "llm_provider": config.llm_provider,
        "gemini_model": config.gemini_model,
        "openai_model": config.openai_model,
        "openai_base_url": config.openai_base_url,
        "research_domain": config.research_domain,
        "supervisor_threshold": config.supervisor_threshold,
        "debate_pass_threshold": config.debate_pass_threshold,
        "experiment_seeds": config.experiment_seeds,
        "max_iterations": config.max_iterations,
        "debate_min_rounds": config.debate_min_rounds,
        "debate_max_rounds": config.debate_max_rounds,
        "novelty_similarity_reject": config.novelty_similarity_reject,
        "experiment_branch_count": config.experiment_branch_count,
        "openalex_email": config.openalex_email,
        "gemini_embedding_model": config.gemini_embedding_model,
        "openai_embedding_model": config.openai_embedding_model,
        "llm_model_cheap": config.llm_model_cheap,
        "llm_model_strong": config.llm_model_strong,
        "llm_model_judge": config.llm_model_judge,
        "ensemble_judge_models": config.ensemble_judge_models,
        "semantic_scholar_enabled": bool(config.semantic_scholar_api_key),
        "scite_enabled": bool(config.scite_api_key),
        "research_db_path": config.research_db_path,
        "checkpoint_path": config.checkpoint_path,
    }


@app.get("/api/keys")
def get_keys():
    """Return stored keys with secrets masked for display."""
    keys = load_keys()
    masked = {}
    secret_keys = {
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "SCITE_API_KEY",
        "SEMANTIC_SCHOLAR_API_KEY",
    }
    for k, v in keys.items():
        if k in secret_keys and v:
            s = str(v)
            masked[k] = (s[:4] + "…" + s[-4:]) if len(s) > 8 else "••••"
        else:
            masked[k] = v
    return {"keys": masked, "raw_present": {k: bool(keys.get(k)) for k in secret_keys}}



@app.post("/api/keys")
def post_keys(payload: KeysPayload):
    existing = load_keys()
    secret_keys = {
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "SCITE_API_KEY",
        "SEMANTIC_SCHOLAR_API_KEY",
    }
    # Don't overwrite secrets with masked placeholders or empty fields
    for k, v in payload.keys.items():
        if v is None or v == "":
            continue
        if k in secret_keys and _is_masked_secret(v):
            continue
        if _is_masked_secret(v) and k in existing:
            continue
        existing[k] = v
    save_keys(existing)
    return {
        "ok": True,
        "saved": list(payload.keys.keys()),
        "path": str(_keys_path()),
        "env_path": str(ROOT / ".env"),
        "has_openai_key": bool(existing.get("OPENAI_API_KEY")),
        "has_google_key": bool(existing.get("GOOGLE_API_KEY")),
        "provider": existing.get("LLM_PROVIDER") or config.llm_provider,
    }


@app.post("/api/keys/test")
def test_llm():
    try:
        validate_config()
        text = call_llm("Reply with exactly: OK", temperature=0, tier="cheap", max_tokens=16)
        return {"ok": True, "response": (text or "").strip()[:200]}
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@app.get("/api/dashboard")
def dashboard(run_id: Optional[str] = None):
    tracker = get_tracker()
    target_run_id = run_id or (tracker.run_id if tracker else None)
    if tracker and (run_id is None or run_id == tracker.run_id):
        dash = tracker.dashboard()
    else:
        # Check historical run from db
        runs = research_db.list_runs(limit=100)
        found = next((r for r in runs if r["run_id"] == target_run_id), None) if target_run_id else None
        if found:
            summary = {}
            try:
                summary = json.loads(found.get("summary_json") or "{}")
            except Exception:
                pass
            dash = {
                "run_id": found["run_id"],
                "status": found["status"],
                "phase": found["phase"] or "completed",
                "stats": summary.get("stats", {}),
                "recent_messages": [],
            }
        else:
            dash = {
                "run_id": target_run_id,
                "status": "idle",
                "phase": "idle",
                "stats": {},
                "recent_messages": [],
            }
    dash["events_tail"] = read_events(limit=50, run_id=target_run_id)
    dash["cross_run"] = {
        "rejections": CrossRunMemory().load("rejection", limit=10),
        "pivots": CrossRunMemory().load("pivot", limit=5),
        "runs": CrossRunMemory().load("run_summary", limit=5),
    }
    dash["config"] = {
        "provider": config.llm_provider,
        "model": config.resolve_model("default"),
        "domain": config.research_domain,
    }
    state = _latest_state or {}

    debates = []
    for item in state.get("debate_results", []):
        debates.append(item.__dict__ if hasattr(item, "__dict__") else item)
    dash["workspace"] = {
        "debates": debates,
        "plan": state.get("plan"),
        "plan_revision_requests": state.get("plan_revision_requests", []),
        "engineer_outputs": state.get("engineer_outputs", {}),
        "paper": state.get("final_paper") or {"sections": state.get("draft_sections", {})},
        "draft_sections": state.get("draft_sections", {}),
        "supervisor_scores": state.get("supervisor_scores", {}),
        "supervisor_feedback": state.get("supervisor_feedback", {}),
        "results_verification": state.get("results_verification", {}),
        "reproducibility": state.get("reproducibility", {}),
    }
    dash["evidence_trace"] = research_db.claims(dash.get("run_id"))
    return dash


@app.get("/api/events")
def events(limit: int = 100, run_id: Optional[str] = None):
    return {"events": read_events(limit=limit, run_id=run_id)}


@app.get("/api/scratchpad")
def scratchpad(limit: int = 50, run_id: Optional[str] = None):
    return {"entries": read_scratchpad(limit=limit, run_id=run_id)}


@app.get("/api/memory/feedback")
def feedback(limit: int = 20):
    return {"feedback": memory.get_recent_feedback(limit=limit)}


@app.get("/api/runs")
def get_runs(limit: int = 50):
    return {"runs": research_db.list_runs(limit=limit)}


@app.post("/api/data/clear")
def clear_all_data():
    global _latest_state, _run_error
    research_db.clear_all()
    memory.clear_all()
    CrossRunMemory().clear()
    _latest_state = {}
    _run_error = None
    return {"ok": True, "message": "All runs, memories, and event logs cleared."}


@app.delete("/api/runs/{run_id}")
def delete_single_run(run_id: str):
    research_db.delete_run(run_id)
    return {"ok": True, "message": f"Run {run_id} deleted."}


def _run_pipeline(domain: Optional[str] = None):
    global _run_error, _latest_state
    _run_error = None
    try:
        if domain:
            apply_runtime_keys({"RESEARCH_DOMAIN": domain})
        validate_config()
        from main import create_research_graph, create_checkpointer, initialize_state, save_results

        tracker = start_run()
        workflow = create_research_graph()
        app_graph = workflow.compile(checkpointer=create_checkpointer())
        state = initialize_state()
        state["run_id"] = tracker.run_id
        _latest_state = state
        cfg = {
            "configurable": {"thread_id": tracker.run_id},
            "recursion_limit": 1000,
        }
        last = state
        for event in app_graph.stream(state, cfg):
            for node_name, node_output in event.items():
                if node_name != "__end__":
                    last = node_output
                    _latest_state = node_output
                    tracker.message(f"{node_name} → {node_output.get('current_phase')}")
        save_results(last)
        terminal_error = last.get("terminal_error")
        _run_error = terminal_error
        if terminal_error:
            tracker.message(terminal_error, level="error")
        tracker.complete(success=bool(last.get("latex_output")) and not terminal_error)
    except Exception as e:
        _run_error = str(e)
        tracker = get_tracker()
        if tracker:
            tracker.message(f"ERROR: {e}", level="error")
            tracker.complete(success=False)


@app.post("/api/run")
def start_research(payload: RunPayload = RunPayload()):
    global _run_thread
    with _run_lock:
        if _run_thread and _run_thread.is_alive():
            raise HTTPException(status_code=409, detail="A run is already in progress")
        if payload.provider:
            apply_runtime_keys({"LLM_PROVIDER": payload.provider})
            reset_llm_client()
        _run_thread = threading.Thread(
            target=_run_pipeline, args=(payload.domain,), daemon=True
        )
        _run_thread.start()
    return {"ok": True, "message": "Research run started"}


@app.get("/api/run/status")
def run_status():
    alive = bool(_run_thread and _run_thread.is_alive())
    tracker = get_tracker()
    return {
        "running": alive,
        "error": _run_error,
        "tracker": tracker.dashboard() if tracker else None,
    }


if STATIC.exists():
    app.mount("/static", StaticFiles(directory=str(STATIC)), name="static")


def main():
    import uvicorn

    uvicorn.run(
        "web.app:app",
        host=config.web_host,
        port=config.web_port,
        reload=False,
    )


if __name__ == "__main__":
    main()
