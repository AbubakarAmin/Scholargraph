"""
ScholarGraph Control Deck — FastAPI backend.
God's-eye view of runs, keys, events, scratchpad, cross-run memory.
"""

from __future__ import annotations

import json
import os
import shutil
import sys
import threading
from pathlib import Path
from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel
from fastapi import Request
from fastapi.responses import EventSourceResponse
import asyncio
import json

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
from core.capabilities import DEFAULT_MANIFESTS
from core.datasets import DATASET_CATALOG

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


class ResetPayload(BaseModel):
    confirmation: str


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
    index_path = STATIC / "admin" / "index.html"
    if index_path.exists():
        return FileResponse(index_path)
    return HTMLResponse("<h1>ScholarGraph</h1><p>static/admin/index.html missing</p>")


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
    values = {
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
        "gemini_embedding_model": config.gemini_embedding_model,
        "openai_embedding_model": config.openai_embedding_model,
        "debug_mode": config.debug_mode,
        "log_level": config.log_level,
        "vector_db_path": config.vector_db_path,
        "memory_size": config.memory_size,
        "cross_run_memory_path": config.cross_run_memory_path,
        "run_log_path": config.run_log_path,
        "run_events_path": config.run_events_path,
        "elo_ratings_path": config.elo_ratings_path,
        "output_dir": config.output_dir,
        "draft_versions_dir": config.draft_versions_dir,
        "debate_log_path": config.debate_log_path,
        "feedback_log_path": config.feedback_log_path,
        "raw_results_dir": config.raw_results_dir,
        "companion_repo_dir": config.companion_repo_dir,
        "sandbox_timeout_sec": config.sandbox_timeout_sec,
        "sandbox_max_output_bytes": config.sandbox_max_output_bytes,
        "web_host": config.web_host,
        "web_port": config.web_port,
        "keys_store_path": config.keys_store_path,
    }
    return values


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
        text = (call_llm("Reply with exactly: OK", temperature=0, tier="cheap", max_tokens=16) or "").strip()
        if not text:
            raise RuntimeError("Provider returned an empty response")
        return {
            "ok": True,
            "response": text[:200],
            "provider": config.llm_provider,
            "model": config.resolve_model("cheap"),
        }
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
                "workspace": summary.get("workspace", {}),
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
    state = _get_latest_state() if tracker and (run_id is None or run_id == tracker.run_id) else {}

    debates = []
    for item in state.get("debate_results", []):
        debates.append(item.__dict__ if hasattr(item, "__dict__") else item)
    if state:
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
            "data_artifacts": state.get("data_artifacts", {}),
            "data_validation": state.get("data_validation", {}),
            "execution_artifacts": state.get("execution_artifacts", {}),
            "analysis_reports": state.get("analysis_reports", {}),
            "verification_findings": state.get("verification_findings", []),
            "evidence_gate": state.get("evidence_gate", {}),
            "terminal_error": state.get("terminal_error"),
            "experiment_contracts": state.get("experiment_contracts", {}),
            "human_approved": state.get("human_approved", False),
        }
    dash["evidence_trace"] = research_db.claims(dash.get("run_id"))
    workspace = dash.get("workspace") or {}
    terminal = bool(workspace.get("terminal_error") or workspace.get("evidence_gate", {}).get("terminal"))
    dash["release"] = {
        "status": "blocked" if terminal else "pending_human_approval" if workspace.get("paper", {}).get("approval_required") and not workspace.get("human_approved", False) else "ready" if workspace.get("reproducibility", {}).get("passed") and not any(
            finding.get("blocking") for finding in workspace.get("verification_findings", [])
        ) else "incomplete",
        "reason": workspace.get("terminal_error") or workspace.get("evidence_gate", {}).get("message", ""),
    }
    dash["capabilities"] = DEFAULT_MANIFESTS
    return dash


@app.post("/api/release/approve")
def approve_release():
    """Record the explicit human checkpoint before a run is publishable."""
    latest_state = _get_latest_state()
    if not latest_state or latest_state.get("terminal_error"):
        raise HTTPException(status_code=409, detail="No successful run is awaiting approval")
    _set_latest_state({**latest_state, "human_approved": True})
    if latest_state.get("final_paper"):
        _set_latest_state({**_get_latest_state(), "final_paper": {**latest_state["final_paper"], "publishable": True}})
    tracker = get_tracker()
    if tracker:
        research_db.update_run_summary(tracker.run_id, {"human_approved": True, "publishable": True})
    from core.artifacts import save_results
    save_results(_get_latest_state())
    return {"ok": True, "publishable": True}


@app.get("/api/capabilities")
def capabilities():
    """Return the role/tool contract used by the current runtime."""
    return {"capabilities": DEFAULT_MANIFESTS}


@app.get("/api/artifacts")
def artifacts(run_id: Optional[str] = None):
    """Return durable artifact records for the selected run."""
    target_run_id = run_id or (get_tracker().run_id if get_tracker() else None)
    return {"artifacts": research_db.artifacts(target_run_id)}


@app.get("/api/events")
def events(limit: int = 100, run_id: Optional[str] = None):
    return {"events": read_events(limit=limit, run_id=run_id)}


@app.get("/api/scratchpad")
def scratchpad(limit: int = 50, run_id: Optional[str] = None):
    return {"entries": read_scratchpad(limit=limit, run_id=run_id)}


@app.get("/api/memory/feedback")
def feedback(limit: int = 20):
    return {"feedback": memory.get_recent_feedback(limit=limit)}


@app.get("/api/admin/export/{data_type}")
def export_data(data_type: str):
    """Export debates or agents data as CSV.
    No authentication required as per user request.
    """
    dash = dashboard()
    # Determine payload based on data_type
    if data_type == "debates":
        items = dash.get("workspace", {}).get("debates", [])
        headers = ["round", "topic", "winner", "score"]
        rows = []
        for d in items:
            # Attempt to extract common fields; fall back to raw dict
            rows.append([
                d.get("round", ""),
                d.get("topic", ""),
                d.get("winner", ""),
                d.get("score", ""),
            ])
    elif data_type == "agents":
        # Use capabilities manifest as a placeholder for agents info
        agents = dash.get("capabilities", [])
        headers = ["name", "role"]
        rows = []
        for a in agents:
            rows.append([a.get("name", ""), a.get("role", "")])
    else:
        raise HTTPException(status_code=400, detail="Unsupported export type")

    # Build CSV string
    import csv, io
    output = io.StringIO()
    writer = csv.writer(output)
    writer.writerow(headers)
    writer.writerows(rows)
    csv_bytes = output.getvalue().encode("utf-8")
    from fastapi.responses import StreamingResponse
    return StreamingResponse(io.BytesIO(csv_bytes), media_type="text/csv", headers={"Content-Disposition": f"attachment; filename={data_type}_export.csv"})



@app.get("/api/admin/stream")
async def admin_stream(request: Request):
    """Server‑Sent Events stream delivering real‑time admin updates.
    Sends a JSON payload with a `type` field so the client can dispatch.
    Currently emits the full dashboard payload every few seconds.
    """
    async def event_generator():
        while True:
            if await request.is_disconnected():
                break
            dash = dashboard()
            payload = {"type": "dashboard", "payload": dash}
            yield json.dumps(payload) + "\n\n"
            await asyncio.sleep(30)  # push updates every half minute
    return EventSourceResponse(event_generator())

@app.post("/api/data/reset/outputs")
def reset_outputs(payload: ResetPayload):
    """Clear generated paper artifacts without touching durable history."""
    global _latest_state, _run_error
    if payload.confirmation != "RESET_OUTPUTS":
        raise HTTPException(status_code=400, detail="Type RESET_OUTPUTS to confirm output reset")
    if _run_thread and _run_thread.is_alive():
        raise HTTPException(status_code=409, detail="Stop the active run before resetting outputs")
    protected = {Path(config.run_events_path).resolve(), Path(config.run_log_path).resolve()}
    candidates = {
        Path(config.output_dir),
        Path(config.draft_versions_dir),
        Path(config.raw_results_dir),
        Path(config.companion_repo_dir),
    }
    removed = []
    for candidate in candidates:
        resolved = candidate.resolve()
        if resolved in protected or not resolved.exists():
            continue
        if resolved.is_dir():
            for child in resolved.iterdir():
                if child.resolve() in protected:
                    continue
                if child.is_dir():
                    shutil.rmtree(child)
                else:
                    child.unlink()
                removed.append(str(child))
        else:
            resolved.unlink()
            removed.append(str(resolved))
    _set_latest_state({})
    _set_run_error(None)
    return {"ok": True, "scope": "outputs", "removed": removed, "history_preserved": True}


@app.post("/api/data/reset/catalog")
def reset_dataset_catalog(payload: ResetPayload):
    """Clear checked-in catalog assets only after a separate hard confirmation."""
    if payload.confirmation != "DELETE_DATASET_CATALOG":
        raise HTTPException(status_code=400, detail="Type DELETE_DATASET_CATALOG to confirm catalog reset")
    catalog_root = (ROOT / "data" / "catalog").resolve()
    removed = []
    if catalog_root.exists():
        for child in catalog_root.iterdir():
            if child.is_dir():
                shutil.rmtree(child)
            else:
                child.unlink()
            removed.append(str(child))
    DATASET_CATALOG.clear()
    return {"ok": True, "scope": "dataset_catalog", "removed": removed, "history_preserved": True}


@app.post("/api/data/clear")
def clear_all_data():
    """Retained as an explicit refusal so clients cannot wipe history by accident."""
    raise HTTPException(status_code=410, detail="Unscoped data clearing is disabled; use a scoped reset")


@app.delete("/api/runs/{run_id}")
def delete_single_run(run_id: str):
    research_db.delete_run(run_id)
    return {"ok": True, "message": f"Run {run_id} deleted."}


@app.get("/api/runs/{run_id}/lineage")
def run_claim_lineage(run_id: str):
    return {"run_id": run_id, "lineage": research_db.claim_lineage(run_id)}

def _workspace_snapshot(state: Dict[str, Any]) -> Dict[str, Any]:
    """Keep enough completed state for the admin panel to survive a refresh."""
    return {
        "debates": [
            item.__dict__ if hasattr(item, "__dict__") else item
            for item in state.get("debate_results", [])
        ],
        "plan": state.get("plan"),
        "plan_revision_requests": state.get("plan_revision_requests", []),
        "engineer_outputs": state.get("engineer_outputs", {}),
        "paper": state.get("final_paper") or {"sections": state.get("draft_sections", {})},
        "draft_sections": state.get("draft_sections", {}),
        "supervisor_scores": state.get("supervisor_scores", {}),
        "supervisor_feedback": state.get("supervisor_feedback", {}),
        "results_verification": state.get("results_verification", {}),
        "reproducibility": state.get("reproducibility", {}),
        "data_artifacts": state.get("data_artifacts", {}),
        "data_validation": state.get("data_validation", {}),
        "execution_artifacts": state.get("execution_artifacts", {}),
        "analysis_reports": state.get("analysis_reports", {}),
        "verification_findings": state.get("verification_findings", []),
        "evidence_gate": state.get("evidence_gate", {}),
        "terminal_error": state.get("terminal_error"),
        "experiment_contracts": state.get("experiment_contracts", {}),
        "current_phase": state.get("current_phase"),
        "iteration": state.get("iteration"),
        "topics": state.get("topics", []),
        "selected_topic": state.get("selected_topic"),
        "run_id": state.get("run_id"),
    }


def _load_state_from_db(run_id: str) -> Optional[Dict[str, Any]]:
    """Load historical workspace state from the research database for resume."""
    runs = research_db.list_runs(limit=100)
    found = next((r for r in runs if r["run_id"] == run_id), None)
    if not found:
        return None
    try:
        summary = json.loads(found.get("summary_json") or "{}")
    except Exception:
        return None
    workspace = summary.get("workspace", {})
    if not workspace:
        return None
    state = initialize_state()
    state["run_id"] = run_id
    state["plan"] = workspace.get("plan")
    state["plan_revision_requests"] = workspace.get("plan_revision_requests", [])
    state["engineer_outputs"] = workspace.get("engineer_outputs", {})
    state["draft_sections"] = workspace.get("draft_sections", {})
    state["supervisor_scores"] = workspace.get("supervisor_scores", {})
    state["supervisor_feedback"] = workspace.get("supervisor_feedback", {})
    state["results_verification"] = workspace.get("results_verification", {})
    state["reproducibility"] = workspace.get("reproducibility", {})
    state["data_artifacts"] = workspace.get("data_artifacts", {})
    state["data_validation"] = workspace.get("data_validation", {})
    state["execution_artifacts"] = workspace.get("execution_artifacts", {})
    state["analysis_reports"] = workspace.get("analysis_reports", {})
    state["verification_findings"] = workspace.get("verification_findings", [])
    state["evidence_gate"] = workspace.get("evidence_gate", {})
    state["terminal_error"] = workspace.get("terminal_error")
    state["experiment_contracts"] = workspace.get("experiment_contracts", {})
    state["current_phase"] = workspace.get("current_phase")
    state["iteration"] = workspace.get("iteration")
    state["topics"] = workspace.get("topics", [])
    state["selected_topic"] = workspace.get("selected_topic")
    if workspace.get("debates"):
        state["debate_results"] = workspace["debates"]
    return state


def _set_latest_state(state: Dict[str, Any]):
    """Thread-safe update of _latest_state."""
    global _latest_state
    with _run_lock:
        _latest_state = dict(state)


def _get_latest_state() -> Dict[str, Any]:
    """Thread-safe read of _latest_state."""
    global _latest_state
    with _run_lock:
        return dict(_latest_state)


def _set_run_error(error: Optional[str]):
    """Thread-safe update of _run_error."""
    global _run_error
    with _run_lock:
        _run_error = error


def _get_run_error() -> Optional[str]:
    """Thread-safe read of _run_error."""
    global _run_error
    with _run_lock:
        return _run_error


def _run_pipeline(domain: Optional[str] = None, resume_run_id: Optional[str] = None):
    global _run_error, _latest_state
    with _run_lock:
        _run_error = None
    try:
        if domain:
            apply_runtime_keys({"RESEARCH_DOMAIN": domain})
        validate_config()
        from main import create_checkpointer, create_research_graph, initialize_state, save_results
        from core.context import create_run_context
        from core.pipeline import ResearchPipeline

        if resume_run_id:
            tracker = start_run(resume_run_id)
            existing_state = _load_state_from_db(resume_run_id)
        else:
            tracker = start_run()
            existing_state = None
        pipeline = ResearchPipeline(
            create_research_graph,
            create_checkpointer,
            context=create_run_context(tracker),
        )
        if resume_run_id and existing_state:
            state = existing_state
        else:
            state = initialize_state()
        state["run_id"] = tracker.run_id
        _set_latest_state(state)
        def on_node(node_name, node_output):
            current = _get_latest_state()
            current.update(node_output)
            _set_latest_state(current)
            tracker.message(f"{node_name} -> {node_output.get('current_phase')}")

        result = pipeline.run(state, tracker.run_id, resume=bool(resume_run_id), on_node=on_node, finalize=save_results)
        last = result.state
        research_db.update_run_summary(tracker.run_id, {"workspace": _workspace_snapshot(last)})
        terminal_error = last.get("terminal_error")
        _set_run_error(terminal_error)
        if terminal_error:
            tracker.message(terminal_error, level="error")
    except Exception as e:
        _set_run_error(str(e))
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
            target=_run_pipeline, args=(payload.domain, None), daemon=True
        )
        _run_thread.start()
    return {"ok": True, "message": "Research run started"}


@app.post("/api/run/resume/{run_id}")
def resume_run(run_id: str):
    """Resume a previous run from where it left off using checkpoint data."""
    global _run_thread
    with _run_lock:
        if _run_thread and _run_thread.is_alive():
            raise HTTPException(status_code=409, detail="A run is already in progress")
        _run_thread = threading.Thread(
            target=_run_pipeline, args=(None, run_id), daemon=True
        )
        _run_thread.start()
    return {"ok": True, "message": f"Resuming run {run_id} from checkpoint"}


@app.get("/api/run/status")
def run_status():
    alive = bool(_run_thread and _run_thread.is_alive())
    tracker = get_tracker()
    return {
        "running": alive,
        "error": _get_run_error(),
        "tracker": tracker.dashboard() if tracker else None,
    }


@app.get("/api/runs")
def list_runs(limit: int = 50):
    """List all recorded runs for the admin panel to select from."""
    runs = research_db.list_runs(limit=limit)
    return {"runs": runs}


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
