"""Phase node implementations for the ScholarGraph research workflow."""

import json
import logging
import re
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from core.config import config
from core.context import get_active_context
from core.evidence_gate import build_contract, gate_engineering_outputs, validate_dataset_identity, validate_experiments
from core.llm import call_llm
from core.run_log import CrossRunMemory, get_tracker
from core.research_db import research_db
from core.state import ResearchState, initialize_state
from core.utils import log_agent_action, parse_json_from_llm, call_llm_json
from core.verification import reproducibility_dossier, validate_empirical_claims, verify_citations

from agents.editor import EditorAgent
from agents.data import DataAgent
from agents.analysis import AnalysisAgent
from agents.engineer import EngineerAgent
from agents.execution import ExecutionAgent
from agents.hypothesis_debate import HypothesisDebateSystem
from agents.meta_agent import MetaAgent
from agents.planner import PlannerAgent
from agents.supervisor import SupervisorAgent
from agents.topic_hunter import ResearchSourceUnavailable, TopicHunterAgent
from agents.verification import VerificationAgent
from agents.writer import WriterAgent

logger = logging.getLogger(__name__)


def _create_agent(agent_class):
    context = get_active_context()
    return agent_class(context) if context is not None else agent_class()


def topic_discovery_node(state: ResearchState) -> ResearchState:
    log_agent_action("Orchestrator", "start_topic_discovery", {"iteration": state["iteration"]})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("topic_discovery")
    try:
        topics = _create_agent(TopicHunterAgent).discover_topics(config.research_domain)
        if topics:
            state["topics"] = topics
            state["current_phase"] = "hypothesis_debate"
            log_agent_action("Orchestrator", "topics_discovered", {"count": len(topics), "iteration": state["iteration"], "topics": [topic["title"] for topic in topics[:3]]})
        elif state["iteration"] >= 5:
            # Exhausting discovery is a failed research run, not a successful
            # completion.  Preserve that distinction so the artifact layer
            # writes a failure dossier rather than an empty normal summary.
            message = "No viable, evidence-supported, sandbox-executable topic was discovered after multiple attempts"
            state["current_phase"] = "complete"
            state["should_continue"] = False
            state["should_reset"] = False
            state["terminal_error"] = message
            state["technical_failures"] = {
                "topic_discovery": {
                    "success": False,
                    "failure_kind": "research_exhausted",
                    "reason_code": "no_viable_topic_after_retries",
                    "message": message,
                }
            }
            state["meta_feedback"].append(message)
            log_agent_action("Orchestrator", "no_topics_found_after_retries", {
                "iteration": state["iteration"],
                "reason_code": "no_viable_topic_after_retries",
            })
        else:
            state["should_reset"] = True
            state["meta_feedback"].append("No topics discovered - resetting")
            log_agent_action("Orchestrator", "no_topics_found", {"iteration": state["iteration"]})
        return state
    except ResearchSourceUnavailable as exc:
        message = str(exc)
        logger.warning(message)
        state["terminal_error"] = message
        state["meta_feedback"].append(message)
        state["current_phase"] = "complete"
        state["should_continue"] = False
        log_agent_action("Orchestrator", "research_sources_unavailable", {"message": message})
        return state
    except Exception as exc:
        logger.error(f"Topic discovery failed: {exc}")
        if state["iteration"] >= 5:
            state["current_phase"] = "complete"
            state["meta_feedback"].append(f"Topic discovery failed after multiple attempts: {exc}")
        else:
            state["should_reset"] = True
            state["meta_feedback"].append(f"Topic discovery error: {exc}")
        return state


def hypothesis_debate_node(state: ResearchState) -> ResearchState:
    log_agent_action("Orchestrator", "start_hypothesis_debate", {"topics_remaining": len(state["topics"])})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("hypothesis_debate")
    if state.get("terminal_error") or state.get("current_phase") == "complete":
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    try:
        # Filter out topics that already failed or were rejected (including within-iteration failures)
        excluded = set(t.strip().lower() for t in CrossRunMemory().excluded_topic_titles() if t)
        state["topics"] = [
            t for t in state["topics"]
            if (t.get("title") or "").strip().lower() not in excluded
        ]
        if not state["topics"]:
            state["should_reset"] = True
            return state
        # Pre-debate self-critique: strengthen hypotheses before adversarial debate
        try:
            _th_agent = _create_agent(TopicHunterAgent)
            state["topics"] = [_th_agent.pre_debate_self_critique(t) for t in state["topics"]]
        except Exception as critique_err:
            log_agent_action("Orchestrator", "pre_debate_critique_error", {"error": str(critique_err)})
        topics_tried = 0
        debater = _create_agent(HypothesisDebateSystem)
        if len(state["topics"]) >= 3 and hasattr(debater, "conduct_tournament"):
            tournament_results = debater.conduct_tournament(state["topics"], rounds=1)
            state["debate_results"].extend(tournament_results)
            winner = next((result for result in tournament_results if result.passed), None)
            if winner:
                state["selected_topic"] = next(
                    topic for topic in state["topics"] if topic.get("title") == winner.topic
                )
                state["hypothesis_passed"] = True
                state["current_phase"] = "planning"
                return state
            for result in tournament_results:
                CrossRunMemory().record_rejection(
                    "topic",
                    getattr(result, "topic", "?"),
                    "failed_hypothesis_debate",
                    {
                        "score": getattr(result, "score", None),
                        "decision": getattr(result, "moderator_decision", None),
                        "unresolved": list(getattr(result, "unresolved_objections", []) or [])[:5],
                    },
                )
            state["topics"] = []
            state["should_reset"] = True
            return state
        while state["topics"]:
            current_topic = state["topics"][0]
            state["selected_topic"] = current_topic
            topics_tried += 1
            log_agent_action("Orchestrator", "trying_topic", {"topic": current_topic["title"], "attempt": topics_tried, "topics_remaining": len(state["topics"])})
            # Both tournament and serial discovery use the same bounded
            # repair protocol.  Without this branch, a run with one or two
            # candidates silently lost the contract-revision capability.
            attempts = (
                debater.conduct_with_repair(current_topic)
                if hasattr(debater, "conduct_with_repair")
                else [debater.conduct_debate(current_topic)]
            )
            state["debate_results"].extend(attempts)
            result = attempts[-1]
            if result.passed:
                state["hypothesis_passed"] = True
                state["current_phase"] = "planning"
                log_agent_action("Orchestrator", "hypothesis_passed", {"topic": current_topic["title"], "attempts": topics_tried})
                return state
            CrossRunMemory().record_rejection(
                "topic",
                current_topic.get("title", "?"),
                "failed_hypothesis_debate",
                {
                    "score": getattr(result, "score", None),
                    "decision": getattr(result, "moderator_decision", None),
                    "unresolved": list(getattr(result, "unresolved_objections", []) or [])[:5],
                    "structured_hypothesis": current_topic.get("structured_hypothesis"),
                },
            )
            state["topics"] = state["topics"][1:]
            log_agent_action("Orchestrator", "topic_failed", {"topic": current_topic["title"], "topics_remaining": len(state["topics"])})
        state["should_reset"] = True
        state["meta_feedback"].append(f"All {topics_tried} topics failed hypothesis debate")
        log_agent_action("Orchestrator", "all_topics_failed", {"topics_tried": topics_tried})
        return state
    except Exception as exc:
        message = f"Hypothesis debate subsystem crashed: {exc}"
        logger.error(message)
        state["meta_feedback"].append(message)
        state["terminal_error"] = message
        state["technical_failures"] = {
            "hypothesis_debate": {
                "success": False,
                "error": str(exc),
                "failure_kind": "technical",
                "subsystem": "hypothesis_debate",
            }
        }
        state["current_phase"] = "complete"
        state["should_continue"] = False
        state["should_reset"] = False
        log_agent_action(
            "Orchestrator",
            "hypothesis_debate_technical_failure",
            {"error": str(exc)},
        )
        return state


def planning_node(state: ResearchState) -> ResearchState:
    log_agent_action("Orchestrator", "start_planning", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("planning")
    if state.get("terminal_error") or state.get("current_phase") == "complete":
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    try:
        if not state.get("selected_topic"):
            state["should_reset"] = True
            return state
        planner = _create_agent(PlannerAgent)
        plan = None
        if state.get("plan") and state.get("plan_revision_requests"):
            plan = state["plan"]
            for request in state["plan_revision_requests"]:
                plan = planner.revise_plan(plan, request, state["selected_topic"])
            state["plan_revision_requests"] = []
        else:
            plan = planner.create_plan(state["selected_topic"])

        experiments = (plan or {}).get("experiments", [])
        plan_errors = validate_experiments(experiments)
        if plan_errors:
            message = "; ".join(plan_errors)
            attempts = int((plan or {}).get("schema_revision_attempts") or 0)
            if attempts < 2:
                plan = dict(plan or {})
                plan["schema_revision_attempts"] = attempts + 1
                state["plan"] = plan
                revision = {
                    "reason": "invalid_experiment_plan",
                    "experiment": None,
                    "detail": message,
                    "timestamp": datetime.now().isoformat(),
                }
                state.setdefault("plan_revision_requests", []).append(revision)
                CrossRunMemory().record_plan_revision("invalid_experiment_plan", meta=revision)
                state["meta_feedback"].append(f"Plan revision requested for schema errors: {message}")
                state["current_phase"] = "planning"
                state["should_continue"] = True
                log_agent_action("Orchestrator", "planning_schema_error_request_revision", {
                    "message": message,
                    "attempts": attempts + 1,
                })
                return state
            else:
                err_msg = f"Planning failed contract validation after retries: {message}"
                state["terminal_error"] = err_msg
                state["evidence_gate"] = {
                    "allowed": False,
                    "terminal": True,
                    "reason_code": "invalid_experiment_plan",
                    "message": message,
                }
                state["current_phase"] = "complete"
                state["should_continue"] = False
                state["meta_feedback"].append(err_msg)
                log_agent_action("Orchestrator", "planning_terminal_schema_failure", {"message": message})
                return state

        state["plan"] = plan
        state["current_phase"] = "writing_narrative"
        log_agent_action("Orchestrator", "plan_created", {"sections": len(plan.get("sections", []))})
        return state
    except Exception as exc:
        logger.error(f"Planning failed: {exc}")
        err_msg = f"Planning failed: {exc}"
        state["terminal_error"] = state.get("terminal_error") or err_msg
        state["meta_feedback"].append(err_msg)
        state["current_phase"] = "complete"
        state["should_continue"] = False
        log_agent_action("Orchestrator", "planning_exception_terminal_failure", {"error": str(exc)})
        return state


def data_validation_node(state: ResearchState) -> ResearchState:
    """Validate an explicitly requested dataset before experiments begin."""
    log_agent_action("Orchestrator", "start_data_validation", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("data_validation")
    if state.get("terminal_error") or state.get("current_phase") == "complete":
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    if not state.get("plan"):
        err_msg = "Data validation aborted: No valid plan available"
        state["terminal_error"] = state.get("terminal_error") or err_msg
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    try:
        plan = state.get("plan") or {}
        dataset_path = plan.get("dataset_path") or plan.get("dataset_file")
        if not dataset_path:
            state["data_validation"] = {
                "passed": True,
                "score": 10.0,
                "note": "No external dataset requested; experiment may use declared synthetic data",
                "checks": {"dataset_requested": False},
            }
            state["current_phase"] = "writing_narrative"
            return state

        spec = plan.get("dataset_spec") or {
            "name": Path(dataset_path).stem,
            "access_policy": "user-provided",
        }
        artifact = _create_agent(DataAgent).validate_dataset(dataset_path, spec)
        key = artifact.get("spec", {}).get("name") or Path(dataset_path).name
        state["data_artifacts"][key] = artifact
        state["data_validation"] = artifact.get("validation", {})
        if not state["data_validation"].get("passed"):
            state["terminal_error"] = "Dataset validation failed"
            state["meta_feedback"].append(json.dumps(state["data_validation"]))
            state["current_phase"] = "complete"
            state["should_continue"] = False
            return state
        state["current_phase"] = "writing_narrative"
        return state
    except Exception as exc:
        logger.error(f"Data validation failed: {exc}")
        state["terminal_error"] = f"Data validation failed: {exc}"
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
NARRATIVE_SECTION_NAMES = {"abstract", "introduction", "related work", "methods", "method"}
RESULTS_SECTION_NAMES = {"abstract", "results", "discussion", "conclusion", "experiments"}


def _plan_section_names(plan: Dict[str, Any]) -> List[str]:
    return [section.get("name", "Unknown") if isinstance(section, dict) else str(section) for section in plan.get("sections", [])]


def _numeric_values(value: Any) -> List[float]:
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [number for item in value.values() for number in _numeric_values(item)]
    if isinstance(value, (list, tuple)):
        return [number for item in value for number in _numeric_values(item)]
    return []


def verify_result_numbers(content: str, engineer_outputs: Dict[str, Any], rtol: float = 0.005) -> Dict[str, Any]:
    allowed = _numeric_values(engineer_outputs)
    claims = [float(match.group(1)) / (100 if match.group(2) else 1) for match in re.finditer(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*(%)?", content)]
    claims = [claim for claim in claims if not float(claim).is_integer() or f"{int(claim)}%" in content]
    mismatches = [claim for claim in claims if not any(abs(claim - raw) <= max(abs(raw) * rtol, 1e-6) for raw in allowed)]
    return {"passed": bool(allowed) and not mismatches, "claims": claims, "allowed_values": allowed, "mismatches": mismatches}


def section_revision_feedback(
    section_name: str,
    check_result: Dict[str, Any],
    supervisor_feedback: Optional[Dict[str, str]] = None,
    editor_findings: Optional[str] = None,
) -> Optional[str]:
    """Build the revision prompt payload for one failing section (pure helper).

    Combines the section's own numeric/empirical check failures with the
    supervisor's review feedback and, when an editor repair round is active,
    the release-referee findings. Returns None when nothing needs repair.
    """
    parts: List[str] = []
    mismatches = check_result.get("mismatches") or []
    if mismatches:
        parts.append(
            "Untraceable numeric claims — every number must be copied exactly from the "
            "experiment source of truth; rewrite or remove these: "
            + "; ".join(str(item) for item in mismatches[:12])
        )
    empirical = check_result.get("empirical_claims") or {}
    if empirical.get("prohibited_text"):
        parts.append(
            "Remove leaked harness diagnostics entirely: "
            + ", ".join(str(item) for item in empirical["prohibited_text"][:5])
        )
    elif not empirical.get("passed", True):
        parts.append(
            "Empirical claims must be grounded in completed experiment outputs; "
            "remove unsupported performance language."
        )
    prior = (supervisor_feedback or {}).get(section_name)
    if prior:
        parts.append("Prior supervisor feedback on this section: " + str(prior)[:1500])
    if editor_findings:
        parts.append("Editor release-referee findings to fix: " + str(editor_findings)[:1200])
    return "\n".join(parts) if parts else None


def editor_repair_route(error_message: str, repair_count: int, max_repairs: int = 1) -> str:
    """Decide whether an editor failure is repairable or terminal (pure helper).

    Only release-referee failures are repairable by re-drafting sections;
    structural failures (e.g. failed experiments) remain terminal.
    """
    message = str(error_message or "")
    if "release referee failed" not in message:
        return "terminal"
    if int(repair_count) >= max_repairs:
        return "terminal"
    return "repair"


def write_narrative_sections(state: ResearchState) -> ResearchState:
    log_agent_action("Orchestrator", "start_writing_narrative", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("writing_narrative")
    if state.get("terminal_error") or state.get("current_phase") == "complete":
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    if not state.get("plan"):
        err_msg = "Writing narrative aborted: No valid plan available"
        state["terminal_error"] = state.get("terminal_error") or err_msg
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    try:
        writer = _create_agent(WriterAgent)
        supervisor_scores = state.get("supervisor_scores") or {}
        supervisor_feedback = state.get("supervisor_feedback") or {}
        revision_allowed = (
            state["iteration"] > 0
            and state["narrative_revision_count"] < 1
            and bool(supervisor_scores)
        )
        for section_name in _plan_section_names(state["plan"]):
            if section_name.lower() not in NARRATIVE_SECTION_NAMES:
                continue
            if section_name not in state["draft_sections"]:
                state["draft_sections"][section_name] = writer.draft_section(
                    section_name, state["selected_topic"], state["plan"], {}
                )
                state["current_section"] = section_name
                log_agent_action("Orchestrator", "section_written", {"section": section_name})
                continue
            # Meta-continue pass: re-draft below-threshold narrative sections with
            # the supervisor's feedback injected into the prompt so the revision
            # repairs specific defects (reviewer-guided revision >> blind re-roll).
            if not revision_allowed:
                continue
            score = supervisor_scores.get(section_name)
            if score is None or score >= config.supervisor_threshold:
                continue
            revision_feedback = section_revision_feedback(
                section_name,
                {},
                supervisor_feedback=supervisor_feedback,
                editor_findings=state.get("editor_repair_findings"),
            )
            if not revision_feedback:
                continue
            state["draft_sections"][section_name] = writer.draft_section(
                section_name,
                state["selected_topic"],
                state["plan"],
                state.get("engineer_outputs") or {},
                revision_feedback,
            )
            state["current_section"] = section_name
            log_agent_action("Orchestrator", "narrative_section_revised_with_feedback", {
                "section": section_name,
                "prior_score": score,
            })
        if revision_allowed:
            state["narrative_revision_count"] += 1
        state["current_phase"] = "engineering"
        return state
    except Exception as exc:
        logger.error(f"Writing failed: {exc}")
        state["meta_feedback"].append(f"Writing error: {exc}")
        state["error_count"] += 1
        if state["error_count"] >= 3:
            state["should_reset"] = True
        else:
            state["current_phase"] = "engineering"
            state["error_count"] = 0
        return state


def write_results_sections(state: ResearchState) -> ResearchState:
    log_agent_action("Orchestrator", "start_writing_results", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("writing_results")
    if state.get("terminal_error") or state.get("evidence_gate", {}).get("terminal") or state.get("current_phase") == "complete":
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    if not state.get("plan"):
        err_msg = "Writing results aborted: No valid plan available"
        state["terminal_error"] = state.get("terminal_error") or err_msg
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    try:
        writer = _create_agent(WriterAgent)
        names = _plan_section_names(state["plan"])
        for required in ("Results", "Discussion", "Abstract"):
            if not any(name.lower() == required.lower() for name in names):
                names.append(required)
        for section_name in names:
            if section_name.lower() in RESULTS_SECTION_NAMES:
                state["draft_sections"][section_name] = writer.draft_section(section_name, state["selected_topic"], state["plan"], state["engineer_outputs"])
                state["current_section"] = section_name
        checked = {}
        for name in names:
            if name.lower() not in RESULTS_SECTION_NAMES or name not in state["draft_sections"]:
                continue
            section = state["draft_sections"][name]
            numeric_check = verify_result_numbers(section, state["engineer_outputs"])
            empirical_check = validate_empirical_claims(section, state["engineer_outputs"])
            checked[name] = {
                **numeric_check,
                "empirical_claims": empirical_check,
                "passed": numeric_check["passed"] and empirical_check["passed"],
            }
            tracker = get_tracker()
            if tracker:
                context = get_active_context()
                ledger = context.research_db if context and context.research_db else research_db
                ledger.record_claim(
                    tracker.run_id,
                    name,
                    section[:1000],
                    "empirical_section",
                    "verified" if checked[name]["passed"] else "rejected",
                    {
                        "numeric": numeric_check,
                        "empirical": empirical_check,
                        "contract_hashes": {
                            exp_name: output.get("contract_hash")
                            for exp_name, output in state["engineer_outputs"].items()
                            if isinstance(output, dict)
                        },
                    },
                )
        failures = {name: result for name, result in checked.items() if not result["passed"]}
        state["results_verification"] = checked
        hard_failures = {
            name: result for name, result in failures.items()
            if result.get("empirical_claims", {}).get("prohibited_text")
        }
        if hard_failures:
            message = "Manuscript contains leaked execution diagnostics"
            state["terminal_error"] = message
            state["evidence_gate"] = {
                "allowed": False,
                "terminal": True,
                "reason_code": "prohibited_manuscript_text",
                "message": message,
                "sections": list(hard_failures),
            }
            state["current_phase"] = "complete"
            state["should_continue"] = False
            return state
        if failures and state["results_redraft_count"] < 2:
            state["results_redraft_count"] += 1
            # Targeted artifact repair: re-draft only the failing sections and
            # feed their specific check failures back into the writer prompt.
            state["meta_feedback"].append(f"Results numeric grounding failed: {failures}")
            for name, result in failures.items():
                feedback = section_revision_feedback(
                    name,
                    result,
                    supervisor_feedback=state.get("supervisor_feedback") or {},
                    editor_findings=state.get("editor_repair_findings"),
                )
                state["draft_sections"][name] = writer.draft_section(
                    name,
                    state["selected_topic"],
                    state["plan"],
                    state["engineer_outputs"],
                    feedback,
                )
                log_agent_action("Orchestrator", "results_section_revised_with_feedback", {
                    "section": name,
                    "had_feedback": bool(feedback),
                })
            state["current_phase"] = "writing_results"
            log_agent_action("Orchestrator", "results_numeric_grounding_failed", {"sections": list(failures)})
            return state
        state["current_phase"] = "supervision"
        return state
    except Exception as exc:
        logger.error(f"Results writing failed: {exc}")
        state["meta_feedback"].append(f"Results writing error: {exc}")
        state["current_phase"] = "supervision"
        return state


def engineering_node(state: ResearchState) -> ResearchState:
    log_agent_action("Orchestrator", "start_engineering", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("engineering")
    if state.get("terminal_error") or state.get("current_phase") == "complete":
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    if not state.get("plan"):
        err_msg = "Engineering aborted: No valid plan available"
        state["terminal_error"] = state.get("terminal_error") or err_msg
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    try:
        experiments = state["plan"].get("experiments", [])
        if (
            isinstance(experiments, list)
            and experiments
            and all(isinstance(item, dict) for item in experiments)
        ):
            # Defensive contract alignment: LLM aliases like experiment_name → name
            experiments = PlannerAgent._normalize_experiments(experiments)
            state["plan"]["experiments"] = experiments
        plan_errors = validate_experiments(experiments)
        plan_errors.extend(validate_dataset_identity(experiments, state.get("data_artifacts")))
        if plan_errors:
            message = "; ".join(plan_errors)
            attempts = int((state.get("plan") or {}).get("schema_revision_attempts") or 0)
            if attempts >= 2:
                state["terminal_error"] = message
                state["evidence_gate"] = {
                    "allowed": False,
                    "terminal": True,
                    "reason_code": "invalid_experiment_plan",
                    "message": message,
                }
                state["current_phase"] = "complete"
                state["should_continue"] = False
                log_agent_action("Orchestrator", "engineering_blocked_invalid_plan", {
                    "message": message,
                    "schema_revision_attempts": attempts,
                })
                return state
            plan = dict(state["plan"] or {})
            plan["schema_revision_attempts"] = attempts + 1
            state["plan"] = plan
            revision = {
                "reason": "invalid_experiment_plan",
                "experiment": None,
                "detail": message,
                "timestamp": datetime.now().isoformat(),
            }
            state.setdefault("plan_revision_requests", []).append(revision)
            CrossRunMemory().record_plan_revision("invalid_experiment_plan", meta=revision)
            state["meta_feedback"].append(f"Plan revision requested for schema errors: {message}")
            state["current_phase"] = "planning"
            state["should_continue"] = True
            state["terminal_error"] = None
            log_agent_action("Orchestrator", "engineering_blocked_invalid_plan", {
                "message": message,
                "action": "request_plan_revision",
                "schema_revision_attempts": plan["schema_revision_attempts"],
            })
            return state
        for experiment in experiments:
            name = experiment["name"]
            contract = build_contract(experiment)
            prior_contract = state["experiment_contracts"].get(name)
            if prior_contract and prior_contract.get("contract_hash") != contract.get("contract_hash"):
                message = f"Committed experiment contract changed: {name}"
                state["terminal_error"] = message
                state["evidence_gate"] = {
                    "allowed": False,
                    "terminal": True,
                    "reason_code": "experiment_contract_drift",
                    "message": message,
                }
                state["current_phase"] = "complete"
                state["should_continue"] = False
                log_agent_action("Orchestrator", "engineering_blocked_contract_drift", {"experiment": name})
                return state
            state["experiment_contracts"][name] = prior_contract or contract
        engineer = _create_agent(EngineerAgent)
        method_text = "\n".join(state["draft_sections"].get(section, "") for section in ("Methods", "Method", "Experiments"))
        branch_winner_name = None
        if experiments and any(isinstance(item, dict) and (item.get("variants") or item.get("alternatives")) for item in experiments):
            try:
                if tracker:
                    tracker.message("Engineering: branching cheap probes…")
                branched = engineer.run_branching_search([item for item in experiments if isinstance(item, dict)], method_description=method_text)
                branch_winner_name = branched.get("experiment_name") or branched.get("approach") or "branch_winner"
                if branch_winner_name in state["experiment_contracts"]:
                    branched["contract_hash"] = state["experiment_contracts"][branch_winner_name]["contract_hash"]
                state["engineer_outputs"][branch_winner_name] = branched
                winner = (branched.get("branch_search") or {}).get("winner")
                if winner and winner not in state["engineer_outputs"]:
                    state["engineer_outputs"][winner] = branched
            except Exception as exc:
                logger.error(f"Branching search failed: {exc}")
                if tracker:
                    tracker.message(f"Branching search failed: {exc}", level="error")
        for experiment in experiments:
            exp_name = experiment.get("name", "unknown_experiment") if isinstance(experiment, dict) else str(experiment)
            exp_config = dict(experiment) if isinstance(experiment, dict) else {"name": exp_name}
            if isinstance(experiment, dict) and exp_name in state["experiment_contracts"]:
                exp_config["contract_hash"] = state["experiment_contracts"][exp_name]["contract_hash"]
            if exp_name in state["engineer_outputs"] or (branch_winner_name and exp_name == branch_winner_name):
                continue
            try:
                if tracker:
                    tracker.message(f"Engineering: running {exp_name}")
                alternatives = list(exp_config.get("variants") or exp_config.get("alternatives") or [])
                state["engineer_outputs"][exp_name] = engineer.run_experiment(exp_config, alternatives=alternatives, method_description=method_text)
                state["engineer_outputs"][exp_name]["contract_hash"] = state["experiment_contracts"][exp_name]["contract_hash"]
                log_agent_action("Orchestrator", "experiment_run", {"experiment": exp_name})
            except Exception as exc:
                logger.error(f"Experiment {exp_name} failed: {exc}")
                state["engineer_outputs"][exp_name] = {
                    "success": False,
                    "error": str(exc),
                    "experiment_name": exp_name,
                    "contract_hash": state["experiment_contracts"][exp_name]["contract_hash"],
                    "failure_kind": "technical",
                }
        gate = gate_engineering_outputs(
            state.get("plan"),
            state.get("engineer_outputs"),
            state.get("experiment_contracts"),
        )
        state["evidence_gate"] = gate
        requests = engineer.consume_plan_revision_requests()
        if not gate.get("allowed"):
            state["technical_failures"] = {
                name: output
                for name, output in state.get("engineer_outputs", {}).items()
                if isinstance(output, dict) and not output.get("success")
            }
            revision_attempts = int((state.get("plan") or {}).get("engineer_revision_attempts") or 0)
            if requests and revision_attempts < 2:
                plan = dict(state.get("plan") or {})
                plan["engineer_revision_attempts"] = revision_attempts + 1
                state["plan"] = plan
                state.setdefault("plan_revision_requests", []).extend(requests)
                # Revised plan gets a fresh contract identity; clear prior commit.
                state["engineer_outputs"] = {}
                state["experiment_contracts"] = {}
                state["terminal_error"] = None
                state["evidence_gate"] = {
                    "allowed": False,
                    "terminal": False,
                    "reason_code": "plan_revision_requested",
                    "message": "Engineering exhausted attempts; requesting Planner revision before terminal failure",
                }
                state["meta_feedback"].append(
                    "Plan revision requested after engineering failures: "
                    + json.dumps(requests, default=str)
                )
                state["current_phase"] = "planning"
                state["should_continue"] = True
                log_agent_action("Orchestrator", "engineering_request_plan_revision", {
                    "gate": gate,
                    "requests": requests,
                    "engineer_revision_attempts": plan["engineer_revision_attempts"],
                })
                return state
            state["terminal_error"] = gate.get("message") or gate.get("reason_code")
            state["current_phase"] = "complete"
            state["should_continue"] = False
            if requests:
                state["meta_feedback"].append(
                    "Plan revision requests discarded at terminal engineering failure: "
                    + json.dumps(requests, default=str)
                )
            log_agent_action("Orchestrator", "engineering_terminal_failure", gate)
            return state
        if requests:
            state["meta_feedback"].append(
                "Plan revision requested after a successful run; contract remains immutable: "
                + json.dumps(requests, default=str)
            )
        state["current_phase"] = "writing_results"
        return state
    except Exception as exc:
        logger.error(f"Engineering failed: {exc}")
        state["meta_feedback"].append(f"Engineering error: {exc}")
        state["error_count"] += 1
        if state["error_count"] >= 3:
            state["should_reset"] = True
        else:
            state["current_phase"] = "writing_results"
            state["error_count"] = 0
        return state


def independent_validation_node(state: ResearchState) -> ResearchState:
    """Replay engineer code, analyze outputs, and record independent findings."""
    log_agent_action("Orchestrator", "start_independent_validation", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("independent_validation")
    if state.get("terminal_error") or state.get("current_phase") == "complete":
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    if not state.get("plan"):
        err_msg = "Independent validation aborted: No valid plan available"
        state["terminal_error"] = state.get("terminal_error") or err_msg
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    try:
        code_artifacts = {
            name: {
                "experiment_name": name,
                "source_code": output.get("code", ""),
                "language": "python",
            }
            for name, output in state.get("engineer_outputs", {}).items()
            if output.get("success") and output.get("code")
        }
        if not code_artifacts:
            message = "Independent validation failed: no executable code artifacts"
            state["terminal_error"] = state.get("terminal_error") or message
            state["evidence_gate"] = {
                "allowed": False,
                "terminal": True,
                "reason_code": "missing_code_artifact",
                "message": message,
            }
            state["meta_feedback"].append(message)
            state["current_phase"] = "complete"
            state["should_continue"] = False
            return state

        execution_agent = _create_agent(ExecutionAgent)
        for name, code_artifact in code_artifacts.items():
            experiment = next(
                (item for item in (state.get("plan") or {}).get("experiments", [])
                 if isinstance(item, dict) and item.get("name") == name),
                {},
            )
            seeds = [42 + index * 1009 for index in range(config.experiment_seeds)]
            state["execution_artifacts"][name] = execution_agent.execute(
                code_artifact,
                {
                    "experiment_name": name,
                    "seeds": seeds,
                    "resource_limits": {"timeout_seconds": config.sandbox_timeout_sec},
                },
            )

        analysis_plan = (state.get("plan") or {}).get("analysis_plan") or {
            "primary_metric": ((state.get("plan") or {}).get("experiments") or [{}])[0].get(
                "evaluation_metrics", ["accuracy"]
            )[0],
            "statistical_test": "Welch t-test",
            "confidence_level": 0.95,
        }
        report = _create_agent(AnalysisAgent).analyze(state["execution_artifacts"], analysis_plan)
        state["analysis_reports"]["independent"] = report
        reports_by_experiment = {
            name: {"metrics": report.get("metrics", {}).get(name, {})}
            for name in state["execution_artifacts"]
        }
        state["verification_findings"] = _create_agent(VerificationAgent).verify(
            state["execution_artifacts"],
            reports_by_experiment,
        )
        if any(finding.get("blocking") for finding in state["verification_findings"]):
            message = "Independent validation produced blocking findings"
            state["terminal_error"] = state.get("terminal_error") or message
            state["evidence_gate"] = {
                "allowed": False,
                "terminal": True,
                "reason_code": "blocking_verification_finding",
                "message": message,
            }
            state["meta_feedback"].append(message)
            state["current_phase"] = "complete"
            state["should_continue"] = False
            return state
        state["current_phase"] = "writing_results"
        return state
    except Exception as exc:
        logger.error(f"Independent validation failed: {exc}")
        state["verification_findings"].append({
            "finding_id": "independent_validation:error",
            "severity": "error",
            "check": "independent_validation",
            "message": str(exc),
            "blocking": True,
            "status": "failed",
        })
        state["meta_feedback"].append(f"Independent validation error: {exc}")
        state["terminal_error"] = state.get("terminal_error") or f"Independent validation failed: {exc}"
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
def supervision_node(state: ResearchState) -> ResearchState:
    log_agent_action("Orchestrator", "start_supervision", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("supervision")
    try:
        supervisor = _create_agent(SupervisorAgent)
        section_requirements = {
            str(section.get("name") or "").lower(): section.get("content_requirements")
            for section in ((state.get("plan") or {}).get("sections") or [])
            if isinstance(section, dict)
        }
        for section_name, content in state["draft_sections"].items():
            score, feedback = supervisor.evaluate_section(
                section_name,
                content,
                engineer_outputs=state.get("engineer_outputs"),
                content_requirements=section_requirements.get(section_name.lower()),
            )
            state["supervisor_scores"][section_name] = score
            state["supervisor_feedback"][section_name] = feedback
            if score < config.supervisor_threshold and get_tracker():
                get_tracker().bump("sections_bounced")
        state["reproducibility"] = reproducibility_dossier(state.get("plan"), state.get("engineer_outputs"))
        state["outcome_calibration"] = research_db.outcome_calibration()
        if not state["reproducibility"]["passed"]:
            state["meta_feedback"].append("Reproducibility dossier incomplete: " + json.dumps(state["reproducibility"]["checks"]))
        has_blocking_findings = any(
            finding.get("blocking") for finding in state.get("verification_findings", [])
        )
        if has_blocking_findings:
            state["current_phase"] = "meta_evaluation"
            log_agent_action("Orchestrator", "release_blocked_by_verification", {
                "findings": len(state.get("verification_findings", [])),
            })
        elif state["supervisor_scores"]:
            overall_score = sum(state["supervisor_scores"].values()) / len(state["supervisor_scores"])
            state["current_phase"] = "editing" if overall_score >= config.supervisor_threshold else "meta_evaluation"
            log_agent_action("Orchestrator", "quality_threshold_met" if overall_score >= config.supervisor_threshold else "quality_below_threshold", {"score": overall_score})
        else:
            state["current_phase"] = "meta_evaluation"
        return state
    except Exception as exc:
        logger.error(f"Supervision failed: {exc}")
        state["meta_feedback"].append(f"Supervision error: {exc}")
        state["current_phase"] = "meta_evaluation"
        return state


def meta_evaluation_node(state: ResearchState) -> ResearchState:
    log_agent_action("Orchestrator", "start_meta_evaluation", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("meta_evaluation")
    if state.get("terminal_error") or state.get("evidence_gate", {}).get("terminal"):
        state["should_continue"] = False
        state["current_phase"] = "complete"
        log_agent_action("Orchestrator", "meta_blocked_terminal_failure", {})
        return state
    try:
        agent = _create_agent(MetaAgent)
        state["meta_feedback"].append(agent.evaluate_system_performance(state))
        if agent.should_reset(state):
            state["should_reset"] = True
            log_agent_action("Orchestrator", "meta_reset_triggered", {"iteration": state["iteration"]})
        elif agent.should_continue(state):
            state["should_continue"] = True
            state["iteration"] += 1
            state["current_phase"] = "writing_narrative"
            log_agent_action("Orchestrator", "meta_continue_triggered", {"iteration": state["iteration"]})
        else:
            state["should_continue"] = False
            log_agent_action("Orchestrator", "meta_stop_triggered", {"iteration": state["iteration"]})
        return state
    except Exception as exc:
        logger.error(f"Meta evaluation failed: {exc}")
        state["meta_feedback"].append(f"Meta evaluation error: {exc}")
        state["terminal_error"] = f"Meta evaluation failed: {exc}"
        state["should_continue"] = False
        state["current_phase"] = "complete"
        log_agent_action("Orchestrator", "meta_error_stop", {})
        return state


def editing_node(state: ResearchState) -> ResearchState:
    log_agent_action("Orchestrator", "start_editing", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("editing")
    if state.get("terminal_error") or state.get("evidence_gate", {}).get("terminal"):
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    try:
        editor = _create_agent(EditorAgent)
        final_paper = editor.create_final_paper(state["selected_topic"], state["draft_sections"], state["plan"], state["engineer_outputs"], debate_results=state.get("debate_results"))
        state["final_paper"] = final_paper
        state["latex_output"] = editor.generate_latex(final_paper)
        state["current_phase"] = "complete"
        log_agent_action("Orchestrator", "editing_complete", {})
        return state
    except RuntimeError as exc:
        # WARA-style artifact repair: a release-referee failure routes the
        # affected sections back for one bounded feedback-aware repair round
        # instead of discarding the whole run.
        route = editor_repair_route(str(exc), state.get("editor_repair_count", 0))
        if route == "repair":
            state["editor_repair_count"] = int(state.get("editor_repair_count", 0)) + 1
            state["editor_repair_findings"] = str(exc)[:4000]
            state["results_redraft_count"] = min(state.get("results_redraft_count", 0), 1)
            state.setdefault("supervisor_feedback", {})["__editor_findings__"] = str(exc)[:4000]
            state["current_phase"] = "writing_results"
            state["meta_feedback"].append(
                "Editor release referee failed; routing sections for one repair pass: "
                + str(exc)[:1500]
            )
            log_agent_action("Orchestrator", "editor_referee_repair_routed", {
                "repair_count": state["editor_repair_count"],
            })
            return state
        message = f"Editing failed terminally: {exc}"
        logger.error(message)
        state["meta_feedback"].append(message)
        state["terminal_error"] = state.get("terminal_error") or message
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    except Exception as exc:
        logger.error(f"Editing failed: {exc}")
        state["meta_feedback"].append(f"Editing error: {exc}")
        state["current_phase"] = "complete"
        return state


# ---------------------------------------------------------------------------
# QA mode nodes
# ---------------------------------------------------------------------------


def qa_literature_retrieval_node(state: ResearchState) -> ResearchState:
    """Retrieve literature for the user query using TopicHunter's shared retrieval."""
    log_agent_action("Orchestrator", "qa_literature_retrieval", {"query": state.get("user_query")})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("qa_literature_retrieval")
    query = state.get("user_query") or ""
    if not query:
        state["terminal_error"] = "QA mode requires a user_query"
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    try:
        hunter = _create_agent(TopicHunterAgent)
        lit = hunter.retrieve_literature(query)
        if not lit.get("papers"):
            state["terminal_error"] = "No literature found for the query"
            state["current_phase"] = "complete"
            state["should_continue"] = False
            return state
        state["literature_context"] = lit
        state["current_phase"] = "qa_answer"
        log_agent_action("Orchestrator", "qa_literature_retrieved", {"paper_count": len(lit.get("papers", []))})
        return state
    except ResearchSourceUnavailable as exc:
        state["terminal_error"] = str(exc)
        state["current_phase"] = "complete"
        state["should_continue"] = False
        log_agent_action("Orchestrator", "qa_research_sources_unavailable", {"message": str(exc)})
        return state
    except Exception as exc:
        logger.error(f"QA literature retrieval failed: {exc}")
        state["terminal_error"] = f"QA literature retrieval failed: {exc}"
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state


def qa_answer_node(state: ResearchState) -> ResearchState:
    """Produce a citation-backed synthesis answer from retrieved literature."""
    log_agent_action("Orchestrator", "qa_answer", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("qa_answer")
    if state.get("terminal_error"):
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    lit = state.get("literature_context") or {}
    papers = lit.get("papers", [])
    query = state.get("user_query") or ""
    if not papers or not query:
        state["terminal_error"] = "No literature or query available for QA answer"
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    try:
        paper_summaries = []
        for p in papers[:15]:
            paper_summaries.append({
                "title": p.get("title", ""),
                "abstract": (p.get("abstract") or "")[:1500],
                "doi": p.get("doi"),
                "arxiv_id": p.get("arxiv_id"),
                "year": p.get("year"),
                "cited_by_count": p.get("cited_by_count"),
            })
        prompt = f"""You are a research synthesis assistant. Answer the following question using ONLY the provided literature.
For every factual claim, cite the source paper using its DOI or arXiv ID in the format: doi:10.XXXX/... or arXiv:XXXX.XXXXX.
If you cannot answer from the provided literature, say so explicitly.

Question: {query}

Relevant Literature:
{json.dumps(paper_summaries, indent=2)[:12000]}

Provide a structured answer with:
1. A concise summary (2-4 paragraphs)
2. Key findings from the literature
3. A bibliography listing each cited paper with its DOI/arXiv ID
4. An limitations section noting what the literature does not cover

Return JSON:
{{
  "answer": "synthesis text with inline citations",
  "key_findings": ["finding 1", "finding 2"],
  "bibliography": [{{"title": "...", "doi": "...", "arxiv_id": "...", "year": 2024}}],
  "limitations": "what the literature does not cover"
}}"""
        raw = call_llm_json(
            prompt,
            temperature=0.3,
            tier="strong",
            attempts=2,
            call_fn=call_llm,
        ) or {}
        parsed = raw if isinstance(raw, dict) else {}
        answer_text = parsed.get("answer", "")
        if not answer_text:
            state["terminal_error"] = "QA answer generation produced empty output"
            state["current_phase"] = "complete"
            state["should_continue"] = False
            return state
        citation_verification = verify_citations(answer_text)
        state["qa_answer"] = {
            "answer": answer_text,
            "key_findings": parsed.get("key_findings", []),
            "bibliography": parsed.get("bibliography", []),
            "limitations": parsed.get("limitations", ""),
            "query": query,
        }
        state["qa_citation_verification"] = citation_verification
        if not citation_verification.get("passed"):
            state["meta_feedback"].append(
                f"QA citation verification failed: {citation_verification.get('note', 'unknown')}"
            )
        state["current_phase"] = "complete"
        state["should_continue"] = False
        log_agent_action("Orchestrator", "qa_answer_complete", {
            "citations_passed": citation_verification.get("passed"),
            "citations_score": citation_verification.get("score"),
        })
        return state
    except Exception as exc:
        logger.error(f"QA answer generation failed: {exc}")
        state["terminal_error"] = f"QA answer generation failed: {exc}"
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state


def qa_verification_node(state: ResearchState) -> ResearchState:
    """Verify QA answer quality, citation integrity, and finalize the run."""
    log_agent_action("Orchestrator", "qa_verification", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("qa_verification")
    if state.get("terminal_error"):
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    qa_answer = state.get("qa_answer")
    if not qa_answer:
        state["terminal_error"] = "QA verification failed: no answer to verify"
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state
    try:
        answer_text = qa_answer.get("answer", "")
        bibliography = qa_answer.get("bibliography", [])
        key_findings = qa_answer.get("key_findings", [])
        citation_verification = state.get("qa_citation_verification", {})
        issues = []
        if not answer_text.strip():
            issues.append("Empty answer text")
        if len(key_findings) == 0:
            issues.append("No key findings extracted")
        if len(bibliography) == 0:
            issues.append("No bibliography entries")
        if not citation_verification.get("passed"):
            score = citation_verification.get("score", 0)
            failed_count = len(citation_verification.get("failed", []))
            issues.append(f"Citation verification issues: score={score:.1f}, {failed_count} unresolved")
        if issues:
            state["meta_feedback"].append(f"QA verification notes: {'; '.join(issues)}")
        else:
            state["meta_feedback"].append("QA verification passed: answer, findings, bibliography, and citations all valid")
        state["current_phase"] = "complete"
        state["should_continue"] = False
        log_agent_action("Orchestrator", "qa_verification_complete", {
            "issues": len(issues),
            "citations_passed": citation_verification.get("passed"),
            "findings_count": len(key_findings),
            "bibliography_count": len(bibliography),
        })
        return state
    except Exception as exc:
        logger.error(f"QA verification failed: {exc}")
        state["meta_feedback"].append(f"QA verification error: {exc}")
        state["current_phase"] = "complete"
        state["should_continue"] = False
        return state


def reset_node(state: ResearchState) -> ResearchState:
    current_iteration = state["iteration"]
    log_agent_action("Orchestrator", "system_reset", {"iteration": current_iteration})
    if current_iteration >= config.max_iterations:
        state["current_phase"] = "complete"
        state["meta_feedback"].append(f"Maximum iterations ({config.max_iterations}) reached - stopping")
        return state
    state = initialize_state()
    state["iteration"] = current_iteration + 1
    state["current_phase"] = "topic_discovery"
    log_agent_action("Orchestrator", "reset_complete", {"new_iteration": state["iteration"]})
    return state


def is_valid_plan(state: ResearchState) -> bool:
    """Guard: return True only when the plan is present and experiment specs
    satisfy the contract schema required by downstream nodes.

    This catches malformed variants (e.g. dicts missing ``name``) that
    ``validate_experiments`` in evidence_gate passes because it only checks
    ``isinstance(variant, dict)``.
    """
    plan = state.get("plan")
    if not plan or not isinstance(plan, dict):
        return False
    experiments = plan.get("experiments")
    if not isinstance(experiments, list) or not experiments:
        return False
    for exp in experiments:
        if not isinstance(exp, dict):
            return False
        name = exp.get("name")
        if not isinstance(name, str) or not name.strip():
            return False
        if not exp.get("evaluation_metrics"):
            return False
        for variant in exp.get("variants") or exp.get("alternatives") or []:
            if not isinstance(variant, dict):
                return False
            if not (isinstance(variant.get("name"), str) and variant["name"].strip()):
                return False
    return True


def terminal_planning_failure(state: ResearchState) -> ResearchState:
    """Terminal node: attribute the failure to PlannerAgent and halt."""
    errors = []
    plan = state.get("plan") or {}
    experiments = plan.get("experiments") or []
    for exp in experiments:
        if not isinstance(exp, dict):
            errors.append(f"experiment is not an object: {type(exp).__name__}")
            continue
        name = exp.get("name")
        if not isinstance(name, str) or not name.strip():
            errors.append(f"experiment missing required 'name' field")
        for idx, variant in enumerate(exp.get("variants") or exp.get("alternatives") or []):
            if not isinstance(variant, dict):
                errors.append(f"{name or 'unnamed'} variant[{idx}] is not an object")
            elif not (isinstance(variant.get("name"), str) and variant["name"].strip()):
                errors.append(f"{name or 'unnamed'} variant[{idx}] missing required 'name' field")
    message = "PlannerAgent produced invalid plan: " + "; ".join(errors) if errors else "PlannerAgent produced invalid plan"
    state["terminal_error"] = state.get("terminal_error") or message
    state["evidence_gate"] = {
        "allowed": False,
        "terminal": True,
        "reason_code": "invalid_experiment_plan",
        "message": message,
    }
    state["current_phase"] = "complete"
    state["should_continue"] = False
    state["meta_feedback"].append(message)
    state["technical_failures"] = {
        "planning": {
            "success": False,
            "failure_kind": "invalid_plan_schema",
            "reason_code": "invalid_experiment_plan",
            "message": message,
        }
    }
    log_agent_action("Orchestrator", "terminal_planning_failure", {"message": message})
    return state


def should_reset(state: ResearchState) -> str:
    if state.get("current_phase") == "complete" or state.get("terminal_error") or not state.get("should_continue", True):
        return "end"
    return "reset" if state.get("should_reset") else "continue"


def should_continue(state: ResearchState) -> str:
    if state.get("current_phase") == "complete" or state.get("terminal_error") or not state.get("should_continue", True):
        return "end"
    return "continue" if state.get("should_continue") else "end"
