"""Phase node implementations for the ScholarGraph research workflow."""

import json
import logging
import re
from typing import Any, Dict, List

from core.config import config
from core.context import get_active_context
from core.run_log import get_tracker
from core.state import ResearchState, initialize_state
from core.utils import log_agent_action
from core.verification import reproducibility_dossier

from agents.editor import EditorAgent
from agents.engineer import EngineerAgent
from agents.hypothesis_debate import HypothesisDebateSystem
from agents.meta_agent import MetaAgent
from agents.planner import PlannerAgent
from agents.supervisor import SupervisorAgent
from agents.topic_hunter import ResearchSourceUnavailable, TopicHunterAgent
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
        elif state["iteration"] >= 3:
            state["current_phase"] = "complete"
            state["meta_feedback"].append("No topics discovered after multiple attempts - stopping")
            log_agent_action("Orchestrator", "no_topics_found_after_retries", {"iteration": state["iteration"]})
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
        if state["iteration"] >= 3:
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
    try:
        if not state["topics"]:
            state["should_reset"] = True
            return state
        topics_tried = 0
        while state["topics"]:
            current_topic = state["topics"][0]
            state["selected_topic"] = current_topic
            topics_tried += 1
            log_agent_action("Orchestrator", "trying_topic", {"topic": current_topic["title"], "attempt": topics_tried, "topics_remaining": len(state["topics"])})
            result = _create_agent(HypothesisDebateSystem).conduct_debate(current_topic)
            state["debate_results"].append(result)
            if result.passed:
                state["hypothesis_passed"] = True
                state["current_phase"] = "planning"
                log_agent_action("Orchestrator", "hypothesis_passed", {"topic": current_topic["title"], "attempts": topics_tried})
                return state
            state["topics"] = state["topics"][1:]
            log_agent_action("Orchestrator", "topic_failed", {"topic": current_topic["title"], "topics_remaining": len(state["topics"])})
        state["should_reset"] = True
        state["meta_feedback"].append(f"All {topics_tried} topics failed hypothesis debate")
        log_agent_action("Orchestrator", "all_topics_failed", {"topics_tried": topics_tried})
        return state
    except Exception as exc:
        logger.error(f"Hypothesis debate failed: {exc}")
        state["meta_feedback"].append(f"Hypothesis debate error: {exc}")
        if len(state["topics"]) > 1:
            state["topics"] = state["topics"][1:]
            log_agent_action("Orchestrator", "trying_next_topic_after_error", {"remaining_topics": len(state["topics"])})
        else:
            state["should_reset"] = True
        return state


def planning_node(state: ResearchState) -> ResearchState:
    log_agent_action("Orchestrator", "start_planning", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("planning")
    try:
        if not state["selected_topic"]:
            state["should_reset"] = True
            return state
        planner = _create_agent(PlannerAgent)
        if state.get("plan") and state.get("plan_revision_requests"):
            plan = state["plan"]
            for request in state["plan_revision_requests"]:
                plan = planner.revise_plan(plan, request, state["selected_topic"])
            state["plan_revision_requests"] = []
        else:
            plan = planner.create_plan(state["selected_topic"])
        state["plan"] = plan
        state["current_phase"] = "writing_narrative"
        log_agent_action("Orchestrator", "plan_created", {"sections": len(plan.get("sections", []))})
        return state
    except Exception as exc:
        logger.error(f"Planning failed: {exc}")
        state["should_reset"] = True
        state["meta_feedback"].append(f"Planning error: {exc}")
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


def write_narrative_sections(state: ResearchState) -> ResearchState:
    log_agent_action("Orchestrator", "start_writing_narrative", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("writing_narrative")
    try:
        if not state["plan"]:
            state["should_reset"] = True
            return state
        writer = _create_agent(WriterAgent)
        for section_name in _plan_section_names(state["plan"]):
            if section_name.lower() in NARRATIVE_SECTION_NAMES and section_name not in state["draft_sections"]:
                state["draft_sections"][section_name] = writer.draft_section(section_name, state["selected_topic"], state["plan"], {})
                state["current_section"] = section_name
                log_agent_action("Orchestrator", "section_written", {"section": section_name})
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
    if not state.get("plan"):
        state["should_reset"] = True
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
        checked = {name: verify_result_numbers(state["draft_sections"][name], state["engineer_outputs"]) for name in names if name.lower() in RESULTS_SECTION_NAMES and name in state["draft_sections"]}
        failures = {name: result for name, result in checked.items() if not result["passed"]}
        state["results_verification"] = checked
        if failures and state["results_redraft_count"] < 2:
            state["results_redraft_count"] += 1
            state["meta_feedback"].append(f"Results numeric grounding failed: {failures}")
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
    try:
        if not state["plan"]:
            state["current_phase"] = "supervision"
            return state
        engineer = _create_agent(EngineerAgent)
        experiments = state["plan"].get("experiments", [])
        method_text = "\n".join(state["draft_sections"].get(section, "") for section in ("Methods", "Method", "Experiments"))
        branch_winner_name = None
        if experiments and any(isinstance(item, dict) and (item.get("variants") or item.get("alternatives")) for item in experiments):
            try:
                if tracker:
                    tracker.message("Engineering: branching cheap probes…")
                branched = engineer.run_branching_search([item for item in experiments if isinstance(item, dict)], method_description=method_text)
                branch_winner_name = branched.get("experiment_name") or branched.get("approach") or "branch_winner"
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
            exp_config = experiment if isinstance(experiment, dict) else {"name": exp_name}
            if exp_name in state["engineer_outputs"] or (branch_winner_name and exp_name == branch_winner_name):
                continue
            try:
                if tracker:
                    tracker.message(f"Engineering: running {exp_name}")
                alternatives = list(exp_config.get("variants") or exp_config.get("alternatives") or [])
                state["engineer_outputs"][exp_name] = engineer.run_experiment(exp_config, alternatives=alternatives, method_description=method_text)
                log_agent_action("Orchestrator", "experiment_run", {"experiment": exp_name})
            except Exception as exc:
                logger.error(f"Experiment {exp_name} failed: {exc}")
                state["engineer_outputs"][exp_name] = {"success": False, "error": str(exc), "experiment_name": exp_name}
        requests = engineer.consume_plan_revision_requests()
        if requests:
            state["plan_revision_requests"] = requests
            state["current_phase"] = "planning"
            state["engineer_outputs"] = {f"prior_{key}": value for key, value in state.get("engineer_outputs", {}).items()}
            log_agent_action("Orchestrator", "plan_revision_requested", {"n": len(requests)})
            return state
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


def supervision_node(state: ResearchState) -> ResearchState:
    log_agent_action("Orchestrator", "start_supervision", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("supervision")
    try:
        supervisor = _create_agent(SupervisorAgent)
        for section_name, content in state["draft_sections"].items():
            score, feedback = supervisor.evaluate_section(section_name, content, engineer_outputs=state.get("engineer_outputs"))
            state["supervisor_scores"][section_name] = score
            state["supervisor_feedback"][section_name] = feedback
            if score < config.supervisor_threshold and get_tracker():
                get_tracker().bump("sections_bounced")
        state["reproducibility"] = reproducibility_dossier(state.get("plan"), state.get("engineer_outputs"))
        if not state["reproducibility"]["passed"]:
            state["meta_feedback"].append("Reproducibility dossier incomplete: " + json.dumps(state["reproducibility"]["checks"]))
        if state["supervisor_scores"]:
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
        state["iteration"] += 1
        state["should_continue"] = True
        state["current_phase"] = "writing_narrative"
        log_agent_action("Orchestrator", "meta_error_continue", {"iteration": state["iteration"]})
        return state


def editing_node(state: ResearchState) -> ResearchState:
    log_agent_action("Orchestrator", "start_editing", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("editing")
    try:
        editor = _create_agent(EditorAgent)
        final_paper = editor.create_final_paper(state["selected_topic"], state["draft_sections"], state["plan"], state["engineer_outputs"], debate_results=state.get("debate_results"))
        state["final_paper"] = final_paper
        state["latex_output"] = editor.generate_latex(final_paper)
        state["current_phase"] = "complete"
        log_agent_action("Orchestrator", "editing_complete", {})
        return state
    except Exception as exc:
        logger.error(f"Editing failed: {exc}")
        state["meta_feedback"].append(f"Editing error: {exc}")
        state["current_phase"] = "complete"
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


def should_reset(state: ResearchState) -> str:
    if state["current_phase"] == "complete":
        return "end"
    return "reset" if state["should_reset"] else "continue"


def should_continue(state: ResearchState) -> str:
    if state["current_phase"] == "complete":
        return "end"
    return "continue" if state["should_continue"] else "end"
