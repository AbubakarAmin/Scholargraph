"""
Main orchestration module for the multi-agent research system.
Uses LangGraph to coordinate all agents and manage the research workflow.
"""

import os
import sys
import json
import logging
import argparse
import re
from pathlib import Path
from typing import TypedDict, Dict, Any, List, Optional
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from langgraph.graph import StateGraph, END

# Import core modules
from core.config import config, validate_config
from core.memory import memory
from core.utils import log_agent_action
from core.run_log import start_run, get_tracker
from core.verification import reproducibility_dossier

# Import agents
from agents.meta_agent import MetaAgent
from agents.topic_hunter import TopicHunterAgent, ResearchSourceUnavailable
from agents.hypothesis_debate import HypothesisDebateSystem
from agents.planner import PlannerAgent
from agents.writer import WriterAgent
from agents.supervisor import SupervisorAgent
from agents.engineer import EngineerAgent
from agents.editor import EditorAgent

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class ResearchState(TypedDict):
    """State definition for the research workflow."""
    # System state
    iteration: int
    current_phase: str
    should_reset: bool
    should_continue: bool
    error_count: int  # Track consecutive errors
    
    # Topic discovery
    topics: List[Dict[str, Any]]
    selected_topic: Optional[Dict[str, Any]]
    
    # Hypothesis debate
    debate_results: List[Dict[str, Any]]
    hypothesis_passed: bool
    
    # Planning
    plan: Optional[Dict[str, Any]]
    
    # Writing
    draft_sections: Dict[str, str]
    current_section: Optional[str]
    
    # Engineering
    engineer_outputs: Dict[str, Any]
    
    # Supervision
    supervisor_scores: Dict[str, float]
    supervisor_feedback: Dict[str, str]
    
    # Meta feedback
    meta_feedback: List[str]
    
    # Final output
    final_paper: Optional[Dict[str, Any]]
    latex_output: Optional[str]

    # Bidirectional plan revision + observability
    plan_revision_requests: List[Dict[str, Any]]
    run_id: Optional[str]
    results_redraft_count: int
    results_verification: Dict[str, Any]
    reproducibility: Dict[str, Any]
    terminal_error: Optional[str]

def initialize_state() -> ResearchState:
    """Initialize the research state."""
    return ResearchState(
        iteration=0,
        current_phase="topic_discovery",
        should_reset=False,
        should_continue=True,
        topics=[],
        selected_topic=None,
        debate_results=[],
        hypothesis_passed=False,
        plan=None,
        draft_sections={},
        current_section=None,
        engineer_outputs={},
        supervisor_scores={},
        supervisor_feedback={},
        meta_feedback=[],
        final_paper=None,
        latex_output=None,
        error_count=0,
        plan_revision_requests=[],
        run_id=None,
        results_redraft_count=0,
        results_verification={},
        reproducibility={},
        terminal_error=None,
    )

def topic_discovery_node(state: ResearchState) -> ResearchState:
    """Discover research topics using TopicHunterAgent."""
    log_agent_action("Orchestrator", "start_topic_discovery", {"iteration": state["iteration"]})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("topic_discovery")
    
    try:
        hunter = TopicHunterAgent()
        topics = hunter.discover_topics(config.research_domain)
        
        if topics:
            state["topics"] = topics
            state["current_phase"] = "hypothesis_debate"
            log_agent_action("Orchestrator", "topics_discovered", {
                "count": len(topics),
                "iteration": state["iteration"],
                "topics": [t["title"] for t in topics[:3]]  # Log first 3 topics
            })
        else:
            # If no topics found and we've tried multiple times, stop
            if state["iteration"] >= 3:
                state["current_phase"] = "complete"
                state["meta_feedback"].append("No topics discovered after multiple attempts - stopping")
                log_agent_action("Orchestrator", "no_topics_found_after_retries", {"iteration": state["iteration"]})
            else:
                state["should_reset"] = True
                state["meta_feedback"].append("No topics discovered - resetting")
                log_agent_action("Orchestrator", "no_topics_found", {"iteration": state["iteration"]})
        
        return state
        
    except ResearchSourceUnavailable as e:
        message = str(e)
        logger.warning(message)
        state["terminal_error"] = message
        state["meta_feedback"].append(message)
        state["current_phase"] = "complete"
        state["should_continue"] = False
        log_agent_action("Orchestrator", "research_sources_unavailable", {"message": message})
        return state
    except Exception as e:
        logger.error(f"Topic discovery failed: {e}")
        # If we've had too many errors, stop
        if state["iteration"] >= 3:
            state["current_phase"] = "complete"
            state["meta_feedback"].append(f"Topic discovery failed after multiple attempts: {str(e)}")
        else:
            state["should_reset"] = True
            state["meta_feedback"].append(f"Topic discovery error: {str(e)}")
        return state

def hypothesis_debate_node(state: ResearchState) -> ResearchState:
    """Conduct hypothesis debate for the selected topic."""
    log_agent_action("Orchestrator", "start_hypothesis_debate", {"topics_remaining": len(state["topics"])})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("hypothesis_debate")
    
    try:
        if not state["topics"]:
            state["should_reset"] = True
            return state
        
        # Try all topics until one passes or we run out
        topics_tried = 0
        while state["topics"]:
            # Select the current best topic
            current_topic = state["topics"][0]
            state["selected_topic"] = current_topic
            topics_tried += 1
            
            log_agent_action("Orchestrator", "trying_topic", {
                "topic": current_topic["title"],
                "attempt": topics_tried,
                "topics_remaining": len(state["topics"])
            })
            
            # Conduct debate
            debate_system = HypothesisDebateSystem()
            debate_result = debate_system.conduct_debate(current_topic)
            state["debate_results"].append(debate_result)
            
            if debate_result.passed:
                state["hypothesis_passed"] = True
                state["current_phase"] = "planning"
                log_agent_action("Orchestrator", "hypothesis_passed", {
                    "topic": current_topic["title"],
                    "attempts": topics_tried
                })
                return state
            else:
                # Remove failed topic and try next
                state["topics"] = state["topics"][1:]
                log_agent_action("Orchestrator", "topic_failed", {
                    "topic": current_topic["title"],
                    "topics_remaining": len(state["topics"])
                })
        
        # If we get here, all topics failed
        state["should_reset"] = True
        state["meta_feedback"].append(f"All {topics_tried} topics failed hypothesis debate")
        log_agent_action("Orchestrator", "all_topics_failed", {"topics_tried": topics_tried})
        return state
        
    except Exception as e:
        logger.error(f"Hypothesis debate failed: {e}")
        state["meta_feedback"].append(f"Hypothesis debate error: {str(e)}")
        if len(state["topics"]) > 1:
            state["topics"] = state["topics"][1:]
            log_agent_action("Orchestrator", "trying_next_topic_after_error", {"remaining_topics": len(state["topics"])})
        else:
            state["should_reset"] = True
        return state

def planning_node(state: ResearchState) -> ResearchState:
    """Create or revise research plan (bidirectional Engineer→Planner)."""
    log_agent_action("Orchestrator", "start_planning", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("planning")
    
    try:
        if not state["selected_topic"]:
            state["should_reset"] = True
            return state
        
        planner = PlannerAgent()
        if state.get("plan") and state.get("plan_revision_requests"):
            plan = state["plan"]
            for req in state["plan_revision_requests"]:
                plan = planner.revise_plan(plan, req, state["selected_topic"])
            state["plan_revision_requests"] = []
            state["plan"] = plan
        else:
            plan = planner.create_plan(state["selected_topic"])
            state["plan"] = plan
        state["current_phase"] = "writing_narrative"
        
        log_agent_action("Orchestrator", "plan_created", {"sections": len(plan.get("sections", []))})
        return state
        
    except Exception as e:
        logger.error(f"Planning failed: {e}")
        state["should_reset"] = True
        state["meta_feedback"].append(f"Planning error: {str(e)}")
        return state

NARRATIVE_SECTION_NAMES = {"abstract", "introduction", "related work", "methods", "method"}
RESULTS_SECTION_NAMES = {"abstract", "results", "discussion", "conclusion", "experiments"}


def _plan_section_names(plan: Dict[str, Any]) -> List[str]:
    return [s.get("name", "Unknown") if isinstance(s, dict) else str(s) for s in plan.get("sections", [])]


def _numeric_values(value: Any) -> List[float]:
    """Collect numbers from raw experiment output, preserving only real numeric values."""
    if isinstance(value, bool):
        return []
    if isinstance(value, (int, float)):
        return [float(value)]
    if isinstance(value, dict):
        return [n for item in value.values() for n in _numeric_values(item)]
    if isinstance(value, (list, tuple)):
        return [n for item in value for n in _numeric_values(item)]
    return []


def verify_result_numbers(content: str, engineer_outputs: Dict[str, Any], rtol: float = 0.005) -> Dict[str, Any]:
    """Ensure decimal/percentage claims in results prose originate in experiment output.

    We intentionally inspect quantitative-looking claims only (decimal numbers and
    percentages), avoiding harmless structural references such as "Section 2".
    """
    allowed = _numeric_values(engineer_outputs)
    claims = [float(m.group(1)) / (100 if m.group(2) else 1) for m in re.finditer(r"(?<![\w.])(-?\d+(?:\.\d+)?)\s*(%)?", content)]
    # Integer values are usually prose/section labels; check them only when written as percentages.
    claims = [c for c in claims if not float(c).is_integer() or f"{int(c)}%" in content]
    mismatches = [c for c in claims if not any(abs(c - raw) <= max(abs(raw) * rtol, 1e-6) for raw in allowed)]
    return {"passed": bool(allowed) and not mismatches, "claims": claims, "allowed_values": allowed, "mismatches": mismatches}


def write_narrative_sections(state: ResearchState) -> ResearchState:
    """Draft non-result narrative before experiments; Methods contains no observed metrics."""
    log_agent_action("Orchestrator", "start_writing_narrative", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("writing_narrative")
    
    try:
        if not state["plan"]:
            state["should_reset"] = True
            return state
        
        writer = WriterAgent()
        for section_name in _plan_section_names(state["plan"]):
            if section_name.lower() not in NARRATIVE_SECTION_NAMES:
                continue
            if section_name not in state["draft_sections"]:
                content = writer.draft_section(
                    section_name,
                    state["selected_topic"],
                    state["plan"],
                    {},  # narrative must not imply results before engineering
                )
                state["draft_sections"][section_name] = content
                state["current_section"] = section_name
                log_agent_action("Orchestrator", "section_written", {"section": section_name})
        
        state["current_phase"] = "engineering"
        return state
        
    except Exception as e:
        logger.error(f"Writing failed: {e}")
        state["meta_feedback"].append(f"Writing error: {str(e)}")
        state["error_count"] += 1
        # If too many consecutive errors, reset
        if state["error_count"] >= 3:
            state["should_reset"] = True
            return state
        # Don't reset on narrative writing errors, just continue to engineering.
        state["current_phase"] = "engineering"
        state["error_count"] = 0  # Reset error count on success
        return state


def write_results_sections(state: ResearchState) -> ResearchState:
    """Write Results/Discussion and refresh Abstract only after engineering completes."""
    log_agent_action("Orchestrator", "start_writing_results", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("writing_results")
    if not state.get("plan"):
        state["should_reset"] = True
        return state
    try:
        writer = WriterAgent()
        names = _plan_section_names(state["plan"])
        # Ensure the required post-experiment sections exist even when an older plan omits one.
        for required in ("Results", "Discussion", "Abstract"):
            if not any(name.lower() == required.lower() for name in names):
                names.append(required)
        for section_name in names:
            if section_name.lower() not in RESULTS_SECTION_NAMES:
                continue
            content = writer.draft_section(section_name, state["selected_topic"], state["plan"], state["engineer_outputs"])
            state["draft_sections"][section_name] = content
            state["current_section"] = section_name

        checked = {name: verify_result_numbers(state["draft_sections"][name], state["engineer_outputs"])
                   for name in names if name.lower() in RESULTS_SECTION_NAMES and name in state["draft_sections"]}
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
    except Exception as e:
        logger.error(f"Results writing failed: {e}")
        state["meta_feedback"].append(f"Results writing error: {e}")
        state["current_phase"] = "supervision"
        return state

def engineering_node(state: ResearchState) -> ResearchState:
    """Run branching experiment search + PIVOT/REFINE; may bounce to Planner."""
    log_agent_action("Orchestrator", "start_engineering", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("engineering")
    
    try:
        if not state["plan"]:
            state["current_phase"] = "supervision"
            return state
        
        engineer = EngineerAgent()
        experiments = state["plan"].get("experiments", [])
        method_text = "\n".join(
            state["draft_sections"].get(s, "")
            for s in ("Methods", "Method", "Experiments")
        )
        
        # Architecture 8.1: branch search when variants exist
        branch_winner_name = None
        if experiments and any(
            isinstance(e, dict) and (e.get("variants") or e.get("alternatives"))
            for e in experiments
        ):
            try:
                if tracker:
                    tracker.message("Engineering: branching cheap probes…")
                branched = engineer.run_branching_search(
                    [e for e in experiments if isinstance(e, dict)],
                    method_description=method_text,
                )
                name = branched.get("experiment_name") or branched.get("approach") or "branch_winner"
                branch_winner_name = name
                state["engineer_outputs"][name] = branched
                # Also alias under plan experiment name when possible
                winner_label = (branched.get("branch_search") or {}).get("winner")
                if winner_label and winner_label not in state["engineer_outputs"]:
                    state["engineer_outputs"][winner_label] = branched
            except Exception as e:
                logger.error(f"Branching search failed: {e}")
                if tracker:
                    tracker.message(f"Branching search failed: {e}", level="error")
        
        for experiment in experiments:
            if isinstance(experiment, dict):
                exp_name = experiment.get("name", "unknown_experiment")
                exp_config = experiment
            else:
                exp_name = str(experiment)
                exp_config = {"name": exp_name}
            
            # Skip if branching already produced this (or winner alias)
            if exp_name in state["engineer_outputs"]:
                continue
            if branch_winner_name and exp_name == branch_winner_name:
                continue
            
            try:
                if tracker:
                    tracker.message(f"Engineering: running {exp_name}")
                alts = list(exp_config.get("variants") or exp_config.get("alternatives") or [])
                output = engineer.run_experiment(
                    exp_config,
                    alternatives=alts,
                    method_description=method_text,
                )
                state["engineer_outputs"][exp_name] = output
                log_agent_action("Orchestrator", "experiment_run", {"experiment": exp_name})
            except Exception as e:
                logger.error(f"Experiment {exp_name} failed: {e}")
                state["engineer_outputs"][exp_name] = {
                    "success": False,
                    "error": str(e),
                    "experiment_name": exp_name,
                }
        
        # Bidirectional: plan revision requests → re-enter planning
        rev_reqs = engineer.consume_plan_revision_requests()
        if rev_reqs:
            state["plan_revision_requests"] = rev_reqs
            state["current_phase"] = "planning"
            # Keep prior outputs archived so Meta/UI can still summarize what failed
            state["engineer_outputs"] = {
                **{f"prior_{k}": v for k, v in state.get("engineer_outputs", {}).items()},
            }
            log_agent_action("Orchestrator", "plan_revision_requested", {"n": len(rev_reqs)})
            return state
        
        state["current_phase"] = "writing_results"
        return state
        
    except Exception as e:
        logger.error(f"Engineering failed: {e}")
        state["meta_feedback"].append(f"Engineering error: {str(e)}")
        state["error_count"] += 1
        if state["error_count"] >= 3:
            state["should_reset"] = True
            return state
        state["current_phase"] = "writing_results"
        state["error_count"] = 0
        return state

def supervision_node(state: ResearchState) -> ResearchState:
    """Evaluate quality using SupervisorAgent."""
    log_agent_action("Orchestrator", "start_supervision", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("supervision")
    
    try:
        supervisor = SupervisorAgent()
        
        # Evaluate each section (hard checks use engineer raw results)
        for section_name, content in state["draft_sections"].items():
            score, feedback = supervisor.evaluate_section(
                section_name,
                content,
                engineer_outputs=state.get("engineer_outputs"),
            )
            state["supervisor_scores"][section_name] = score
            state["supervisor_feedback"][section_name] = feedback
            if score < config.supervisor_threshold:
                tracker = get_tracker()
                if tracker:
                    tracker.bump("sections_bounced")
        
        state["reproducibility"] = reproducibility_dossier(state.get("plan"), state.get("engineer_outputs"))
        if not state["reproducibility"]["passed"]:
            state["meta_feedback"].append("Reproducibility dossier incomplete: " + json.dumps(state["reproducibility"]["checks"]))
        # Calculate overall score
        if state["supervisor_scores"]:
            overall_score = sum(state["supervisor_scores"].values()) / len(state["supervisor_scores"])
            
            if overall_score >= config.supervisor_threshold:
                state["current_phase"] = "editing"
                log_agent_action("Orchestrator", "quality_threshold_met", {"score": overall_score})
            else:
                state["current_phase"] = "meta_evaluation"
                log_agent_action("Orchestrator", "quality_below_threshold", {"score": overall_score})
        else:
            state["current_phase"] = "meta_evaluation"
        
        return state
        
    except Exception as e:
        logger.error(f"Supervision failed: {e}")
        state["meta_feedback"].append(f"Supervision error: {str(e)}")
        state["current_phase"] = "meta_evaluation"
        return state

def meta_evaluation_node(state: ResearchState) -> ResearchState:
    """Evaluate system performance using MetaAgent."""
    log_agent_action("Orchestrator", "start_meta_evaluation", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("meta_evaluation")
    
    try:
        meta_agent = MetaAgent()
        
        # Evaluate system performance
        feedback = meta_agent.evaluate_system_performance(state)
        state["meta_feedback"].append(feedback)
        
        # Check if should reset
        if meta_agent.should_reset(state):
            state["should_reset"] = True
            log_agent_action("Orchestrator", "meta_reset_triggered", {"iteration": state["iteration"]})
        elif meta_agent.should_continue(state):
            state["should_continue"] = True
            state["iteration"] += 1
            state["current_phase"] = "writing_narrative"  # Loop back through both writing passes
            log_agent_action("Orchestrator", "meta_continue_triggered", {"iteration": state["iteration"]})
        else:
            state["should_continue"] = False
            log_agent_action("Orchestrator", "meta_stop_triggered", {"iteration": state["iteration"]})
        
        return state
        
    except Exception as e:
        logger.error(f"Meta evaluation failed: {e}")
        state["meta_feedback"].append(f"Meta evaluation error: {str(e)}")
        # On meta evaluation error, increment iteration and continue
        state["iteration"] += 1
        state["should_continue"] = True
        state["current_phase"] = "writing_narrative"  # Loop back through both writing passes
        log_agent_action("Orchestrator", "meta_error_continue", {"iteration": state["iteration"]})
        return state

def editing_node(state: ResearchState) -> ResearchState:
    """Generate final LaTeX output using EditorAgent."""
    log_agent_action("Orchestrator", "start_editing", {})
    tracker = get_tracker()
    if tracker:
        tracker.set_phase("editing")
    
    try:
        editor = EditorAgent()
        
        final_paper = editor.create_final_paper(
            state["selected_topic"],
            state["draft_sections"],
            state["plan"],
            state["engineer_outputs"],
            debate_results=state.get("debate_results"),
        )
        
        latex_output = editor.generate_latex(final_paper)
        
        state["final_paper"] = final_paper
        state["latex_output"] = latex_output
        state["current_phase"] = "complete"
        
        log_agent_action("Orchestrator", "editing_complete", {})
        return state
        
    except Exception as e:
        logger.error(f"Editing failed: {e}")
        state["meta_feedback"].append(f"Editing error: {str(e)}")
        state["current_phase"] = "complete"
        return state

def reset_node(state: ResearchState) -> ResearchState:
    """Reset the system state."""
    current_iteration = state["iteration"]
    log_agent_action("Orchestrator", "system_reset", {"iteration": current_iteration})
    
    # Check if we've exceeded maximum iterations
    if current_iteration >= config.max_iterations:
        state["current_phase"] = "complete"
        state["meta_feedback"].append(f"Maximum iterations ({config.max_iterations}) reached - stopping")
        return state
    
    # Reset to initial state but increment iteration count
    state = initialize_state()
    state["iteration"] = current_iteration + 1
    state["current_phase"] = "topic_discovery"
    
    log_agent_action("Orchestrator", "reset_complete", {"new_iteration": state["iteration"]})
    return state

def should_reset(state: ResearchState) -> str:
    """Determine if system should reset."""
    if state["current_phase"] == "complete":
        return "end"
    return "reset" if state["should_reset"] else "continue"

def should_continue(state: ResearchState) -> str:
    """Determine if system should continue."""
    if state["current_phase"] == "complete":
        return "end"
    elif state["should_continue"]:
        return "continue"
    else:
        return "end"

def create_research_graph() -> StateGraph:
    """Create the LangGraph workflow for the research system."""
    
    # Create the graph
    workflow = StateGraph(ResearchState)
    
    # Add nodes
    workflow.add_node("topic_discovery", topic_discovery_node)
    workflow.add_node("hypothesis_debate", hypothesis_debate_node)
    workflow.add_node("planning", planning_node)
    workflow.add_node("writing_narrative", write_narrative_sections)
    workflow.add_node("engineering", engineering_node)
    workflow.add_node("writing_results", write_results_sections)
    workflow.add_node("supervision", supervision_node)
    workflow.add_node("meta_evaluation", meta_evaluation_node)
    workflow.add_node("editing", editing_node)
    workflow.add_node("reset", reset_node)
    
    # Set entry point
    workflow.set_entry_point("topic_discovery")
    
    # Add conditional edges
    workflow.add_conditional_edges(
        "topic_discovery",
        should_reset,
        {
            "reset": "reset",
            "continue": "hypothesis_debate",
            "end": END
        }
    )
    
    workflow.add_conditional_edges(
        "hypothesis_debate",
        should_reset,
        {
            "reset": "reset",
            "continue": "planning",
            "end": END
        }
    )
    
    workflow.add_edge("planning", "writing_narrative")
    workflow.add_edge("writing_narrative", "engineering")
    
    # Engineer may bounce back to Planner for revision
    workflow.add_conditional_edges(
        "engineering",
        lambda state: "planning" if state.get("current_phase") == "planning" else "writing_results",
        {
            "planning": "planning",
            "writing_results": "writing_results",
        },
    )
    workflow.add_conditional_edges(
        "writing_results",
        lambda state: "redraft" if state.get("current_phase") == "writing_results" else "supervision",
        {"redraft": "writing_results", "supervision": "supervision"},
    )
    
    workflow.add_conditional_edges(
        "supervision",
        lambda state: "editing" if state["current_phase"] == "editing" else "meta_evaluation",
        {
            "editing": "editing",
            "meta_evaluation": "meta_evaluation"
        }
    )
    
    workflow.add_conditional_edges(
        "meta_evaluation",
        should_continue,
        {
            "continue": "writing_narrative",
            "end": END
        }
    )
    
    workflow.add_edge("editing", END)
    workflow.add_edge("reset", "topic_discovery")
    
    return workflow


def create_checkpointer():
    """Create the on-disk checkpoint store used by both CLI and web runs."""
    try:
        from langgraph.checkpoint.sqlite import SqliteSaver
    except ImportError as exc:
        raise RuntimeError(
            "Durable checkpoints require langgraph-checkpoint-sqlite; install requirements.txt."
        ) from exc
    checkpoint_path = Path(config.checkpoint_path)
    checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
    # SqliteSaver owns the connection for the process lifetime.
    import sqlite3
    return SqliteSaver(sqlite3.connect(str(checkpoint_path), check_same_thread=False))

def save_results(state: ResearchState, output_dir: str = None):
    """Save research results to files."""
    try:
        output_dir = output_dir or config.output_dir
        os.makedirs(output_dir, exist_ok=True)
        
        # Handle StateSnapshot objects
        if hasattr(state, 'value'):
            state = state.value
        elif hasattr(state, '__dict__'):
            state = state.__dict__
        
        # Save final paper
        if state.get("latex_output"):
            latex_file = os.path.join(output_dir, "paper_output.tex")
            with open(latex_file, "w", encoding="utf-8") as f:
                f.write(state["latex_output"])
            logger.info(f"LaTeX output saved to {latex_file}")
        
        # Save plan
        if state.get("plan"):
            plan_file = os.path.join(output_dir, "plan.yaml")
            import yaml
            with open(plan_file, "w") as f:
                yaml.dump(state["plan"], f, default_flow_style=False)
            logger.info(f"Plan saved to {plan_file}")
        
        # Save state summary
        summary = {
            "iteration": state.get("iteration", 0),
            "selected_topic": state.get("selected_topic"),
            "sections_written": list(state.get("draft_sections", {}).keys()),
            "supervisor_scores": state.get("supervisor_scores", {}),
            "experiments_run": list(state.get("engineer_outputs", {}).keys()),
            "meta_feedback": state.get("meta_feedback", [])
        }
        
        summary_file = os.path.join(output_dir, "research_summary.json")
        with open(summary_file, "w") as f:
            json.dump(summary, f, indent=2)
        logger.info(f"Research summary saved to {summary_file}")
        
    except Exception as e:
        logger.error(f"Error saving results: {e}")
        print(f"Warning: Could not save results: {e}")

def main():
    """Main function to run the research system."""
    print("🤖 Multi-Agent AI Research System")
    print("=" * 50)
    
    try:
        # Validate configuration
        validate_config()
        print("✅ Configuration validated")
        
        # Create output directories
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.draft_versions_dir, exist_ok=True)
        os.makedirs("memory/vector_db", exist_ok=True)  # Create memory directory for FAISS
        print("✅ Output directories created")
        
        # Create the research graph
        workflow = create_research_graph()
        checkpointer = create_checkpointer()
        app = workflow.compile(checkpointer=checkpointer)
        print("✅ Research workflow compiled")
        
        # Initialize state + run tracker (observability / UI events)
        parser = argparse.ArgumentParser(description="Run or resume a ScholarGraph research workflow")
        parser.add_argument("--resume", metavar="RUN_ID", help="resume a durable checkpoint by run id")
        args = parser.parse_args()
        tracker = start_run(args.resume)
        initial_state = initialize_state()
        initial_state["run_id"] = args.resume or tracker.run_id
        print(f"✅ Initial state created (run_id={tracker.run_id})")
        
        # Run the research workflow
        print("\n🚀 Starting research workflow...")
        print("=" * 50)
        
        config_dict = {
            "configurable": {"thread_id": initial_state["run_id"]},
            "recursion_limit": 1000  # Increase recursion limit to prevent infinite loop errors
        }
        
        # Supplying None resumes from the latest checkpoint; a new run supplies initial state.
        for event in app.stream(None if args.resume else initial_state, config_dict):
            for node_name, node_output in event.items():
                if node_name != "__end__":
                    state = node_output
                    print(f"\n📋 {node_name.upper()}")
                    print(f"   Phase: {state['current_phase']}")
                    print(f"   Iteration: {state['iteration']}")
                    
                    if state['selected_topic']:
                        print(f"   Topic: {state['selected_topic']['title']}")
                    
                    if state['topics'] and node_name == "topic_discovery":
                        print(f"   Topics Found: {len(state['topics'])}")
                        if state['topics']:
                            print(f"   Best Topic: {state['topics'][0]['title']}")
                    
                    if state['debate_results'] and node_name == "hypothesis_debate":
                        print(f"   Debates Completed: {len(state['debate_results'])}")
                        if state['debate_results']:
                            last_debate = state['debate_results'][-1]
                            print(f"   Last Debate: {'PASSED' if last_debate.passed else 'FAILED'}")
                            if last_debate.passed:
                                print(f"   Selected Topic: {state['selected_topic']['title']}")
                            else:
                                print(f"   Topics Remaining: {len(state['topics'])}")
                    
                    if node_name == "reset":
                        print(f"   System Reset - New Iteration: {state['iteration']}")
                    
                    if state['supervisor_scores']:
                        avg_score = sum(state['supervisor_scores'].values()) / len(state['supervisor_scores'])
                        print(f"   Avg Score: {avg_score:.2f}")
                    
                    if state['draft_sections']:
                        print(f"   Sections: {len(state['draft_sections'])}")
        
        # Save results using the last known state
        save_results(state)
        if tracker:
            tracker.complete(success=bool(state.get("latex_output")))
        
        print("\n" + "=" * 50)
        print("🎉 Research workflow completed!")
        
        if state["latex_output"]:
            print(f"📄 LaTeX paper generated: {config.output_dir}/paper_output.tex")
        
        if state["supervisor_scores"]:
            avg_score = sum(state["supervisor_scores"].values()) / len(state["supervisor_scores"])
            print(f"📊 Final average score: {avg_score:.2f}")
        
        print(f"📁 All outputs saved to: {config.output_dir}")
        
    except Exception as e:
        logger.error(f"Research system failed: {e}")
        print(f"❌ Error: {e}")
        return 1
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code) 
