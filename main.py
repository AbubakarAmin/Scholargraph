"""
Main orchestration module for the multi-agent research system.
Uses LangGraph to coordinate all agents and manage the research workflow.
"""

import os
import sys
import json
import logging
from pathlib import Path
from typing import TypedDict, Dict, Any, List, Optional
from datetime import datetime

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver

# Import core modules
from core.config import config, validate_config
from core.memory import memory
from core.utils import setup_gemini, log_agent_action

# Import agents
from agents.meta_agent import MetaAgent
from agents.topic_hunter import TopicHunterAgent
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
        error_count=0
    )

def topic_discovery_node(state: ResearchState) -> ResearchState:
    """Discover research topics using TopicHunterAgent."""
    log_agent_action("Orchestrator", "start_topic_discovery", {"iteration": state["iteration"]})
    
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
        # Try next topic if available, otherwise reset
        if len(state["topics"]) > 1:
            state["topics"] = state["topics"][1:]
            log_agent_action("Orchestrator", "trying_next_topic_after_error", {"remaining_topics": len(state["topics"])})
        else:
            state["should_reset"] = True
        return state
        
    except Exception as e:
        logger.error(f"Hypothesis debate failed: {e}")
        state["meta_feedback"].append(f"Hypothesis debate error: {str(e)}")
        # Try next topic if available, otherwise reset
        if len(state["topics"]) > 1:
            state["topics"] = state["topics"][1:]
            log_agent_action("Orchestrator", "trying_next_topic_after_error", {"remaining_topics": len(state["topics"])})
        else:
            state["should_reset"] = True
        return state

def planning_node(state: ResearchState) -> ResearchState:
    """Create research plan using PlannerAgent."""
    log_agent_action("Orchestrator", "start_planning", {})
    
    try:
        if not state["selected_topic"]:
            state["should_reset"] = True
            return state
        
        planner = PlannerAgent()
        plan = planner.create_plan(state["selected_topic"])
        state["plan"] = plan
        state["current_phase"] = "writing"
        
        log_agent_action("Orchestrator", "plan_created", {"sections": len(plan.get("sections", []))})
        return state
        
    except Exception as e:
        logger.error(f"Planning failed: {e}")
        state["should_reset"] = True
        state["meta_feedback"].append(f"Planning error: {str(e)}")
        return state

def writing_node(state: ResearchState) -> ResearchState:
    """Write paper sections using WriterAgent."""
    log_agent_action("Orchestrator", "start_writing", {})
    
    try:
        if not state["plan"]:
            state["should_reset"] = True
            return state
        
        writer = WriterAgent()
        sections = state["plan"].get("sections", [])
        
        # Extract section names if sections are objects
        section_names = []
        for section in sections:
            if isinstance(section, dict):
                section_names.append(section.get("name", "Unknown"))
            else:
                section_names.append(section)
        
        for section_name in section_names:
            if section_name not in state["draft_sections"]:
                content = writer.draft_section(
                    section_name,
                    state["selected_topic"],
                    state["plan"],
                    state["engineer_outputs"]
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
        # Don't reset on writing errors, just continue to engineering
        state["current_phase"] = "engineering"
        state["error_count"] = 0  # Reset error count on success
        return state

def engineering_node(state: ResearchState) -> ResearchState:
    """Run experiments using EngineerAgent."""
    log_agent_action("Orchestrator", "start_engineering", {})
    
    try:
        if not state["plan"]:
            state["current_phase"] = "supervision"
            return state
        
        engineer = EngineerAgent()
        experiments = state["plan"].get("experiments", [])
        
        for experiment in experiments:
            if isinstance(experiment, dict):
                exp_name = experiment.get("name", "unknown_experiment")
                exp_config = experiment
            else:
                exp_name = str(experiment)
                exp_config = {"name": exp_name}
            
            if exp_name not in state["engineer_outputs"]:
                try:
                    output = engineer.run_experiment(exp_config)
                    state["engineer_outputs"][exp_name] = output
                    log_agent_action("Orchestrator", "experiment_run", {"experiment": exp_name})
                except Exception as e:
                    logger.error(f"Experiment {exp_name} failed: {e}")
                    state["engineer_outputs"][exp_name] = {
                        "success": False,
                        "error": str(e),
                        "experiment_name": exp_name
                    }
        
        state["current_phase"] = "supervision"
        return state
        
    except Exception as e:
        logger.error(f"Engineering failed: {e}")
        state["meta_feedback"].append(f"Engineering error: {str(e)}")
        state["error_count"] += 1
        # If too many consecutive errors, reset
        if state["error_count"] >= 3:
            state["should_reset"] = True
            return state
        # Don't reset on engineering errors, just continue to supervision
        state["current_phase"] = "supervision"
        state["error_count"] = 0  # Reset error count on success
        return state

def supervision_node(state: ResearchState) -> ResearchState:
    """Evaluate quality using SupervisorAgent."""
    log_agent_action("Orchestrator", "start_supervision", {})
    
    try:
        supervisor = SupervisorAgent()
        
        # Evaluate each section
        for section_name, content in state["draft_sections"].items():
            score, feedback = supervisor.evaluate_section(
                section_name,
                content
            )
            state["supervisor_scores"][section_name] = score
            state["supervisor_feedback"][section_name] = feedback
        
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
            state["current_phase"] = "writing"  # Loop back to writing for revisions
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
        state["current_phase"] = "writing"  # Loop back to writing for revisions
        log_agent_action("Orchestrator", "meta_error_continue", {"iteration": state["iteration"]})
        return state

def editing_node(state: ResearchState) -> ResearchState:
    """Generate final LaTeX output using EditorAgent."""
    log_agent_action("Orchestrator", "start_editing", {})
    
    try:
        editor = EditorAgent()
        
        final_paper = editor.create_final_paper(
            state["selected_topic"],
            state["draft_sections"],
            state["plan"],
            state["engineer_outputs"]
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
    workflow.add_node("writing", writing_node)
    workflow.add_node("engineering", engineering_node)
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
    
    workflow.add_edge("planning", "writing")
    workflow.add_edge("writing", "engineering")
    workflow.add_edge("engineering", "supervision")
    
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
            "continue": "writing",
            "end": END
        }
    )
    
    workflow.add_edge("editing", END)
    workflow.add_edge("reset", "topic_discovery")
    
    return workflow

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
        app = workflow.compile(checkpointer=MemorySaver())
        print("✅ Research workflow compiled")
        
        # Initialize state
        initial_state = initialize_state()
        print("✅ Initial state created")
        
        # Run the research workflow
        print("\n🚀 Starting research workflow...")
        print("=" * 50)
        
        config_dict = {
            "configurable": {"thread_id": "research_session"},
            "recursion_limit": 1000  # Increase recursion limit to prevent infinite loop errors
        }
        
        for event in app.stream(initial_state, config_dict):
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