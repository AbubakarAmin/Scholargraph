"""Composition root for the multi-agent research system."""

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.append(str(Path(__file__).parent))

from langgraph.graph import StateGraph

from core.artifacts import save_results as write_results
from core.config import config, validate_config
from core.context import create_run_context, get_active_context
from core.memory import memory
from core.pipeline import ResearchPipeline
from core.run_log import get_tracker, start_run
from core.state import ResearchState, initialize_state
from core.utils import log_agent_action
from core.verification import reproducibility_dossier
from core.workflow import create_research_graph as build_research_graph
from core.workflow import create_qa_graph as build_qa_graph
from core import workflow_nodes
from core.workflow_nodes import (
    editing_node,
    data_validation_node,
    engineering_node,
    hypothesis_debate_node,
    independent_validation_node,
    is_valid_plan,
    meta_evaluation_node,
    planning_node,
    qa_answer_node,
    qa_literature_retrieval_node,
    qa_verification_node,
    reset_node,
    should_continue,
    should_reset,
    supervision_node,
    terminal_planning_failure,
    verify_result_numbers,
    write_narrative_sections,
)
from agents.editor import EditorAgent
from agents.engineer import EngineerAgent
from agents.hypothesis_debate import HypothesisDebateSystem
from agents.meta_agent import MetaAgent
from agents.planner import PlannerAgent
from agents.supervisor import SupervisorAgent
from agents.topic_hunter import ResearchSourceUnavailable, TopicHunterAgent
from agents.writer import WriterAgent

logging.basicConfig(
    level=getattr(logging, config.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def topic_discovery_node(state: ResearchState) -> ResearchState:
    """Compatibility wrapper that keeps monkeypatchable agent imports working."""
    workflow_nodes.TopicHunterAgent = TopicHunterAgent
    return workflow_nodes.topic_discovery_node(state)


def write_results_sections(state: ResearchState) -> ResearchState:
    """Compatibility wrapper that keeps monkeypatchable agent imports working."""
    workflow_nodes.WriterAgent = WriterAgent
    return workflow_nodes.write_results_sections(state)


def create_research_graph() -> StateGraph:
    """Create the LangGraph workflow from the extracted phase nodes."""
    return build_research_graph(
        {
            "topic_discovery": topic_discovery_node,
            "hypothesis_debate": hypothesis_debate_node,
            "planning": planning_node,
            "terminal_planning_failure": terminal_planning_failure,
            "data_validation": data_validation_node,
            "writing_narrative": write_narrative_sections,
            "engineering": engineering_node,
            "independent_validation": independent_validation_node,
            "writing_results": write_results_sections,
            "supervision": supervision_node,
            "meta_evaluation": meta_evaluation_node,
            "editing": editing_node,
            "reset": reset_node,
            "should_reset": should_reset,
            "should_continue": should_continue,
            "is_valid_plan": is_valid_plan,
        }
    )


def create_qa_mode_graph() -> StateGraph:
    """Create the QA-mode LangGraph workflow."""
    return build_qa_graph(
        {
            "qa_literature_retrieval": qa_literature_retrieval_node,
            "qa_answer": qa_answer_node,
            "qa_verification": qa_verification_node,
        }
    )


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
    import sqlite3
    return SqliteSaver(sqlite3.connect(str(checkpoint_path), check_same_thread=False))


def save_results(state: ResearchState, output_dir: str = None):
    """Compatibility wrapper for the artifact service."""
    write_results(state, output_dir)


def main():
    """Run the research system from the command line."""
    print("🤖 Multi-Agent AI Research System")
    print("=" * 50)
    try:
        validate_config()
        print("✅ Configuration validated")
        os.makedirs(config.output_dir, exist_ok=True)
        os.makedirs(config.draft_versions_dir, exist_ok=True)
        os.makedirs("memory/vector_db", exist_ok=True)
        print("✅ Output directories created")

        parser = argparse.ArgumentParser(description="Run or resume a ScholarGraph research workflow")
        parser.add_argument("--resume", metavar="RUN_ID", help="resume a durable checkpoint by run id")
        parser.add_argument("--mode", choices=["full_research", "qa"], default="full_research",
                            help="run mode: full_research (default) or qa (literature synthesis)")
        parser.add_argument("--query", metavar="QUERY", help="user query for QA mode")
        args = parser.parse_args()
        if args.mode == "qa" and not args.query:
            parser.error("--query is required when --mode=qa")
        tracker = start_run(args.resume)
        pipeline = ResearchPipeline(
            create_research_graph,
            create_checkpointer,
            context=create_run_context(tracker),
            mode_graphs={"qa": create_qa_mode_graph},
        )
        initial_state = initialize_state(mode=args.mode)
        initial_state["run_id"] = args.resume or tracker.run_id
        if args.mode == "qa" and args.query:
            initial_state["user_query"] = args.query
        print(f"✅ Initial state created (run_id={tracker.run_id})")
        print("\n🚀 Starting research workflow...")
        print("=" * 50)

        def print_node_progress(node_name, node_output):
            state = node_output
            print(f"\n📋 {node_name.upper()}")
            print(f"   Phase: {state['current_phase']}")
            print(f"   Iteration: {state['iteration']}")
            if state["selected_topic"]:
                print(f"   Topic: {state['selected_topic']['title']}")
            if state["topics"] and node_name == "topic_discovery":
                print(f"   Topics Found: {len(state['topics'])}")
                print(f"   Best Topic: {state['topics'][0]['title']}")
            if state["debate_results"] and node_name == "hypothesis_debate":
                last_debate = state["debate_results"][-1]
                print(f"   Debates Completed: {len(state['debate_results'])}")
                print(f"   Last Debate: {'PASSED' if last_debate.passed else 'FAILED'}")
                if last_debate.passed:
                    print(f"   Selected Topic: {state['selected_topic']['title']}")
                else:
                    print(f"   Topics Remaining: {len(state['topics'])}")
            if node_name == "reset":
                print(f"   System Reset - New Iteration: {state['iteration']}")
            if state["supervisor_scores"]:
                average = sum(state["supervisor_scores"].values()) / len(state["supervisor_scores"])
                print(f"   Avg Score: {average:.2f}")
            if state["draft_sections"]:
                print(f"   Sections: {len(state['draft_sections'])}")

        result = pipeline.run(
            initial_state,
            initial_state["run_id"],
            resume=bool(args.resume),
            on_node=print_node_progress,
            finalize=save_results,
        )
        state = result.state
        print("\n" + "=" * 50)
        if state.get("mode") == "qa":
            print("✅ QA synthesis completed!")
            qa = state.get("qa_answer")
            if qa:
                print(f"\n📝 Query: {qa.get('query') or state.get('user_query', '?')}")
                print(f"\n📖 Answer:\n{qa.get('answer', 'No answer')}")
                findings = qa.get("key_findings", [])
                if findings:
                    print(f"\n🔑 Key Findings:")
                    for i, f in enumerate(findings, 1):
                        print(f"  {i}. {f}")
                bib = qa.get("bibliography", [])
                if bib:
                    print(f"\n📚 Bibliography ({len(bib)} sources):")
                    for b in bib:
                        print(f"  - {b.get('title', '?')} ({b.get('year', '?')})")
                citation = state.get("qa_citation_verification", {})
                if citation:
                    print(f"\n✔️  Citations: {'PASSED' if citation.get('passed') else 'ISSUES'} (score: {citation.get('score', 0):.1f})")
            print(f"\n📁 Output saved to: {config.output_dir}")
        else:
            print("🎉 Research workflow completed!")
            if state["latex_output"]:
                print(f"📄 LaTeX paper generated: {config.output_dir}/paper_output.tex")
            if state["supervisor_scores"]:
                average = sum(state["supervisor_scores"].values()) / len(state["supervisor_scores"])
                print(f"📊 Final average score: {average:.2f}")
            print(f"📁 All outputs saved to: {config.output_dir}")
    except Exception as exc:
        logger.error(f"Research system failed: {exc}")
        print(f"❌ Error: {exc}")
        return 1
    return 0


if __name__ == "__main__":
    sys.exit(main())
