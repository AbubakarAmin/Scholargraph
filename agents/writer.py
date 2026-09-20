"""
WriterAgent - Drafts paper sections and integrates citations and code outputs.
Handles abstract, introduction, methods, results, and conclusion sections.
"""

import json
import logging
from typing import Dict, Any, List, Optional
from datetime import datetime

from core.config import config
from core.utils import log_agent_action, extract_citations, is_degenerate_llm_output, strip_markdown_headers
from core.llm import call_llm, generate_embedding, get_llm_client
from core.context import RunContext, get_active_context
from core.contracts import ExperimentOutput, Plan, PlanSection, Topic
from core.memory import memory

logger = logging.getLogger(__name__)


_MIN_SECTION_BODY_CHARS = {
    "abstract": 120,
    "introduction": 400,
    "related work": 400,
    "methods": 300,
    "experiments": 250,
    "results": 200,
    "discussion": 250,
    "conclusion": 150,
    "limitations": 120,
}

class WriterAgent:
    """Agent for drafting research paper sections."""
    
    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()
        self.client = get_llm_client()

    @property
    def runtime_config(self):
        return self.context.config if self.context else config

    @property
    def vector_memory(self):
        return self.context.memory if self.context else memory
    
class WriterAgent:
    """Agent for drafting research paper sections."""

    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()
        self.client = get_llm_client()
        self._active_revision_feedback: Optional[str] = None

    @property
    def runtime_config(self):
        return self.context.config if self.context else config

    @property
    def vector_memory(self):
        return self.context.memory if self.context else memory

    def draft_section(self, section_name: str, topic: Topic,
                      plan: Plan, engineer_outputs: Dict[str, ExperimentOutput],
                      revision_feedback: Optional[str] = None) -> str:
        """Draft a specific section of the research paper.

        revision_feedback carries deterministic check failures / reviewer
        feedback from a prior draft so revision passes repair specific
        defects instead of re-rolling the same prompt blind.
        """
        log_agent_action("WriterAgent", "start_drafting", {"section": section_name})
        logger.info("Drafting section: %s for topic: %s", section_name, topic.get("title", "unknown"))
        if revision_feedback:
            logger.info("Revision feedback present for section '%s': %.200s...", section_name, revision_feedback)
        self._active_revision_feedback = revision_feedback

        # Released exemplars only — cold-start returns empty until runs clear the release gate.
        exemplars = self._released_exemplars(section_name)

        # Get section requirements from plan
        section_plan = self._get_section_plan(section_name, plan)
        
        # Generate content based on section type
        if section_name.lower() == 'abstract':
            content = self._draft_abstract(topic, plan, engineer_outputs, exemplars=exemplars)
        elif section_name.lower() == 'introduction':
            content = self._draft_introduction(topic, plan, exemplars=exemplars)
        elif section_name.lower() == 'related work':
            content = self._draft_related_work(topic, plan)
        elif section_name.lower() == 'methods':
            content = self._draft_methods(topic, plan, engineer_outputs)
        elif section_name.lower() == 'experiments':
            content = self._draft_experiments(topic, plan, engineer_outputs)
        elif section_name.lower() == 'results':
            content = self._draft_results(topic, plan, engineer_outputs)
        elif section_name.lower() == 'discussion':
            content = self._draft_discussion(topic, plan, engineer_outputs)
        elif section_name.lower() == 'conclusion':
            content = self._draft_conclusion(topic, plan, engineer_outputs)
        else:
            content = self._draft_generic_section(section_name, topic, plan, engineer_outputs)
        
        # Store in memory
        self._store_section(section_name, content, topic)
        self._active_revision_feedback = None

        logger.info("Section '%s' complete: %d characters", section_name, len(content))

        log_agent_action("WriterAgent", "section_complete", {
            "section": section_name,
            "content_length": len(content),
            "revised": bool(revision_feedback),
        })

        return content

    def _revision_block(self) -> str:
        """Prompt block injecting prior check failures / reviewer feedback."""
        feedback = getattr(self, "_active_revision_feedback", None)
        if not feedback:
            return ""
        return (
            "\n\nREVISION REQUIRED — a previous draft of this section failed deterministic checks "
            "or review. Repair every issue below in the new draft; keep what already passed intact:\n"
            + str(feedback)[:2400]
            + "\n"
        )

    def _literature_block(self, topic: Topic, max_papers: int = 6, snippet_chars: int = 400) -> str:
        """Render retrieved literature evidence for prompt grounding."""
        papers = topic.get("literature_evidence") or []
        slim = []
        for paper in papers[:max_papers]:
            if not isinstance(paper, dict):
                continue
            slim.append({
                "title": paper.get("title", ""),
                "abstract": (paper.get("abstract") or "")[:snippet_chars],
                "doi": paper.get("doi"),
                "arxiv_id": paper.get("arxiv_id"),
                "year": paper.get("year"),
            })
        slim = [p for p in slim if p["title"] or p["abstract"]]
        if not slim:
            return "No retrieved literature evidence is available. Do not cite any paper by DOI, arXiv ID, or author-year."
        return json.dumps(slim, default=str)[:8000]

    def _released_exemplars(self, section_name: str) -> List[Dict[str, Any]]:
        """Only released structured exemplars may enter Writer prompts."""
        return self.vector_memory.get_prompt_context(
            namespace="writer_exemplars",
            outcome_status="released",
            k=3,
            purpose=section_name.lower(),
        )
    
    def _get_section_plan(self, section_name: str, plan: Plan) -> PlanSection:
        """Get the plan for a specific section."""
        for section in plan.get('sections', []):
            if section['name'].lower() == section_name.lower():
                return section
        return {}

    def _min_body_chars(self, section_name: str) -> int:
        return _MIN_SECTION_BODY_CHARS.get(section_name.lower(), 200)

    def _draft_with_retry(
        self,
        section_name: str,
        prompt: str,
        *,
        temperature: float = 0.6,
        fallback: Optional[str] = None,
    ) -> str:
        """Call the LLM once, retry on empty/garbled stubs, then fall back."""
        logger.info("LLM call for section '%s' (max 2 attempts, min %d chars)", section_name, self._min_body_chars(section_name))
        last_content = ""
        min_chars = self._min_body_chars(section_name)
        full_prompt = prompt + self._revision_block()
        for attempt in range(2):
            try:
                raw = call_llm(full_prompt, temperature=temperature, tier="strong")
                formatted = self._format_section_content(raw, section_name)
                body = strip_markdown_headers(formatted)
                if not is_degenerate_llm_output(formatted, min_chars=min_chars) and len(body) >= min_chars:
                    return formatted
                last_content = formatted
                logger.warning("Degenerate output for section '%s' on attempt %d", section_name, attempt + 1)
                log_agent_action("WriterAgent", "degenerate_section", {
                    "section": section_name,
                    "attempt": attempt + 1,
                    "content_length": len(formatted),
                    "preview": formatted[:120],
                })
            except Exception as exc:
                log_agent_action("WriterAgent", f"{section_name.lower().replace(' ', '_')}_error", {"error": str(exc)})
                break
        if fallback:
            return fallback
        if last_content and not is_degenerate_llm_output(last_content, min_chars=20):
            return last_content
        return self._format_section_content(
            f"Section draft unavailable after degenerate model output for {section_name}.",
            section_name,
        )
    
    def _draft_abstract(self, topic: Topic, plan: Plan,
                       engineer_outputs: Dict[str, ExperimentOutput],
                       exemplars: Optional[List[Dict[str, Any]]] = None) -> str:
        """Draft the abstract section."""
        prompt = f"""
        Write a concise abstract for the following research paper:

        Topic: {topic['title']}
        Description: {topic['description']}
        Research Questions: {plan.get('research_questions', [])}
        Expected Contributions: {plan.get('expected_contributions', [])}
        Released exemplar signals (metadata only): {json.dumps(exemplars or [])[:800]}

        Retrieved literature evidence (grounding context):
        {self._literature_block(topic, max_papers=4, snippet_chars=250)}

        Key Results (if available):
        {self._format_engineer_outputs(engineer_outputs)}

        The abstract should:
        1. State the problem clearly
        2. Describe the approach/methodology
        3. Summarize key results — only numbers copied exactly from Key Results above
        4. Highlight contributions and impact
        5. Be 150-250 words

        Citation policy: only cite sources that appear in the retrieved literature evidence
        above (by DOI or arXiv ID). If the evidence does not support a claim, omit the claim.
        Never invent DOIs, arXiv IDs, author-year pairs, or paper titles.

        Write a professional, academic abstract suitable for a research paper.
        """
        
        return self._draft_with_retry(
            "Abstract",
            prompt,
            temperature=0.6,
            fallback=self._create_fallback_abstract(topic, plan),
        )
    
    def _draft_introduction(self, topic: Topic, plan: Plan,
                            exemplars: Optional[List[Dict[str, Any]]] = None) -> str:
        """Draft the introduction section."""
        prompt = f"""
        Write an introduction section for the following research paper:

        Topic: {topic['title']}
        Description: {topic['description']}
        Rationale: {topic.get('rationale', 'N/A')}
        Impact: {topic.get('impact', 'N/A')}
        Research Questions: {plan.get('research_questions', [])}
        Expected Contributions: {plan.get('expected_contributions', [])}
        Released exemplar signals (metadata only): {json.dumps(exemplars or [])[:800]}

        Retrieved literature evidence (use these sources for background and gap claims):
        {self._literature_block(topic)}

        The introduction should include:
        1. Background and motivation, grounded in the retrieved literature evidence
        2. Problem statement
        3. Challenges and limitations of existing work (reference the specific papers above)
        4. Our approach and contributions
        5. Paper organization

        Citation policy: every citation must come from the retrieved literature evidence
        above, cited by DOI (doi:10.XXXX/...) or arXiv ID (arXiv:XXXX.XXXXX). Never invent
        DOIs, arXiv IDs, author-year pairs, or paper titles. If the evidence is insufficient
        for a claim, state that explicitly instead of citing.

        Write 2-3 pages of professional academic content.
        """
        
        return self._draft_with_retry(
            "Introduction",
            prompt,
            temperature=0.7,
            fallback=self._create_fallback_introduction(topic, plan),
        )
    
    def _draft_related_work(self, topic: Topic, plan: Plan) -> str:
        """Draft the related work section."""
        retrieved = topic.get("literature_evidence") or plan.get("literature_evidence") or []
        prompt = f"""
        Write a related work section for the following research topic:
        
        Topic: {topic['title']}
        Description: {topic['description']}
        Research Questions: {plan.get('research_questions', [])}

        Retrieved literature evidence (use only these abstracts for paper-specific claims):
        {json.dumps(retrieved, indent=2, default=str)[:12000]}
        
        The related work should:
        1. Survey relevant literature
        2. Identify gaps in existing work
        3. Position our contribution
        4. Discuss limitations of current approaches
        5. Build motivation for our work
        
        Write 2-3 pages of comprehensive literature review.
        Do not invent paper methods, authors, titles, or citations. If the retrieved
        evidence is insufficient, state that the claim is not established.
        """
        
        return self._draft_with_retry(
            "Related Work",
            prompt,
            temperature=0.6,
            fallback=self._create_fallback_related_work(topic, plan),
        )
    
    def _draft_methods(self, topic: Topic, plan: Plan,
                      engineer_outputs: Dict[str, ExperimentOutput]) -> str:
        """Draft the methods section."""
        prompt = f"""
        Write a methods section for the following research:
        
        Topic: {topic['title']}
        Methodology: {plan.get('methodology', 'N/A')}
        Experiments: {json.dumps(plan.get('experiments', []), indent=2)[:6000]}
        
        Implementation Details:
        {self._format_engineer_outputs(engineer_outputs)}
        
        The methods section should:
        1. Describe the overall approach
        2. Detail the methodology
        3. Explain implementation details
        4. Describe experimental setup
        5. Include algorithms and pseudocode where appropriate
        
        Write 3-4 pages of detailed methodology description.
        """
        
        return self._draft_with_retry(
            "Methods",
            prompt,
            temperature=0.5,
            fallback=self._create_fallback_methods(topic, plan),
        )
    
    def _draft_experiments(self, topic: Topic, plan: Plan,
                          engineer_outputs: Dict[str, ExperimentOutput]) -> str:
        """Draft the experiments section."""
        prompt = f"""
        Write an experiments section for the following research:
        
        Topic: {topic['title']}
        Experiments: {json.dumps(plan.get('experiments', []), indent=2)[:6000]}
        
        Experimental Results:
        {self._format_engineer_outputs(engineer_outputs)}
        
        The experiments section should:
        1. Describe experimental setup
        2. Detail datasets and baselines
        3. Present experimental results
        4. Include tables and figures
        5. Analyze performance metrics
        
        Write 4-5 pages of comprehensive experimental evaluation.
        """
        
        return self._draft_with_retry(
            "Experiments",
            prompt,
            temperature=0.6,
            fallback=self._create_fallback_experiments(topic, plan),
        )
    
    def _draft_results(self, topic: Topic, plan: Plan,
                      engineer_outputs: Dict[str, ExperimentOutput]) -> str:
        """Draft the results section."""
        prompt = f"""
        Write a results section for the following research:

        Topic: {topic['title']}
        Expected Contributions: {plan.get('expected_contributions', [])}

        Experimental Results (single source of truth):
        {self._format_engineer_outputs(engineer_outputs)}

        The results section should:
        1. Analyze experimental results
        2. Compare against baselines
        3. Conduct ablation studies
        4. Provide insights and analysis
        5. Discuss implications

        Number policy (deterministic checks enforce this):
        - Copy every number EXACTLY from the Experimental Results source of truth above.
        - Every quantitative sentence must include n= (seed/sample count) and the
          standard deviation or a confidence interval for the reported metric.
        - Report the statistical test outcome (test name and p-value) for each comparison
          where the source of truth provides one.
        - Do not estimate, round, extrapolate, or invent any number. If a measurement is
          absent from the source of truth, describe it qualitatively instead.
        - For each comparison, state whether it supports or falsifies the hypothesis's
          falsifiable prediction; report negative or inconclusive outcomes honestly.
        """
        
        return self._draft_with_retry(
            "Results",
            prompt,
            temperature=0.6,
            fallback=self._create_fallback_results(topic, plan),
        )
    
    def _draft_conclusion(self, topic: Topic, plan: Plan,
                         engineer_outputs: Dict[str, ExperimentOutput]) -> str:
        """Draft the conclusion section."""
        prompt = f"""
        Write a conclusion section for the following research:
        
        Topic: {topic['title']}
        Expected Contributions: {plan.get('expected_contributions', [])}
        
        Key Results:
        {self._format_engineer_outputs(engineer_outputs)}
        
        The conclusion should:
        1. Summarize key contributions
        2. Highlight main results
        3. Discuss limitations
        4. Suggest future work
        5. End with impact statement
        
        Write 1-2 pages of conclusion.
        """
        
        return self._draft_with_retry(
            "Conclusion",
            prompt,
            temperature=0.7,
            fallback=self._create_fallback_conclusion(topic, plan),
        )

    def _draft_discussion(self, topic: Topic, plan: Plan,
                          engineer_outputs: Dict[str, ExperimentOutput]) -> str:
        """Interpret results after engineering without introducing new measurements."""
        prompt = f"""
Write the Discussion section for this paper.
Topic: {topic['title']}
Experimental source of truth:
{self._format_engineer_outputs(engineer_outputs)}

Interpret implications, limitations, and failures. Any quantitative statement must
be copied from the experimental source of truth above; do not estimate, fabricate,
or introduce a new number. If a measurement is absent, describe it qualitatively.
"""
        return self._draft_with_retry(
            "Discussion",
            prompt,
            temperature=0.5,
            fallback="# Discussion\n\nThe observed experimental results are interpreted using the recorded outputs; no additional measurements are claimed.",
        )
    
    def _draft_generic_section(self, section_name: str, topic: Topic,
                              plan: Plan, engineer_outputs: Dict[str, ExperimentOutput]) -> str:
        """Draft a generic section via _draft_with_retry for revision feedback and min-char guards."""
        prompt = f"""
        Write a {section_name} section for the following research:
        
        Topic: {topic['title']}
        Description: {topic['description']}
        
        The {section_name} should be appropriate for a research paper and cover relevant content for this section.
        Write professional academic content suitable for publication.
        """

        try:
            content = self._draft_with_retry(section_name, prompt, temperature=0.6)
            return self._format_section_content(content, section_name)
        except Exception as e:
            log_agent_action("WriterAgent", "generic_section_error", {"error": str(e)})
            return f"Error drafting {section_name} section: {str(e)}"
    
    def _format_engineer_outputs(self, engineer_outputs: Dict[str, ExperimentOutput]) -> str:
        """Serialize structured evidence without exposing raw Engineer diagnostics."""
        if not engineer_outputs:
            return json.dumps({"experiments": []}, sort_keys=True)

        evidence = []
        failures = []
        for exp_name, output in engineer_outputs.items():
            if not isinstance(output, dict):
                continue
            if not output.get("success"):
                failures.append(self._summarize_engineer_failure(exp_name, output))
                continue
            evidence.append({
                "experiment": exp_name,
                "outcome": output.get("outcome", "inconclusive"),
                "aggregate_metrics": output.get("aggregate_metrics") or {},
                "results": output.get("results") or {},
                "contract_hash": output.get("contract_hash"),
            })
        return json.dumps(
            {"experiments": evidence, "failure_summaries": failures},
            sort_keys=True,
            default=str,
        )

    @staticmethod
    def _summarize_engineer_failure(exp_name: str, output: Dict[str, Any]) -> Dict[str, Any]:
        """Map failure evidence to safe categories; never copy exception text."""
        raw = str(output.get("error") or "").lower()
        if any(token in raw for token in ("timeout", "timed out", "resource")):
            category = "timeout_or_resource"
            impact = "The experiment did not complete within the execution budget."
        elif any(token in raw for token in ("sandbox", "forbidden", "blocked")):
            category = "sandbox_validation"
            impact = "The generated implementation was rejected by execution policy."
        elif any(token in raw for token in ("syntax", "nameerror", "typeerror", "import", "dependency", "api")):
            category = "dependency_or_api"
            impact = "The implementation could not be executed with the available runtime dependencies."
        elif "claim" in raw or output.get("code_claim_consistency"):
            category = "code_claim_mismatch"
            impact = "The implementation did not satisfy the committed methodological claims."
        elif "feasible" in raw or "dataset" in raw:
            category = "plan_infeasible"
            impact = "The committed experiment requirements were not executable in the available environment."
        else:
            category = "missing_metrics"
            impact = "The experiment did not produce a verified result artifact."
        return {
            "experiment": exp_name,
            "status": "failed",
            "category": category,
            "impact": impact,
            "attempt_count": len(output.get("decision_log") or output.get("attempts") or []),
        }
    
    def _format_section_content(self, content: str, section_name: str) -> str:
        """Format section content with proper structure."""
        # Clean up content
        content = content.strip()
        
        # Add section header if not present
        if not content.startswith(f"# {section_name}") and not content.startswith(f"## {section_name}"):
            content = f"# {section_name}\n\n{content}"
        
        return content
    
    def _store_section(self, section_name: str, content: str, topic: Topic):
        """Store section content in memory as generated_narrative (prompt-ineligible)."""
        try:
            from core.run_log import get_tracker

            citations = extract_citations(content)
            tracker = get_tracker()
            self.vector_memory.add_embedding(
                generate_embedding(content),
                {
                    "type": "paper_section",
                    "namespace": "writer_sections",
                    "content_class": "generated_narrative",
                    "retrieval_eligible": False,
                    "agent": "WriterAgent",
                    "run_id": tracker.run_id if tracker else "unknown",
                    "outcome_status": "unknown",
                    "section": section_name,
                    "topic": topic["title"],
                    "content_length": len(content),
                    "citations": citations,
                    "content": content[:2000],
                    "timestamp": datetime.now().isoformat(),
                },
            )
        except Exception as e:
            log_agent_action("WriterAgent", "section_storage_error", {"error": str(e)})

    def store_released_exemplar(self, section_name: str, topic: Topic, quality_score: float):
        """Publish a prompt-eligible exemplar only after a section clears the release gate."""
        try:
            from core.run_log import get_tracker

            tracker = get_tracker()
            # Structured signal only — no draft body — so exemplars cannot poison prose.
            self.vector_memory.add_embedding(
                generate_embedding(f"{section_name} {topic.get('title', '')}"),
                {
                    "namespace": "writer_exemplars",
                    "content_class": "structured_signal",
                    "retrieval_eligible": True,
                    "agent": "WriterAgent",
                    "run_id": tracker.run_id if tracker else "unknown",
                    "outcome_status": "released",
                    "signal": {
                        "purpose": section_name.lower(),
                        "section": section_name,
                        "topic_title": topic.get("title"),
                        "quality_score": float(quality_score),
                    },
                },
            )
        except Exception as e:
            log_agent_action("WriterAgent", "exemplar_storage_error", {"error": str(e)})
    
    # Fallback methods for error handling
    def _create_fallback_abstract(self, topic: Dict[str, Any], plan: Dict[str, Any]) -> str:
        return f"""
# Abstract

This paper presents research on {topic['title']}. We address the problem of {topic['description']} 
and propose a novel approach that {topic.get('impact', 'provides significant improvements')}. 
        Our contributions include {', '.join(str(c.get('claim', c)) if isinstance(c, dict) else str(c) for c in plan.get('expected_contributions', ['novel methodology', 'comprehensive evaluation']))}. 
        No verified experimental results are available for this draft. Do not make empirical claims.
"""
    
    def _create_fallback_introduction(self, topic: Dict[str, Any], plan: Dict[str, Any]) -> str:
        return f"""
# Introduction

{topic['title']} represents an important challenge in {self.runtime_config.research_domain}. 
Current approaches have limitations in {topic.get('rationale', 'scalability and efficiency')}. 
This work addresses these challenges by {topic.get('impact', 'introducing novel methods')}.

Our main contributions are:
{chr(10).join([f"- {c.get('claim', c) if isinstance(c, dict) else c}" for c in plan.get('expected_contributions', ['Novel approach', 'Comprehensive evaluation', 'Practical insights'])])}

The remainder of this paper is organized as follows: Section 2 reviews related work, 
Section 3 describes our methodology, Section 4 presents experimental results, 
and Section 5 concludes with future work.
"""
    
    def _create_fallback_related_work(self, topic: Dict[str, Any], plan: Dict[str, Any]) -> str:
        return f"""
# Related Work

Previous work in {self.runtime_config.research_domain} has addressed various aspects of {topic['title']}. 
However, existing approaches have limitations in {topic.get('rationale', 'scalability and efficiency')}. 
Our work builds upon these foundations while addressing key gaps in the literature.

Recent advances in the field have shown promise, but challenges remain in implementation 
and practical deployment. Our approach addresses these limitations through {topic.get('impact', 'novel methodology')}.
"""
    
    def _create_fallback_methods(self, topic: Dict[str, Any], plan: Dict[str, Any]) -> str:
        return f"""
# Methods

Our approach to {topic['title']} involves {plan.get('methodology', 'experimental evaluation with quantitative analysis')}. 
We implement a comprehensive methodology that addresses the key challenges identified in our problem statement.

The experimental setup includes standard benchmarks and evaluation metrics to ensure 
reproducibility and fair comparison with existing methods.
"""
    
    def _create_fallback_experiments(self, topic: Dict[str, Any], plan: Dict[str, Any]) -> str:
        return f"""
# Experiments

We conduct extensive experiments to evaluate our approach on {topic['title']}. 
Our experimental setup includes multiple datasets and baseline comparisons to ensure 
comprehensive evaluation of our contributions.

No verified results are available. This section must not claim improvements.
"""
    
    def _create_fallback_results(self, topic: Dict[str, Any], plan: Dict[str, Any]) -> str:
        return f"""
# Results

Verified experimental results are required before making claims about {topic['title']}.

Analysis reveals important insights about the effectiveness of different components 
and provides guidance for future research directions.
"""
    
    def _create_fallback_conclusion(self, topic: Dict[str, Any], plan: Dict[str, Any]) -> str:
        return f"""
# Conclusion

This paper presented research on {topic['title']}, addressing key challenges in {self.runtime_config.research_domain}. 
Our main contributions include {', '.join(str(c.get('claim', c)) if isinstance(c, dict) else str(c) for c in plan.get('expected_contributions', ['novel methodology', 'comprehensive evaluation']))}.

Future work will explore extensions to other domains and applications, building upon 
the foundation established in this research.
"""

# Example usage
if __name__ == "__main__":
    writer = WriterAgent()
    
    example_topic = {
        'title': 'Novel Attention Mechanisms for Transformer Models',
        'description': 'Developing new attention mechanisms that improve efficiency and interpretability',
        'rationale': 'Current attention mechanisms have limitations in scalability',
        'impact': 'Could enable larger, more efficient language models',
        'feasibility': 8
    }
    
    example_plan = {
        'research_questions': ['How can we improve attention efficiency?'],
        'expected_contributions': ['Novel attention mechanism', 'Improved efficiency'],
        'methodology': 'Experimental evaluation with quantitative analysis'
    }
    
    abstract = writer.draft_section('Abstract', example_topic, example_plan, {})
    print(f"Generated abstract ({len(abstract)} characters)") 
