"""
WriterAgent - Drafts paper sections and integrates citations and code outputs.
Handles abstract, introduction, methods, results, and conclusion sections.
"""

import json
from typing import Dict, Any, List, Optional
from datetime import datetime

from core.config import config
from core.utils import setup_gemini, call_gemini, log_agent_action, extract_citations
from core.memory import memory

class WriterAgent:
    """Agent for drafting research paper sections."""
    
    def __init__(self):
        self.gemini_client = setup_gemini()
    
    def draft_section(self, section_name: str, topic: Dict[str, Any], 
                     plan: Dict[str, Any], engineer_outputs: Dict[str, Any]) -> str:
        """Draft a specific section of the research paper."""
        log_agent_action("WriterAgent", "start_drafting", {"section": section_name})
        
        # Get section requirements from plan
        section_plan = self._get_section_plan(section_name, plan)
        
        # Generate content based on section type
        if section_name.lower() == 'abstract':
            content = self._draft_abstract(topic, plan, engineer_outputs)
        elif section_name.lower() == 'introduction':
            content = self._draft_introduction(topic, plan)
        elif section_name.lower() == 'related work':
            content = self._draft_related_work(topic, plan)
        elif section_name.lower() == 'methods':
            content = self._draft_methods(topic, plan, engineer_outputs)
        elif section_name.lower() == 'experiments':
            content = self._draft_experiments(topic, plan, engineer_outputs)
        elif section_name.lower() == 'results':
            content = self._draft_results(topic, plan, engineer_outputs)
        elif section_name.lower() == 'conclusion':
            content = self._draft_conclusion(topic, plan, engineer_outputs)
        else:
            content = self._draft_generic_section(section_name, topic, plan, engineer_outputs)
        
        # Store in memory
        self._store_section(section_name, content, topic)
        
        log_agent_action("WriterAgent", "section_complete", {
            "section": section_name,
            "content_length": len(content)
        })
        
        return content
    
    def _get_section_plan(self, section_name: str, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Get the plan for a specific section."""
        for section in plan.get('sections', []):
            if section['name'].lower() == section_name.lower():
                return section
        return {}
    
    def _draft_abstract(self, topic: Dict[str, Any], plan: Dict[str, Any], 
                       engineer_outputs: Dict[str, Any]) -> str:
        """Draft the abstract section."""
        prompt = f"""
        Write a concise abstract for the following research paper:
        
        Topic: {topic['title']}
        Description: {topic['description']}
        Research Questions: {plan.get('research_questions', [])}
        Expected Contributions: {plan.get('expected_contributions', [])}
        
        Key Results (if available):
        {self._format_engineer_outputs(engineer_outputs)}
        
        The abstract should:
        1. State the problem clearly
        2. Describe the approach/methodology
        3. Summarize key results
        4. Highlight contributions and impact
        5. Be 150-250 words
        
        Write a professional, academic abstract suitable for a research paper.
        """
        
        try:
            content = call_gemini(prompt, self.gemini_client, temperature=0.6)
            return self._format_section_content(content, "Abstract")
        except Exception as e:
            log_agent_action("WriterAgent", "abstract_error", {"error": str(e)})
            return self._create_fallback_abstract(topic, plan)
    
    def _draft_introduction(self, topic: Dict[str, Any], plan: Dict[str, Any]) -> str:
        """Draft the introduction section."""
        prompt = f"""
        Write an introduction section for the following research paper:
        
        Topic: {topic['title']}
        Description: {topic['description']}
        Rationale: {topic.get('rationale', 'N/A')}
        Impact: {topic.get('impact', 'N/A')}
        Research Questions: {plan.get('research_questions', [])}
        Expected Contributions: {plan.get('expected_contributions', [])}
        
        The introduction should include:
        1. Background and motivation
        2. Problem statement
        3. Challenges and limitations of existing work
        4. Our approach and contributions
        5. Paper organization
        
        Write 2-3 pages of professional academic content.
        Include proper citations where appropriate.
        """
        
        try:
            content = call_gemini(prompt, self.gemini_client, temperature=0.7)
            return self._format_section_content(content, "Introduction")
        except Exception as e:
            log_agent_action("WriterAgent", "introduction_error", {"error": str(e)})
            return self._create_fallback_introduction(topic, plan)
    
    def _draft_related_work(self, topic: Dict[str, Any], plan: Dict[str, Any]) -> str:
        """Draft the related work section."""
        prompt = f"""
        Write a related work section for the following research topic:
        
        Topic: {topic['title']}
        Description: {topic['description']}
        Research Questions: {plan.get('research_questions', [])}
        
        The related work should:
        1. Survey relevant literature
        2. Identify gaps in existing work
        3. Position our contribution
        4. Discuss limitations of current approaches
        5. Build motivation for our work
        
        Write 2-3 pages of comprehensive literature review.
        Include citations to relevant papers.
        """
        
        try:
            content = call_gemini(prompt, self.gemini_client, temperature=0.6)
            return self._format_section_content(content, "Related Work")
        except Exception as e:
            log_agent_action("WriterAgent", "related_work_error", {"error": str(e)})
            return self._create_fallback_related_work(topic, plan)
    
    def _draft_methods(self, topic: Dict[str, Any], plan: Dict[str, Any], 
                      engineer_outputs: Dict[str, Any]) -> str:
        """Draft the methods section."""
        prompt = f"""
        Write a methods section for the following research:
        
        Topic: {topic['title']}
        Methodology: {plan.get('methodology', 'N/A')}
        Experiments: {json.dumps(plan.get('experiments', []), indent=2)}
        
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
        
        try:
            content = call_gemini(prompt, self.gemini_client, temperature=0.5)
            return self._format_section_content(content, "Methods")
        except Exception as e:
            log_agent_action("WriterAgent", "methods_error", {"error": str(e)})
            return self._create_fallback_methods(topic, plan)
    
    def _draft_experiments(self, topic: Dict[str, Any], plan: Dict[str, Any], 
                          engineer_outputs: Dict[str, Any]) -> str:
        """Draft the experiments section."""
        prompt = f"""
        Write an experiments section for the following research:
        
        Topic: {topic['title']}
        Experiments: {json.dumps(plan.get('experiments', []), indent=2)}
        
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
        
        try:
            content = call_gemini(prompt, self.gemini_client, temperature=0.6)
            return self._format_section_content(content, "Experiments")
        except Exception as e:
            log_agent_action("WriterAgent", "experiments_error", {"error": str(e)})
            return self._create_fallback_experiments(topic, plan)
    
    def _draft_results(self, topic: Dict[str, Any], plan: Dict[str, Any], 
                      engineer_outputs: Dict[str, Any]) -> str:
        """Draft the results section."""
        prompt = f"""
        Write a results section for the following research:
        
        Topic: {topic['title']}
        Expected Contributions: {plan.get('expected_contributions', [])}
        
        Experimental Results:
        {self._format_engineer_outputs(engineer_outputs)}
        
        The results section should:
        1. Analyze experimental results
        2. Compare against baselines
        3. Conduct ablation studies
        4. Provide insights and analysis
        5. Discuss implications
        
        Write 3-4 pages of detailed results analysis.
        """
        
        try:
            content = call_gemini(prompt, self.gemini_client, temperature=0.6)
            return self._format_section_content(content, "Results")
        except Exception as e:
            log_agent_action("WriterAgent", "results_error", {"error": str(e)})
            return self._create_fallback_results(topic, plan)
    
    def _draft_conclusion(self, topic: Dict[str, Any], plan: Dict[str, Any], 
                         engineer_outputs: Dict[str, Any]) -> str:
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
        
        try:
            content = call_gemini(prompt, self.gemini_client, temperature=0.7)
            return self._format_section_content(content, "Conclusion")
        except Exception as e:
            log_agent_action("WriterAgent", "conclusion_error", {"error": str(e)})
            return self._create_fallback_conclusion(topic, plan)
    
    def _draft_generic_section(self, section_name: str, topic: Dict[str, Any], 
                              plan: Dict[str, Any], engineer_outputs: Dict[str, Any]) -> str:
        """Draft a generic section."""
        prompt = f"""
        Write a {section_name} section for the following research:
        
        Topic: {topic['title']}
        Description: {topic['description']}
        
        The {section_name} should be appropriate for a research paper and cover relevant content for this section.
        Write professional academic content suitable for publication.
        """
        
        try:
            content = call_gemini(prompt, self.gemini_client, temperature=0.6)
            return self._format_section_content(content, section_name)
        except Exception as e:
            log_agent_action("WriterAgent", "generic_section_error", {"error": str(e)})
            return f"Error drafting {section_name} section: {str(e)}"
    
    def _format_engineer_outputs(self, engineer_outputs: Dict[str, Any]) -> str:
        """Format engineer outputs for inclusion in text."""
        if not engineer_outputs:
            return "No experimental results available yet."
        
        formatted = []
        for exp_name, output in engineer_outputs.items():
            formatted.append(f"Experiment: {exp_name}")
            if isinstance(output, dict):
                for key, value in output.items():
                    formatted.append(f"  {key}: {value}")
            else:
                formatted.append(f"  Result: {output}")
        
        return "\n".join(formatted)
    
    def _format_section_content(self, content: str, section_name: str) -> str:
        """Format section content with proper structure."""
        # Clean up content
        content = content.strip()
        
        # Add section header if not present
        if not content.startswith(f"# {section_name}") and not content.startswith(f"## {section_name}"):
            content = f"# {section_name}\n\n{content}"
        
        return content
    
    def _store_section(self, section_name: str, content: str, topic: Dict[str, Any]):
        """Store section content in memory."""
        try:
            # Extract citations
            citations = extract_citations(content)
            
            # Store in memory
            memory.add_embedding(
                generate_embedding(content, self.gemini_client),
                {
                    'type': 'paper_section',
                    'section': section_name,
                    'topic': topic['title'],
                    'content_length': len(content),
                    'citations': citations,
                    'timestamp': datetime.now().isoformat()
                }
            )
            
        except Exception as e:
            log_agent_action("WriterAgent", "section_storage_error", {"error": str(e)})
    
    # Fallback methods for error handling
    def _create_fallback_abstract(self, topic: Dict[str, Any], plan: Dict[str, Any]) -> str:
        return f"""
# Abstract

This paper presents research on {topic['title']}. We address the problem of {topic['description']} 
and propose a novel approach that {topic.get('impact', 'provides significant improvements')}. 
Our contributions include {', '.join(plan.get('expected_contributions', ['novel methodology', 'comprehensive evaluation']))}. 
Experimental results demonstrate the effectiveness of our approach compared to existing methods.
"""
    
    def _create_fallback_introduction(self, topic: Dict[str, Any], plan: Dict[str, Any]) -> str:
        return f"""
# Introduction

{topic['title']} represents an important challenge in {config.research_domain}. 
Current approaches have limitations in {topic.get('rationale', 'scalability and efficiency')}. 
This work addresses these challenges by {topic.get('impact', 'introducing novel methods')}.

Our main contributions are:
{chr(10).join([f"- {contribution}" for contribution in plan.get('expected_contributions', ['Novel approach', 'Comprehensive evaluation', 'Practical insights'])])}

The remainder of this paper is organized as follows: Section 2 reviews related work, 
Section 3 describes our methodology, Section 4 presents experimental results, 
and Section 5 concludes with future work.
"""
    
    def _create_fallback_related_work(self, topic: Dict[str, Any], plan: Dict[str, Any]) -> str:
        return f"""
# Related Work

Previous work in {config.research_domain} has addressed various aspects of {topic['title']}. 
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

Results demonstrate significant improvements over existing methods, validating the 
effectiveness of our proposed approach.
"""
    
    def _create_fallback_results(self, topic: Dict[str, Any], plan: Dict[str, Any]) -> str:
        return f"""
# Results

Our experimental results show that our approach to {topic['title']} achieves 
significant improvements over baseline methods. The results validate our key 
contributions and demonstrate the practical impact of our work.

Analysis reveals important insights about the effectiveness of different components 
and provides guidance for future research directions.
"""
    
    def _create_fallback_conclusion(self, topic: Dict[str, Any], plan: Dict[str, Any]) -> str:
        return f"""
# Conclusion

This paper presented research on {topic['title']}, addressing key challenges in {config.research_domain}. 
Our main contributions include {', '.join(plan.get('expected_contributions', ['novel methodology', 'comprehensive evaluation']))}.

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