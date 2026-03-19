"""
PlannerAgent - Creates detailed project roadmaps for research papers.
Defines sections, experiments, dependencies, and agent assignments.
"""

import json
from typing import Dict, Any, List
from datetime import datetime

from core.config import config
from core.utils import setup_gemini, call_gemini, log_agent_action
from core.memory import memory

class PlannerAgent:
    """Agent for creating detailed research project plans."""
    
    def __init__(self):
        self.gemini_client = setup_gemini()
    
    def create_plan(self, topic: Dict[str, Any]) -> Dict[str, Any]:
        """Create a comprehensive research plan for the topic."""
        log_agent_action("PlannerAgent", "start_planning", {"topic": topic['title']})
        
        # Generate plan using LLM
        plan = self._generate_plan_structure(topic)
        
        # Add experiments
        plan['experiments'] = self._generate_experiments(topic, plan)
        
        # Add dependencies
        plan['dependencies'] = self._generate_dependencies(plan)
        
        # Add timeline
        plan['timeline'] = self._generate_timeline(plan)
        
        # Store plan in memory
        self._store_plan(plan, topic)
        
        log_agent_action("PlannerAgent", "plan_created", {
            "sections": len(plan.get('sections', [])),
            "experiments": len(plan.get('experiments', [])),
            "dependencies": len(plan.get('dependencies', []))
        })
        
        return plan
    
    def _generate_plan_structure(self, topic: Dict[str, Any]) -> Dict[str, Any]:
        """Generate the basic plan structure."""
        prompt = f"""
        Create a detailed research plan for the following topic:
        
        Topic: {topic['title']}
        Description: {topic['description']}
        Rationale: {topic.get('rationale', 'N/A')}
        Impact: {topic.get('impact', 'N/A')}
        Feasibility: {topic.get('feasibility', 5)}/10
        
        Create a comprehensive research plan with the following structure:
        
        1. Paper sections (Abstract, Introduction, Related Work, Methods, Experiments, Results, Discussion, Conclusion)
        2. For each section, specify:
           - Content requirements
           - Key points to cover
           - Expected length
           - Dependencies on other sections
        
        Format as JSON:
        {{
            "title": "Research paper title",
            "sections": [
                {{
                    "name": "Abstract",
                    "content_requirements": "Brief summary of the entire paper",
                    "key_points": ["Problem statement", "Methodology", "Results", "Impact"],
                    "expected_length": "150-250 words",
                    "dependencies": []
                }},
                {{
                    "name": "Introduction",
                    "content_requirements": "Motivation and problem statement",
                    "key_points": ["Background", "Problem definition", "Contributions", "Paper outline"],
                    "expected_length": "2-3 pages",
                    "dependencies": []
                }}
            ],
            "research_questions": [
                "Specific research questions to address"
            ],
            "methodology": "High-level approach description",
            "expected_contributions": [
                "List of expected contributions"
            ]
        }}
        """
        
        try:
            response = call_gemini(prompt, self.gemini_client, temperature=0.6)
            
            # Parse JSON response
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                
                plan = json.loads(json_str)
                
                # Add metadata
                plan['created_at'] = datetime.now().isoformat()
                plan['topic'] = topic['title']
                plan['domain'] = config.research_domain
                
                return plan
            else:
                # Fallback plan structure
                return self._create_fallback_plan(topic)
                
        except Exception as e:
            log_agent_action("PlannerAgent", "plan_generation_error", {"error": str(e)})
            return self._create_fallback_plan(topic)
    
    def _generate_experiments(self, topic: Dict[str, Any], plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate experiments and simulations for the research."""
        prompt = f"""
        Based on the research topic and plan, design specific experiments and simulations:
        
        Topic: {topic['title']}
        Methodology: {plan.get('methodology', 'N/A')}
        Research Questions: {plan.get('research_questions', [])}
        
        Design experiments that will:
        1. Validate the proposed approach
        2. Compare against baseline methods
        3. Demonstrate the contributions
        4. Provide quantitative results
        
        For each experiment, specify:
        - Name and purpose
        - Methodology
        - Expected outcomes
        - Code requirements
        - Data requirements
        - Evaluation metrics
        
        Format as JSON array:
        [
            {{
                "name": "Experiment name",
                "purpose": "What this experiment tests",
                "methodology": "How to conduct the experiment",
                "expected_outcomes": "What results to expect",
                "code_requirements": "Programming language, libraries, etc.",
                "data_requirements": "What data is needed",
                "evaluation_metrics": ["metric1", "metric2"],
                "baseline_comparison": "What to compare against"
            }}
        ]
        """
        
        try:
            response = call_gemini(prompt, self.gemini_client, temperature=0.7)
            
            if '[' in response and ']' in response:
                start = response.find('[')
                end = response.rfind(']') + 1
                json_str = response[start:end]
                
                experiments = json.loads(json_str)
                return experiments
            else:
                return self._create_fallback_experiments(topic)
                
        except Exception as e:
            log_agent_action("PlannerAgent", "experiment_generation_error", {"error": str(e)})
            return self._create_fallback_experiments(topic)
    
    def _generate_dependencies(self, plan: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Generate dependencies between plan components."""
        dependencies = []
        
        # Section dependencies
        sections = plan.get('sections', [])
        for i, section in enumerate(sections):
            if section['name'] in ['Methods', 'Experiments', 'Results']:
                # These sections depend on Introduction and Related Work
                dependencies.append({
                    'from': 'Introduction',
                    'to': section['name'],
                    'type': 'content_dependency',
                    'description': f'{section["name"]} builds on concepts introduced in Introduction'
                })
        
        # Experiment dependencies
        experiments = plan.get('experiments', [])
        for experiment in experiments:
            dependencies.append({
                'from': 'Methods',
                'to': experiment['name'],
                'type': 'methodology_dependency',
                'description': f'Experiment {experiment["name"]} uses methodology defined in Methods'
            })
        
        return dependencies
    
    def _generate_timeline(self, plan: Dict[str, Any]) -> Dict[str, Any]:
        """Generate a timeline for the research project."""
        timeline = {
            'phases': [
                {
                    'name': 'Planning and Literature Review',
                    'duration': '1-2 weeks',
                    'tasks': ['Literature review', 'Problem definition', 'Methodology design']
                },
                {
                    'name': 'Implementation',
                    'duration': '2-4 weeks',
                    'tasks': ['Code development', 'Experiment setup', 'Data preparation']
                },
                {
                    'name': 'Experimentation',
                    'duration': '1-2 weeks',
                    'tasks': ['Running experiments', 'Data collection', 'Preliminary analysis']
                },
                {
                    'name': 'Analysis and Writing',
                    'duration': '2-3 weeks',
                    'tasks': ['Data analysis', 'Paper writing', 'Results interpretation']
                },
                {
                    'name': 'Revision and Finalization',
                    'duration': '1 week',
                    'tasks': ['Paper revision', 'Final review', 'Submission preparation']
                }
            ],
            'total_duration': '7-12 weeks',
            'critical_path': ['Planning', 'Implementation', 'Experimentation', 'Writing']
        }
        
        return timeline
    
    def _store_plan(self, plan: Dict[str, Any], topic: Dict[str, Any]):
        """Store the plan in memory for future reference."""
        try:
            # Save plan to file
            plan_filename = f"plan_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            plan_path = f"{config.output_dir}/{plan_filename}"
            
            with open(plan_path, 'w') as f:
                json.dump(plan, f, indent=2)
            
            # Store in memory
            memory.add_embedding(
                generate_embedding(json.dumps(plan), self.gemini_client),
                {
                    'type': 'research_plan',
                    'topic': topic['title'],
                    'plan_file': plan_path,
                    'sections': len(plan.get('sections', [])),
                    'experiments': len(plan.get('experiments', []))
                }
            )
            
        except Exception as e:
            log_agent_action("PlannerAgent", "plan_storage_error", {"error": str(e)})
    
    def _create_fallback_plan(self, topic: Dict[str, Any]) -> Dict[str, Any]:
        """Create a fallback plan if LLM generation fails."""
        return {
            'title': f"Research on {topic['title']}",
            'sections': [
                {
                    'name': 'Abstract',
                    'content_requirements': 'Brief summary of the entire paper',
                    'key_points': ['Problem statement', 'Methodology', 'Results', 'Impact'],
                    'expected_length': '150-250 words',
                    'dependencies': []
                },
                {
                    'name': 'Introduction',
                    'content_requirements': 'Motivation and problem statement',
                    'key_points': ['Background', 'Problem definition', 'Contributions', 'Paper outline'],
                    'expected_length': '2-3 pages',
                    'dependencies': []
                },
                {
                    'name': 'Related Work',
                    'content_requirements': 'Literature review and background',
                    'key_points': ['Previous work', 'Gaps in literature', 'Our contribution'],
                    'expected_length': '2-3 pages',
                    'dependencies': ['Introduction']
                },
                {
                    'name': 'Methods',
                    'content_requirements': 'Detailed methodology description',
                    'key_points': ['Approach', 'Algorithm', 'Implementation details'],
                    'expected_length': '3-4 pages',
                    'dependencies': ['Related Work']
                },
                {
                    'name': 'Experiments',
                    'content_requirements': 'Experimental setup and results',
                    'key_points': ['Dataset', 'Baselines', 'Metrics', 'Results'],
                    'expected_length': '4-5 pages',
                    'dependencies': ['Methods']
                },
                {
                    'name': 'Results',
                    'content_requirements': 'Detailed analysis of results',
                    'key_points': ['Performance analysis', 'Ablation studies', 'Discussion'],
                    'expected_length': '3-4 pages',
                    'dependencies': ['Experiments']
                },
                {
                    'name': 'Conclusion',
                    'content_requirements': 'Summary and future work',
                    'key_points': ['Summary', 'Contributions', 'Future work'],
                    'expected_length': '1-2 pages',
                    'dependencies': ['Results']
                }
            ],
            'research_questions': [
                'How can we improve upon existing approaches?',
                'What are the key limitations of current methods?',
                'What novel contributions does our work make?'
            ],
            'methodology': 'Experimental evaluation with quantitative analysis',
            'expected_contributions': [
                'Novel approach to the problem',
                'Comprehensive experimental evaluation',
                'Insights into the research area'
            ],
            'created_at': datetime.now().isoformat(),
            'topic': topic['title'],
            'domain': config.research_domain
        }
    
    def _create_fallback_experiments(self, topic: Dict[str, Any]) -> List[Dict[str, Any]]:
        """Create fallback experiments if LLM generation fails."""
        return [
            {
                'name': 'Baseline Comparison',
                'purpose': 'Compare our approach against existing methods',
                'methodology': 'Implement baseline methods and compare performance',
                'expected_outcomes': 'Quantitative performance metrics',
                'code_requirements': 'Python, scikit-learn, numpy',
                'data_requirements': 'Standard benchmark dataset',
                'evaluation_metrics': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
                'baseline_comparison': 'State-of-the-art methods in the field'
            },
            {
                'name': 'Ablation Study',
                'purpose': 'Analyze the contribution of different components',
                'methodology': 'Remove components one by one and measure impact',
                'expected_outcomes': 'Understanding of component importance',
                'code_requirements': 'Python, modular implementation',
                'data_requirements': 'Same dataset as main experiment',
                'evaluation_metrics': ['Performance degradation', 'Component importance'],
                'baseline_comparison': 'Full system vs. ablated versions'
            }
        ]

# Example usage
if __name__ == "__main__":
    planner = PlannerAgent()
    
    example_topic = {
        'title': 'Novel Attention Mechanisms for Transformer Models',
        'description': 'Developing new attention mechanisms that improve efficiency and interpretability',
        'rationale': 'Current attention mechanisms have limitations in scalability',
        'impact': 'Could enable larger, more efficient language models',
        'feasibility': 8
    }
    
    plan = planner.create_plan(example_topic)
    print(f"Created plan with {len(plan['sections'])} sections and {len(plan['experiments'])} experiments") 