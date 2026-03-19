"""
Hypothesis Debate System - Multi-agent debate for validating research hypotheses.
Includes ProposerAgent, ChallengerAgent, and ModeratorAgent.
"""

import json
from typing import Dict, Any, List, Tuple
from dataclasses import dataclass

from core.config import config
from core.utils import setup_gemini, call_gemini, log_agent_action
from core.memory import memory

@dataclass
class DebateResult:
    """Result of a hypothesis debate."""
    topic: str
    proposer_argument: str
    challenger_argument: str
    moderator_decision: str
    score: float
    passed: bool
    reasoning: str

class ProposerAgent:
    """Agent that builds the case for a hypothesis."""
    
    def __init__(self):
        self.gemini_client = setup_gemini()
    
    def build_argument(self, topic: Dict[str, Any]) -> str:
        """Build a compelling argument for the hypothesis."""
        prompt = f"""
        You are a research proposer. Build a compelling argument for the following research topic:
        
        Topic: {topic['title']}
        Description: {topic['description']}
        Rationale: {topic.get('rationale', 'N/A')}
        Impact: {topic.get('impact', 'N/A')}
        Feasibility: {topic.get('feasibility', 5)}/10
        
        Create a structured argument that includes:
        1. Clear hypothesis statement
        2. Theoretical foundation
        3. Evidence from existing literature
        4. Novel contribution
        5. Methodology outline
        6. Expected outcomes
        
        Be persuasive but realistic. Focus on the scientific merit and potential impact.
        """
        
        try:
            argument = call_gemini(prompt, self.gemini_client, temperature=0.7)
            log_agent_action("ProposerAgent", "built_argument", {
                "topic": topic['title'],
                "argument_length": len(argument)
            })
            return argument
        except Exception as e:
            log_agent_action("ProposerAgent", "argument_error", {"error": str(e)})
            return "Failed to build argument."

class ChallengerAgent:
    """Agent that constructs rebuttals and identifies limitations."""
    
    def __init__(self):
        self.gemini_client = setup_gemini()
    
    def build_rebuttal(self, topic: Dict[str, Any], proposer_argument: str) -> str:
        """Build a rebuttal to the proposer's argument."""
        prompt = f"""
        You are a research challenger. Analyze the following topic and argument critically:
        
        Topic: {topic['title']}
        Description: {topic['description']}
        Proposer's Argument: {proposer_argument}
        
        Construct a thoughtful rebuttal that considers:
        1. Potential limitations and weaknesses
        2. Alternative explanations or approaches
        3. Methodological concerns
        4. Feasibility issues
        5. Potential conflicts with existing research
        6. Ethical or practical considerations
        
        Be constructive but critical. Identify genuine concerns that should be addressed.
        Focus on scientific rigor and potential pitfalls.
        """
        
        try:
            rebuttal = call_gemini(prompt, self.gemini_client, temperature=0.6)
            log_agent_action("ChallengerAgent", "built_rebuttal", {
                "topic": topic['title'],
                "rebuttal_length": len(rebuttal)
            })
            return rebuttal
        except Exception as e:
            log_agent_action("ChallengerAgent", "rebuttal_error", {"error": str(e)})
            return "Failed to build rebuttal."

class ModeratorAgent:
    """Agent that scores arguments and makes final decisions."""
    
    def __init__(self):
        self.gemini_client = setup_gemini()
    
    def evaluate_debate(self, topic: Dict[str, Any], proposer_argument: str, 
                       challenger_argument: str) -> Tuple[str, float, str]:
        """Evaluate the debate and make a decision."""
        prompt = f"""
        You are a research moderator evaluating a hypothesis debate.
        
        Topic: {topic['title']}
        Description: {topic['description']}
        
        PROPOSER ARGUMENT:
        {proposer_argument}
        
        CHALLENGER REBUTTAL:
        {challenger_argument}
        
        Evaluate this debate on the following criteria (1-10 scale each):
        1. Logical consistency of proposer's argument
        2. Strength of evidence presented
        3. Novelty and originality
        4. Feasibility of the proposed research
        5. Quality of challenger's critique
        6. Overall scientific merit
        
        Provide:
        1. Individual scores for each criterion
        2. Overall score (average of all criteria)
        3. Decision: "PASS" if overall score >= 7.5, "FAIL" otherwise
        4. Detailed reasoning for your decision
        
        Format your response as JSON:
        {{
            "scores": {{
                "logical_consistency": 8,
                "evidence_strength": 7,
                "novelty": 8,
                "feasibility": 6,
                "critique_quality": 7,
                "scientific_merit": 7
            }},
            "overall_score": 7.2,
            "decision": "PASS",
            "reasoning": "Detailed explanation of the decision..."
        }}
        """
        
        try:
            response = call_gemini(prompt, self.gemini_client, temperature=0.3)
            
            # Parse JSON response
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                
                evaluation = json.loads(json_str)
                
                decision = evaluation.get('decision', 'FAIL')
                overall_score = evaluation.get('overall_score', 0.0)
                reasoning = evaluation.get('reasoning', 'No reasoning provided.')
                
                log_agent_action("ModeratorAgent", "evaluated_debate", {
                    "topic": topic['title'],
                    "decision": decision,
                    "score": overall_score
                })
                
                return decision, overall_score, reasoning
            else:
                log_agent_action("ModeratorAgent", "parse_error", {"response": response})
                return "FAIL", 0.0, "Failed to parse evaluation."
                
        except Exception as e:
            log_agent_action("ModeratorAgent", "evaluation_error", {"error": str(e)})
            return "FAIL", 0.0, f"Evaluation error: {str(e)}"

class HypothesisDebateSystem:
    """Main system for coordinating hypothesis debates."""
    
    def __init__(self):
        self.proposer = ProposerAgent()
        self.challenger = ChallengerAgent()
        self.moderator = ModeratorAgent()
    
    def conduct_debate(self, topic: Dict[str, Any]) -> DebateResult:
        """Conduct a complete debate for a research topic."""
        log_agent_action("DebateSystem", "start_debate", {"topic": topic['title']})
        
        # Proposer builds argument
        proposer_argument = self.proposer.build_argument(topic)
        
        # Challenger builds rebuttal
        challenger_argument = self.challenger.build_rebuttal(topic, proposer_argument)
        
        # Moderator evaluates
        decision, score, reasoning = self.moderator.evaluate_debate(
            topic, proposer_argument, challenger_argument
        )
        
        # Create result
        result = DebateResult(
            topic=topic['title'],
            proposer_argument=proposer_argument,
            challenger_argument=challenger_argument,
            moderator_decision=decision,
            score=score,
            passed=decision == "PASS",
            reasoning=reasoning
        )
        
        # Store in memory
        memory.add_debate_entry(
            topic['title'],
            proposer_argument,
            challenger_argument,
            decision,
            score
        )
        
        log_agent_action("DebateSystem", "debate_complete", {
            "topic": topic['title'],
            "passed": result.passed,
            "score": score
        })
        
        return result
    
    def debate_multiple_topics(self, topics: List[Dict[str, Any]], 
                             max_topics: int = 3) -> List[DebateResult]:
        """Debate multiple topics until one passes or all are exhausted."""
        results = []
        
        for i, topic in enumerate(topics[:max_topics]):
            log_agent_action("DebateSystem", "debating_topic", {
                "topic_index": i,
                "topic": topic['title']
            })
            
            result = self.conduct_debate(topic)
            results.append(result)
            
            if result.passed:
                log_agent_action("DebateSystem", "found_passing_topic", {
                    "topic": topic['title'],
                    "score": result.score
                })
                break
        
        return results

# Example usage
if __name__ == "__main__":
    # Example topic
    example_topic = {
        "title": "Novel Attention Mechanisms for Transformer Models",
        "description": "Developing new attention mechanisms that improve efficiency and interpretability",
        "rationale": "Current attention mechanisms have limitations in scalability",
        "impact": "Could enable larger, more efficient language models",
        "feasibility": 8
    }
    
    debate_system = HypothesisDebateSystem()
    result = debate_system.conduct_debate(example_topic)
    
    print(f"Topic: {result.topic}")
    print(f"Decision: {result.moderator_decision}")
    print(f"Score: {result.score}")
    print(f"Passed: {result.passed}")
    print(f"Reasoning: {result.reasoning}") 