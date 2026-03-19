"""
MetaAgent - Self-reflection and oversight for the research system.
Monitors performance, detects stuck loops, and makes strategic decisions.
"""

import json
from typing import Dict, Any, List
from datetime import datetime

from core.config import config
from core.utils import setup_gemini, call_gemini, log_agent_action
from core.memory import memory

class MetaAgent:
    """Agent for system-wide oversight and strategic decision making."""
    
    def __init__(self):
        self.gemini_client = setup_gemini()
    
    def evaluate_system_performance(self, state) -> str:
        """Evaluate overall system performance and provide feedback."""
        log_agent_action("MetaAgent", "start_evaluation", {})
        
        try:
            # Gather performance metrics
            metrics = self._gather_performance_metrics(state)
            
            # Analyze trends
            trends = self._analyze_trends(state)
            
            # Generate feedback
            feedback = self._generate_performance_feedback(metrics, trends)
            
            log_agent_action("MetaAgent", "evaluation_complete", {
                "metrics": metrics,
                "trends": trends
            })
            
            return feedback
            
        except Exception as e:
            log_agent_action("MetaAgent", "evaluation_error", {"error": str(e)})
            return f"Meta evaluation error: {str(e)}"
    
    def should_reset(self, state) -> bool:
        """Determine if the system should reset and start over."""
        # Check for stuck loops
        if self._detect_stuck_loop(state):
            return True
        
        # Check for low-quality research
        if self._detect_low_quality_research(state):
            return True
        
        # Check for excessive iterations
        if state["iteration"] >= config.max_iterations:
            return True
        
        # Check for no progress
        if self._detect_no_progress(state):
            return True
        
        return False
    
    def should_continue(self, state) -> bool:
        """Determine if the system should continue with revisions."""
        # Check if we have a good foundation
        if not state["selected_topic"]:
            return False
        
        # Check if we have a plan
        if not state["plan"]:
            return False
        
        # Check if we have some content
        if not state["draft_sections"]:
            return False
        
        # Check if scores are improving
        if self._scores_improving(state):
            return True
        
        # Check if we're close to threshold
        if self._close_to_threshold(state):
            return True
        
        return False
    
    def _gather_performance_metrics(self, state) -> Dict[str, Any]:
        """Gather comprehensive performance metrics."""
        metrics = {
            'iteration': state["iteration"],
            'topics_discovered': len(state["topics"]),
            'debates_completed': len(state["debate_results"]),
            'sections_written': len(state["draft_sections"]),
            'experiments_run': len(state["engineer_outputs"]),
            'average_score': 0.0,
            'best_score': 0.0,
            'progress_made': False
        }
        
        # Calculate average supervisor score
        if state["supervisor_scores"]:
            scores = list(state["supervisor_scores"].values())
            metrics['average_score'] = sum(scores) / len(scores)
            metrics['best_score'] = max(scores)
        
        # Check if we have a selected topic
        metrics['has_topic'] = state["selected_topic"] is not None
        
        # Check if we have a plan
        metrics['has_plan'] = state["plan"] is not None
        
        # Check if we have content
        metrics['has_content'] = len(state["draft_sections"]) > 0
        
        # Check if we have experiments
        metrics['has_experiments'] = len(state["engineer_outputs"]) > 0
        
        return metrics
    
    def _analyze_trends(self, state) -> Dict[str, Any]:
        """Analyze trends in system performance."""
        trends = {
            'score_trend': 'stable',
            'content_growth': 'stable',
            'quality_improvement': 'stable',
            'stuck_pattern': False
        }
        
        # Get recent feedback
        recent_feedback = memory.get_recent_feedback(limit=10)
        
        if len(recent_feedback) >= 2:
            # Analyze score trends
            scores = [entry['score'] for entry in recent_feedback]
            if len(scores) >= 3:
                recent_avg = sum(scores[-3:]) / 3
                older_avg = sum(scores[:-3]) / (len(scores) - 3) if len(scores) > 3 else scores[0]
                
                if recent_avg > older_avg + 0.5:
                    trends['score_trend'] = 'improving'
                elif recent_avg < older_avg - 0.5:
                    trends['score_trend'] = 'declining'
                else:
                    trends['score_trend'] = 'stable'
            
            # Check for stuck patterns (same scores repeatedly)
            if len(scores) >= 3:
                recent_scores = scores[-3:]
                if len(set(recent_scores)) <= 1:  # All same score
                    trends['stuck_pattern'] = True
        
        # Analyze content growth
        if state["draft_sections"]:
            total_content_length = sum(len(content) for content in state["draft_sections"].values())
            if total_content_length > 5000:  # Arbitrary threshold
                trends['content_growth'] = 'good'
            elif total_content_length > 1000:
                trends['content_growth'] = 'moderate'
            else:
                trends['content_growth'] = 'poor'
        
        return trends
    
    def _generate_performance_feedback(self, metrics: Dict[str, Any], trends: Dict[str, Any]) -> str:
        """Generate comprehensive performance feedback."""
        prompt = f"""
        Analyze the following research system performance metrics and provide strategic feedback:
        
        Performance Metrics:
        {json.dumps(metrics, indent=2)}
        
        Trends:
        {json.dumps(trends, indent=2)}
        
        Provide feedback on:
        1. Overall system health
        2. Quality of research direction
        3. Progress assessment
        4. Recommendations for improvement
        5. Strategic next steps
        
        Format as JSON:
        {{
            "system_health": "good/medium/poor",
            "research_quality": "high/medium/low",
            "progress_assessment": "excellent/good/fair/poor",
            "recommendations": ["list", "of", "recommendations"],
            "next_steps": "continue/reset/refine"
        }}
        """
        
        try:
            response = call_gemini(prompt, self.gemini_client, temperature=0.4)
            
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                
                feedback_data = json.loads(json_str)
                
                # Format feedback as readable text
                feedback = f"""
System Performance Analysis:
- System Health: {feedback_data.get('system_health', 'unknown')}
- Research Quality: {feedback_data.get('research_quality', 'unknown')}
- Progress Assessment: {feedback_data.get('progress_assessment', 'unknown')}

Recommendations:
{chr(10).join([f"- {rec}" for rec in feedback_data.get('recommendations', [])])}

Next Steps: {feedback_data.get('next_steps', 'continue')}
"""
                
                return feedback.strip()
            else:
                return "Could not parse performance feedback."
                
        except Exception as e:
            return f"Performance feedback error: {str(e)}"
    
    def _detect_stuck_loop(self, state) -> bool:
        """Detect if the system is stuck in a low-quality loop."""
        # Check recent feedback for stuck patterns
        recent_feedback = memory.get_recent_feedback(limit=5)
        
        if len(recent_feedback) >= 5:  # Require more feedback before detecting stuck loop
            scores = [entry['score'] for entry in recent_feedback]
            
            # Check if scores are consistently very low (below 3.0)
            if all(score < 3.0 for score in scores):
                return True
            
            # Check if scores are not improving over multiple iterations
            if len(scores) >= 5:
                recent_avg = sum(scores[-3:]) / 3
                older_avg = sum(scores[:-3]) / (len(scores) - 3) if len(scores) > 3 else scores[0]
                
                # Only reset if scores are significantly declining
                if recent_avg < older_avg - 1.0:  # Significant decline
                    return True
        
        return False
    
    def _detect_low_quality_research(self, state) -> bool:
        """Detect if the research direction is fundamentally flawed."""
        # Check if we have a topic
        if not state["selected_topic"]:
            return True
        
        # Check if topic has reasonable feasibility
        if state["selected_topic"].get('feasibility', 5) < 3:
            return True
        
        # Check if we have any successful experiments
        successful_experiments = sum(1 for output in state["engineer_outputs"].values() 
                                   if output.get('success', False))
        
        if len(state["engineer_outputs"]) > 0 and successful_experiments == 0:
            return True
        
        return False
    
    def _detect_no_progress(self, state) -> bool:
        """Detect if the system is making no meaningful progress."""
        # Check if we have any content
        if not state["draft_sections"]:
            return True
        
        # Check if content is too short (but be more lenient)
        total_content_length = sum(len(content) for content in state["draft_sections"].values())
        if total_content_length < 200:  # More lenient threshold
            return True
        
        # Check if scores are consistently very low
        if state["supervisor_scores"]:
            avg_score = sum(state["supervisor_scores"].values()) / len(state["supervisor_scores"])
            if avg_score < 2.0:  # Very low scores threshold
                return True
        
        return False
    
    def _scores_improving(self, state) -> bool:
        """Check if supervisor scores are improving."""
        if not state["supervisor_scores"]:
            return False
        
        # Get recent feedback
        recent_feedback = memory.get_recent_feedback(limit=5)
        
        if len(recent_feedback) >= 2:
            scores = [entry['score'] for entry in recent_feedback]
            if len(scores) >= 3:
                recent_avg = sum(scores[-3:]) / 3
                older_avg = sum(scores[:-3]) / (len(scores) - 3) if len(scores) > 3 else scores[0]
                
                return recent_avg > older_avg
        
        return False
    
    def _close_to_threshold(self, state) -> bool:
        """Check if we're close to the quality threshold."""
        if not state["supervisor_scores"]:
            return False
        
        avg_score = sum(state["supervisor_scores"].values()) / len(state["supervisor_scores"])
        threshold = config.supervisor_threshold
        
        # Consider "close" if within 1.0 of threshold
        return avg_score >= (threshold - 1.0)
    
    def get_system_recommendations(self, state) -> List[str]:
        """Get specific recommendations for system improvement."""
        recommendations = []
        
        # Check for common issues
        if not state["selected_topic"]:
            recommendations.append("No research topic selected - need to discover topics")
        
        if not state["plan"]:
            recommendations.append("No research plan created - need to create detailed plan")
        
        if not state["draft_sections"]:
            recommendations.append("No content written - need to draft paper sections")
        
        if state["supervisor_scores"]:
            avg_score = sum(state["supervisor_scores"].values()) / len(state["supervisor_scores"])
            if avg_score < config.supervisor_threshold:
                recommendations.append(f"Quality below threshold ({avg_score:.1f}/{config.supervisor_threshold}) - need revisions")
        
        if state["iteration"] >= config.max_iterations // 2:
            recommendations.append("Approaching iteration limit - consider reset if no improvement")
        
        # Check for stuck patterns
        recent_feedback = memory.get_recent_feedback(limit=3)
        if len(recent_feedback) >= 3:
            scores = [entry['score'] for entry in recent_feedback]
            if len(set(scores)) <= 1:  # All same score
                recommendations.append("Scores not improving - consider changing approach")
        
        return recommendations

# Example usage
if __name__ == "__main__":
    meta_agent = MetaAgent()
    
    # Mock state for testing
    class MockState:
        def __init__(self):
            self.iteration = 2
            self.topics = [{'title': 'Test Topic'}]
            self.selected_topic = {'title': 'Test Topic', 'feasibility': 7}
            self.plan = {'sections': []}
            self.draft_sections = {'Introduction': 'Some content'}
            self.engineer_outputs = {'exp1': {'success': True}}
            self.supervisor_scores = {'Introduction': 7.5}
            self.meta_feedback = []
    
    mock_state = MockState()
    
    feedback = meta_agent.evaluate_system_performance(mock_state)
    print("Performance Feedback:")
    print(feedback)
    
    should_reset = meta_agent.should_reset(mock_state)
    should_continue = meta_agent.should_continue(mock_state)
    
    print(f"\nShould reset: {should_reset}")
    print(f"Should continue: {should_continue}")
    
    recommendations = meta_agent.get_system_recommendations(mock_state)
    print(f"\nRecommendations: {recommendations}") 