"""
SupervisorAgent - Multi-layered quality checking for research papers.
Includes hallucination detection, math validation, code verification, and peer review simulation.
"""

import json
import re
from typing import Dict, Any, Tuple, List
import sympy as sp

from core.config import config
from core.utils import setup_gemini, call_gemini, log_agent_action, validate_math_expression
from core.memory import memory

class SupervisorAgent:
    """Agent for comprehensive quality checking of research content."""
    
    def __init__(self):
        self.gemini_client = setup_gemini()
        self.sub_agents = {
            'hallucination_checker': HallucinationChecker(self.gemini_client),
            'math_checker': MathChecker(),
            'code_checker': CodeChecker(),
            'reviewer_bot': ReviewerBot(self.gemini_client)
        }
    
    def evaluate_section(self, section_name: str, content: str) -> Tuple[float, str]:
        """Evaluate a section and return score and feedback."""
        log_agent_action("SupervisorAgent", "start_evaluation", {"section": section_name})
        
        scores = {}
        feedbacks = []
        
        # Run all sub-agents
        for agent_name, agent in self.sub_agents.items():
            try:
                score, feedback = agent.evaluate(content, section_name)
                scores[agent_name] = score
                feedbacks.append(f"{agent_name}: {feedback}")
                
                log_agent_action("SupervisorAgent", f"{agent_name}_complete", {
                    "section": section_name,
                    "score": score
                })
                
            except Exception as e:
                log_agent_action("SupervisorAgent", f"{agent_name}_error", {"error": str(e)})
                scores[agent_name] = 5.0  # Neutral score on error
                feedbacks.append(f"{agent_name}: Error - {str(e)}")
        
        # Calculate overall score (weighted average)
        overall_score = self._calculate_overall_score(scores)
        overall_feedback = "\n".join(feedbacks)
        
        # Store evaluation in memory
        memory.add_feedback_entry(
            "SupervisorAgent",
            section_name,
            overall_score,
            overall_feedback,
            1  # iteration
        )
        
        log_agent_action("SupervisorAgent", "evaluation_complete", {
            "section": section_name,
            "overall_score": overall_score,
            "sub_scores": scores
        })
        
        return overall_score, overall_feedback
    
    def _calculate_overall_score(self, scores: Dict[str, float]) -> float:
        """Calculate weighted overall score from sub-agent scores."""
        weights = {
            'hallucination_checker': 0.3,
            'math_checker': 0.2,
            'code_checker': 0.2,
            'reviewer_bot': 0.3
        }
        
        weighted_sum = 0
        total_weight = 0
        
        for agent_name, score in scores.items():
            weight = weights.get(agent_name, 0.25)
            weighted_sum += score * weight
            total_weight += weight
        
        return weighted_sum / total_weight if total_weight > 0 else 5.0

class HallucinationChecker:
    """Sub-agent for detecting hallucinations and invalid claims."""
    
    def __init__(self, gemini_client):
        self.gemini_client = gemini_client
    
    def evaluate(self, content: str, section_name: str) -> Tuple[float, str]:
        """Check for hallucinations and invalid claims."""
        prompt = f"""
        Analyze the following research content for potential hallucinations, invalid claims, or unsupported statements:
        
        Section: {section_name}
        Content: {content}
        
        Look for:
        1. Claims without proper citations
        2. Factual inaccuracies
        3. Unsupported generalizations
        4. Contradictory statements
        5. Claims that seem too good to be true
        
        Rate the content on a scale of 1-10 where:
        10 = No hallucinations, all claims properly supported
        5 = Some minor issues, mostly accurate
        1 = Multiple serious hallucinations or unsupported claims
        
        Provide:
        1. A score (1-10)
        2. Specific issues found (if any)
        3. Recommendations for improvement
        
        Format as JSON:
        {{
            "score": 8,
            "issues": ["List of specific issues"],
            "recommendations": ["List of recommendations"]
        }}
        """
        
        try:
            response = call_gemini(prompt, self.gemini_client, temperature=0.3)
            
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                
                result = json.loads(json_str)
                score = result.get('score', 5.0)
                issues = result.get('issues', [])
                recommendations = result.get('recommendations', [])
                
                feedback = f"Score: {score}/10"
                if issues:
                    feedback += f"\nIssues: {', '.join(issues)}"
                if recommendations:
                    feedback += f"\nRecommendations: {', '.join(recommendations)}"
                
                return score, feedback
            else:
                return 5.0, "Could not parse hallucination check response"
                
        except Exception as e:
            return 5.0, f"Hallucination check error: {str(e)}"

class MathChecker:
    """Sub-agent for validating mathematical expressions and equations."""
    
    def __init__(self):
        pass
    
    def evaluate(self, content: str, section_name: str) -> Tuple[float, str]:
        """Check mathematical expressions for validity."""
        # Extract mathematical expressions
        math_expressions = self._extract_math_expressions(content)
        
        if not math_expressions:
            return 10.0, "No mathematical expressions found"
        
        valid_expressions = 0
        invalid_expressions = []
        
        for expr in math_expressions:
            if validate_math_expression(expr):
                valid_expressions += 1
            else:
                invalid_expressions.append(expr)
        
        total_expressions = len(math_expressions)
        score = (valid_expressions / total_expressions) * 10 if total_expressions > 0 else 10.0
        
        feedback = f"Math validation: {valid_expressions}/{total_expressions} expressions valid"
        if invalid_expressions:
            feedback += f"\nInvalid expressions: {', '.join(invalid_expressions[:3])}"
        
        return score, feedback
    
    def _extract_math_expressions(self, content: str) -> List[str]:
        """Extract mathematical expressions from text."""
        # Common patterns for mathematical expressions
        patterns = [
            r'\$([^$]+)\$',  # LaTeX inline math
            r'\\\[([^\]]+)\\\]',  # LaTeX display math
            r'\\begin\{equation\}(.*?)\\end\{equation\}',  # LaTeX equation environment
            r'([a-zA-Z_][a-zA-Z0-9_]*\s*[+\-*/=<>]\s*[a-zA-Z0-9_]+)',  # Simple expressions
        ]
        
        expressions = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            expressions.extend(matches)
        
        return list(set(expressions))  # Remove duplicates

class CodeChecker:
    """Sub-agent for verifying code snippets and algorithms."""
    
    def __init__(self):
        pass
    
    def evaluate(self, content: str, section_name: str) -> Tuple[float, str]:
        """Check code snippets for syntax and logic."""
        # Extract code snippets
        code_snippets = self._extract_code_snippets(content)
        
        if not code_snippets:
            return 10.0, "No code snippets found"
        
        valid_snippets = 0
        invalid_snippets = []
        
        for snippet in code_snippets:
            if self._validate_code_snippet(snippet):
                valid_snippets += 1
            else:
                invalid_snippets.append(snippet[:50] + "...")
        
        total_snippets = len(code_snippets)
        score = (valid_snippets / total_snippets) * 10 if total_snippets > 0 else 10.0
        
        feedback = f"Code validation: {valid_snippets}/{total_snippets} snippets valid"
        if invalid_snippets:
            feedback += f"\nInvalid snippets: {', '.join(invalid_snippets[:3])}"
        
        return score, feedback
    
    def _extract_code_snippets(self, content: str) -> List[str]:
        """Extract code snippets from text."""
        # Common code block patterns
        patterns = [
            r'```[\w]*\n(.*?)\n```',  # Markdown code blocks
            r'`([^`]+)`',  # Inline code
            r'\\begin\{verbatim\}(.*?)\\end\{verbatim\}',  # LaTeX verbatim
            r'\\begin\{lstlisting\}(.*?)\\end\{lstlisting\}',  # LaTeX listings
        ]
        
        snippets = []
        for pattern in patterns:
            matches = re.findall(pattern, content, re.DOTALL)
            snippets.extend(matches)
        
        return list(set(snippets))
    
    def _validate_code_snippet(self, snippet: str) -> bool:
        """Basic validation of code snippet syntax."""
        try:
            # Basic checks for common programming languages
            if 'def ' in snippet or 'import ' in snippet or 'class ' in snippet:
                # Python-like code
                return self._validate_python_syntax(snippet)
            elif 'function ' in snippet or 'var ' in snippet or 'const ' in snippet:
                # JavaScript-like code
                return self._validate_js_syntax(snippet)
            elif 'public ' in snippet or 'private ' in snippet or 'class ' in snippet:
                # Java-like code
                return self._validate_java_syntax(snippet)
            else:
                # Generic validation
                return self._validate_generic_syntax(snippet)
        except Exception:
            return False
    
    def _validate_python_syntax(self, snippet: str) -> bool:
        """Validate Python syntax."""
        try:
            compile(snippet, '<string>', 'exec')
            return True
        except SyntaxError:
            return False
    
    def _validate_js_syntax(self, snippet: str) -> bool:
        """Basic JavaScript syntax validation."""
        # Check for balanced braces and parentheses
        return self._check_balanced_delimiters(snippet)
    
    def _validate_java_syntax(self, snippet: str) -> bool:
        """Basic Java syntax validation."""
        # Check for balanced braces and semicolons
        return self._check_balanced_delimiters(snippet)
    
    def _validate_generic_syntax(self, snippet: str) -> bool:
        """Generic syntax validation."""
        return self._check_balanced_delimiters(snippet)
    
    def _check_balanced_delimiters(self, text: str) -> bool:
        """Check if delimiters are balanced."""
        stack = []
        pairs = {')': '(', '}': '{', ']': '['}
        
        for char in text:
            if char in '({[':
                stack.append(char)
            elif char in ')}]':
                if not stack or stack.pop() != pairs[char]:
                    return False
        
        return len(stack) == 0

class ReviewerBot:
    """Sub-agent that simulates peer review evaluation."""
    
    def __init__(self, gemini_client):
        self.gemini_client = gemini_client
    
    def evaluate(self, content: str, section_name: str) -> Tuple[float, str]:
        """Simulate peer review evaluation."""
        prompt = f"""
        You are a peer reviewer evaluating a research paper section. Review the following content:
        
        Section: {section_name}
        Content: {content}
        
        Evaluate based on these criteria (1-10 scale each):
        1. Clarity and readability
        2. Technical accuracy
        3. Logical flow and organization
        4. Completeness of coverage
        5. Academic writing standards
        
        Provide:
        1. Individual scores for each criterion
        2. Overall score (average)
        3. Specific feedback and suggestions
        4. Major strengths and weaknesses
        
        Format as JSON:
        {{
            "scores": {{
                "clarity": 8,
                "accuracy": 7,
                "flow": 8,
                "completeness": 6,
                "writing": 7
            }},
            "overall_score": 7.2,
            "strengths": ["List of strengths"],
            "weaknesses": ["List of weaknesses"],
            "suggestions": ["List of suggestions"]
        }}
        """
        
        try:
            response = call_gemini(prompt, self.gemini_client, temperature=0.4)
            
            if '{' in response and '}' in response:
                start = response.find('{')
                end = response.rfind('}') + 1
                json_str = response[start:end]
                
                result = json.loads(json_str)
                overall_score = result.get('overall_score', 5.0)
                strengths = result.get('strengths', [])
                weaknesses = result.get('weaknesses', [])
                suggestions = result.get('suggestions', [])
                
                feedback = f"Peer review score: {overall_score}/10"
                if strengths:
                    feedback += f"\nStrengths: {', '.join(strengths[:2])}"
                if weaknesses:
                    feedback += f"\nWeaknesses: {', '.join(weaknesses[:2])}"
                if suggestions:
                    feedback += f"\nSuggestions: {', '.join(suggestions[:2])}"
                
                return overall_score, feedback
            else:
                return 5.0, "Could not parse peer review response"
                
        except Exception as e:
            return 5.0, f"Peer review error: {str(e)}"

# Example usage
if __name__ == "__main__":
    supervisor = SupervisorAgent()
    
    example_content = """
    # Introduction
    
    Machine learning has revolutionized many fields. Our approach achieves 99.9% accuracy,
    which is significantly better than previous methods. The mathematical formulation
    is given by: $f(x) = \sum_{i=1}^n w_i x_i + b$.
    
    ```python
    def train_model(X, y):
        model = LinearRegression()
        model.fit(X, y)
        return model
    ```
    """
    
    score, feedback = supervisor.evaluate_section("Introduction", example_content)
    print(f"Score: {score}/10")
    print(f"Feedback: {feedback}") 