"""
EngineerAgent - Runs experiments, simulations, and code blocks.
Validates reproducibility and generates outputs for research papers.
"""

import json
import subprocess
import tempfile
import os
from typing import Dict, Any, List, Optional
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from datetime import datetime

from core.config import config
from core.utils import setup_gemini, call_gemini, log_agent_action, generate_embedding
from core.memory import memory

class EngineerAgent:
    """Agent for running experiments and generating code outputs."""
    
    def __init__(self):
        self.gemini_client = setup_gemini()
        self.output_dir = config.output_dir
        os.makedirs(self.output_dir, exist_ok=True)
    
    def run_experiment(self, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Run a specific experiment and return results."""
        log_agent_action("EngineerAgent", "start_experiment", {"experiment": experiment['name']})
        
        try:
            # Generate code for the experiment
            code = self._generate_experiment_code(experiment)
            
            # Execute the experiment
            results = self._execute_experiment(code, experiment)
            
            # Validate results
            validation = self._validate_results(results, experiment)
            
            # Generate visualizations if needed
            visualizations = self._generate_visualizations(results, experiment)
            
            # Compile final output
            output = {
                'experiment_name': experiment['name'],
                'code': code,
                'results': results,
                'validation': validation,
                'visualizations': visualizations,
                'success': validation['is_valid'],
                'timestamp': str(datetime.now())
            }
            
            # Store in memory
            self._store_experiment_results(output, experiment)
            
            log_agent_action("EngineerAgent", "experiment_complete", {
                "experiment": experiment['name'],
                "success": output['success']
            })
            
            return output
            
        except Exception as e:
            log_agent_action("EngineerAgent", "experiment_error", {
                "experiment": experiment['name'],
                "error": str(e)
            })
            
            return {
                'experiment_name': experiment['name'],
                'error': str(e),
                'success': False,
                'timestamp': str(datetime.now())
            }
    
    def _generate_experiment_code(self, experiment: Dict[str, Any]) -> str:
        """Generate Python code for the experiment."""
        prompt = f"""
        Generate Python code for the following experiment:
        
        Experiment: {experiment['name']}
        Purpose: {experiment['purpose']}
        Methodology: {experiment['methodology']}
        Code Requirements: {experiment['code_requirements']}
        Data Requirements: {experiment['data_requirements']}
        Evaluation Metrics: {experiment['evaluation_metrics']}
        
        The code should:
        1. Be complete and runnable
        2. Include necessary imports
        3. Generate synthetic data if needed
        4. Implement the methodology
        5. Calculate evaluation metrics
        6. Return results in a structured format
        
        Return only the Python code, no explanations.
        """
        
        try:
            code = call_gemini(prompt, self.gemini_client, temperature=0.3)
            return self._clean_code(code)
        except Exception as e:
            log_agent_action("EngineerAgent", "code_generation_error", {"error": str(e)})
            return self._generate_fallback_code(experiment)
    
    def _execute_experiment(self, code: str, experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Execute the experiment code and capture results."""
        try:
            # Create temporary file
            with tempfile.NamedTemporaryFile(mode='w', suffix='.py', delete=False) as f:
                f.write(code)
                temp_file = f.name
            
            # Execute the code
            result = subprocess.run(
                ['python', temp_file],
                capture_output=True,
                text=True,
                timeout=300  # 5 minute timeout
            )
            
            # Clean up
            os.unlink(temp_file)
            
            # Parse results
            if result.returncode == 0:
                # Try to extract JSON results from stdout
                try:
                    # Look for JSON in the output
                    output_lines = result.stdout.split('\n')
                    for line in output_lines:
                        if line.strip().startswith('{') and line.strip().endswith('}'):
                            return json.loads(line.strip())
                    
                    # If no JSON found, create structured output
                    return {
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                        'return_code': result.returncode,
                        'metrics': self._extract_metrics(result.stdout),
                        'data': self._extract_data(result.stdout)
                    }
                except json.JSONDecodeError:
                    return {
                        'stdout': result.stdout,
                        'stderr': result.stderr,
                        'return_code': result.returncode,
                        'metrics': self._extract_metrics(result.stdout),
                        'data': self._extract_data(result.stdout)
                    }
            else:
                return {
                    'error': result.stderr,
                    'return_code': result.returncode,
                    'stdout': result.stdout
                }
                
        except subprocess.TimeoutExpired:
            return {'error': 'Experiment timed out after 5 minutes'}
        except Exception as e:
            return {'error': f'Execution error: {str(e)}'}
    
    def _validate_results(self, results: Dict[str, Any], experiment: Dict[str, Any]) -> Dict[str, Any]:
        """Validate the experiment results."""
        validation = {
            'is_valid': False,
            'issues': [],
            'warnings': []
        }
        
        # Check for execution errors
        if 'error' in results:
            validation['issues'].append(f"Execution error: {results['error']}")
            return validation
        
        # Check for required metrics
        expected_metrics = experiment.get('evaluation_metrics', [])
        actual_metrics = results.get('metrics', {})
        
        for metric in expected_metrics:
            if metric not in actual_metrics:
                validation['warnings'].append(f"Missing metric: {metric}")
        
        # Check for reasonable values
        for metric, value in actual_metrics.items():
            if isinstance(value, (int, float)):
                if value < 0 or value > 1:  # Assuming normalized metrics
                    validation['warnings'].append(f"Unusual value for {metric}: {value}")
        
        # Check for data presence
        if not results.get('data') and not results.get('metrics'):
            validation['issues'].append("No results data generated")
        
        # Overall validation
        validation['is_valid'] = len(validation['issues']) == 0
        
        return validation
    
    def _generate_visualizations(self, results: Dict[str, Any], experiment: Dict[str, Any]) -> List[str]:
        """Generate visualizations for the experiment results."""
        visualizations = []
        
        try:
            # Generate plots based on results
            if 'metrics' in results and results['metrics']:
                # Create bar chart of metrics
                fig, ax = plt.subplots(figsize=(10, 6))
                metrics = results['metrics']
                ax.bar(metrics.keys(), metrics.values())
                ax.set_title(f'Results for {experiment["name"]}')
                ax.set_ylabel('Score')
                plt.xticks(rotation=45)
                plt.tight_layout()
                
                # Save plot
                plot_path = os.path.join(self.output_dir, f"{experiment['name']}_metrics.png")
                plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                plt.close()
                visualizations.append(plot_path)
            
            # Generate additional visualizations based on data
            if 'data' in results and results['data']:
                # Create scatter plot if data has x,y coordinates
                if isinstance(results['data'], dict) and 'x' in results['data'] and 'y' in results['data']:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.scatter(results['data']['x'], results['data']['y'])
                    ax.set_title(f'Data Visualization for {experiment["name"]}')
                    ax.set_xlabel('X')
                    ax.set_ylabel('Y')
                    plt.tight_layout()
                    
                    plot_path = os.path.join(self.output_dir, f"{experiment['name']}_data.png")
                    plt.savefig(plot_path, dpi=300, bbox_inches='tight')
                    plt.close()
                    visualizations.append(plot_path)
                    
        except Exception as e:
            log_agent_action("EngineerAgent", "visualization_error", {"error": str(e)})
        
        return visualizations
    
    def _extract_metrics(self, stdout: str) -> Dict[str, Any]:
        """Extract metrics from stdout."""
        metrics = {}
        
        # Look for metric patterns in output
        lines = stdout.split('\n')
        for line in lines:
            if ':' in line and any(keyword in line.lower() for keyword in ['accuracy', 'precision', 'recall', 'f1', 'score', 'metric']):
                try:
                    key, value = line.split(':', 1)
                    key = key.strip()
                    value = value.strip()
                    
                    # Try to convert to number
                    try:
                        value = float(value)
                    except ValueError:
                        pass
                    
                    metrics[key] = value
                except ValueError:
                    continue
        
        return metrics
    
    def _extract_data(self, stdout: str) -> Dict[str, Any]:
        """Extract data from stdout."""
        # Look for data patterns (lists, arrays, etc.)
        lines = stdout.split('\n')
        data = {}
        
        for line in lines:
            if line.strip().startswith('[') and line.strip().endswith(']'):
                try:
                    # Try to parse as list
                    data_list = json.loads(line.strip())
                    if isinstance(data_list, list):
                        data['values'] = data_list
                except json.JSONDecodeError:
                    continue
        
        return data
    
    def _clean_code(self, code: str) -> str:
        """Clean and format the generated code."""
        # Remove markdown formatting if present
        if code.startswith('```python'):
            code = code[9:]
        if code.endswith('```'):
            code = code[:-3]
        
        # Remove leading/trailing whitespace
        code = code.strip()
        
        return code
    
    def _generate_fallback_code(self, experiment: Dict[str, Any]) -> str:
        """Generate fallback code if LLM generation fails."""
        return f"""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier

# Generate synthetic data for {experiment['name']}
np.random.seed(42)
n_samples = 1000
n_features = 10

X = np.random.randn(n_samples, n_features)
y = np.random.randint(0, 2, n_samples)

# Split data
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Calculate metrics
metrics = {{
    'accuracy': accuracy_score(y_test, y_pred),
    'precision': precision_score(y_test, y_pred),
    'recall': recall_score(y_test, y_pred),
    'f1_score': f1_score(y_test, y_pred)
}}

# Print results
print("Experiment Results:")
for metric, value in metrics.items():
    print(f"{{metric}}: {{value:.4f}}")

# Output JSON for parsing
import json
print(json.dumps({{'metrics': metrics, 'data': {{'X_shape': X.shape, 'y_shape': y.shape}}}}))
"""
    
    def _store_experiment_results(self, output: Dict[str, Any], experiment: Dict[str, Any]):
        """Store experiment results in memory."""
        try:
            # Save results to file
            results_file = os.path.join(self.output_dir, f"{experiment['name']}_results.json")
            with open(results_file, 'w') as f:
                json.dump(output, f, indent=2)
            
            # Store in memory
            memory.add_embedding(
                generate_embedding(json.dumps(output), self.gemini_client),
                {
                    'type': 'experiment_results',
                    'experiment': experiment['name'],
                    'success': output['success'],
                    'results_file': results_file,
                    'timestamp': output['timestamp']
                }
            )
            
        except Exception as e:
            log_agent_action("EngineerAgent", "storage_error", {"error": str(e)})

# Example usage
if __name__ == "__main__":
    engineer = EngineerAgent()
    
    example_experiment = {
        'name': 'Baseline Comparison',
        'purpose': 'Compare our approach against existing methods',
        'methodology': 'Implement baseline methods and compare performance',
        'expected_outcomes': 'Quantitative performance metrics',
        'code_requirements': 'Python, scikit-learn, numpy',
        'data_requirements': 'Standard benchmark dataset',
        'evaluation_metrics': ['Accuracy', 'Precision', 'Recall', 'F1-Score'],
        'baseline_comparison': 'State-of-the-art methods in the field'
    }
    
    results = engineer.run_experiment(example_experiment)
    print(f"Experiment completed: {results['success']}")
    if 'metrics' in results.get('results', {}):
        print(f"Metrics: {results['results']['metrics']}") 