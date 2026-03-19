"""
Test script for the multi-agent research system.
Tests individual components and basic functionality.
"""

import os
import sys
from pathlib import Path

# Add the project root to the Python path
sys.path.append(str(Path(__file__).parent))

def test_config():
    """Test configuration loading."""
    print("Testing configuration...")
    try:
        from core.config import config, validate_config
        print(f"OK: Configuration loaded successfully")
        print(f"  Research domain: {config.research_domain}")
        print(f"  Supervisor threshold: {config.supervisor_threshold}")
        return True
    except Exception as e:
        print(f"ERROR: Configuration test failed: {e}")
        return False

def test_memory():
    """Test memory system."""
    print("\nTesting memory system...")
    try:
        from core.memory import memory
        import numpy as np
        
        # Test adding embedding
        test_embedding = np.random.rand(768)
        memory.add_embedding(test_embedding, {
            'type': 'test',
            'content': 'test content'
        })
        print("OK: Memory system working")
        return True
    except Exception as e:
        print(f"ERROR: Memory test failed: {e}")
        return False

def test_utils():
    """Test utility functions."""
    print("\nTesting utility functions...")
    try:
        from core.utils import extract_citations, validate_math_expression
        
        # Test citation extraction
        test_text = "Previous work [Author et al., 2023] shows that..."
        citations = extract_citations(test_text)
        print(f"OK: Citation extraction: {citations}")
        
        # Test math validation
        is_valid = validate_math_expression("x + y = z")
        print(f"OK: Math validation: {is_valid}")
        
        return True
    except Exception as e:
        print(f"ERROR: Utils test failed: {e}")
        return False

def test_agents():
    """Test agent imports."""
    print("\nTesting agent imports...")
    try:
        from agents.topic_hunter import TopicHunterAgent
        from agents.hypothesis_debate import HypothesisDebateSystem
        from agents.planner import PlannerAgent
        from agents.writer import WriterAgent
        from agents.supervisor import SupervisorAgent
        from agents.engineer import EngineerAgent
        from agents.editor import EditorAgent
        from agents.meta_agent import MetaAgent
        
        print("OK: All agents imported successfully")
        return True
    except Exception as e:
        print(f"ERROR: Agent import test failed: {e}")
        return False

def test_main_orchestration():
    """Test main orchestration import."""
    print("\nTesting main orchestration...")
    try:
        from main import create_research_graph, ResearchState
        print("OK: Main orchestration imported successfully")
        return True
    except Exception as e:
        print(f"ERROR: Main orchestration test failed: {e}")
        return False

def test_dependencies():
    """Test that all required dependencies are available."""
    print("\nTesting dependencies...")
    required_packages = [
        'langgraph', 'langchain', 'google.generativeai', 'numpy', 
        'pandas', 'matplotlib', 'sympy', 'requests', 'arxiv'
    ]
    
    missing_packages = []
    for package in required_packages:
        try:
            __import__(package)
            print(f"OK: {package}")
        except ImportError:
            print(f"Missing: {package}")
            missing_packages.append(package)
    
    if missing_packages:
        print(f"\nMissing packages: {missing_packages}")
        print("Install with: pip install -r requirements.txt")
        return False
    else:
        print("OK: All dependencies available")
        return True

def test_directory_structure():
    """Test that all required directories exist."""
    print("\nTesting directory structure...")
    required_dirs = [
        'agents', 'core', 'templates', 'output', 'memory'
    ]
    
    missing_dirs = []
    for dir_name in required_dirs:
        if os.path.exists(dir_name):
            print(f"OK: {dir_name}/")
        else:
            print(f"Missing: {dir_name}/")
            missing_dirs.append(dir_name)
    
    if missing_dirs:
        print(f"\nMissing directories: {missing_dirs}")
        return False
    else:
        print("OK: All directories present")
        return True

def main():
    """Run all tests."""
    print("Testing Multi-Agent Research System")
    print("=" * 50)
    
    tests = [
        test_dependencies,
        test_directory_structure,
        test_config,
        test_memory,
        test_utils,
        test_agents,
        test_main_orchestration
    ]
    
    passed = 0
    total = len(tests)
    
    for test in tests:
        if test():
            passed += 1
    
    print("\n" + "=" * 50)
    print(f"Test Results: {passed}/{total} tests passed")
    
    if passed == total:
        print("All tests passed! System is ready to use.")
        print("\nNext steps:")
        print("1. Copy env_example.txt to .env")
        print("2. Add your API keys to .env")
        print("3. Run: python main.py")
    else:
        print("Some tests failed. Please fix the issues above.")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1) 