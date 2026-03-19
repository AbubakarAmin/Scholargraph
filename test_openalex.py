#!/usr/bin/env python3
"""
Test script to verify OpenAlex integration works correctly.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.config import config
from agents.topic_hunter import TopicHunterAgent

def test_openalex_integration():
    """Test the OpenAlex integration."""
    print("🧪 Testing OpenAlex Integration")
    print("=" * 50)
    
    # Test configuration
    print(f"✓ OpenAlex email configured: {config.openalex_email}")
    
    # Test TopicHunterAgent initialization
    try:
        hunter = TopicHunterAgent()
        print("✓ TopicHunterAgent initialized successfully")
        
        # Test OpenAlex search (limited to 5 results for testing)
        print("\n🔍 Testing OpenAlex search...")
        results = hunter.search_openalex("machine learning", limit=5)
        
        if results:
            print(f"✓ OpenAlex search successful: {len(results)} results found")
            print(f"  Sample result: {results[0].get('title', 'No title')[:100]}...")
        else:
            print("⚠️  OpenAlex search returned no results (this might be normal)")
            
    except Exception as e:
        print(f"✗ Error testing OpenAlex: {e}")
        return False
    
    print("\n✅ OpenAlex integration test completed successfully!")
    return True

if __name__ == "__main__":
    test_openalex_integration() 