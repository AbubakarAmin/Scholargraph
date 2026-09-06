"""
Test script to verify Gemini API integration with the new google-genai library.
"""

import os
import sys
from pathlib import Path

# Add project root to path
sys.path.append(str(Path(__file__).parent))

from core.config import config
from core.llm import call_gemini, generate_embedding, setup_gemini

def test_gemini_integration():
    """Test the Gemini API integration."""
    print("🧪 Testing Gemini API Integration")
    print("=" * 50)
    
    try:
        # Test 1: Setup Gemini client
        print("1. Testing Gemini client setup...")
        client = setup_gemini()
        print("   ✅ Client created successfully")
        
        # Test 2: Simple content generation
        print("2. Testing content generation...")
        test_prompt = "Explain how AI works in a few words"
        
        # Apply rate limiting
        _rate_limit()
        
        result = client.chat(test_prompt, temperature=0.7, max_tokens=100)
        print(f"   ✅ Response received: {result[:50]}...")
        
        # Test 3: Test the call_gemini utility function
        print("3. Testing call_gemini utility function...")
        result = call_gemini(test_prompt, client, temperature=0.7)
        print(f"   ✅ Utility function works: {result[:50]}...")
        
        # Test 4: Test embedding generation
        print("4. Testing embedding generation...")
        embedding = generate_embedding("test text", client)
        print(f"   ✅ Embedding generated: shape {embedding.shape}")
        
        print("\n🎉 All tests passed! Gemini API integration is working correctly.")
        return True
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        print(f"   Error type: {type(e).__name__}")
        return False

if __name__ == "__main__":
    success = test_gemini_integration()
    sys.exit(0 if success else 1) 