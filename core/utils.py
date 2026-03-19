"""
Utility functions for the multi-agent research system.
Includes LLM setup, embedding generation, and common operations.
"""

import json
import logging
import time
from typing import List, Dict, Any, Optional
from datetime import datetime
from google import genai
import numpy as np
from pathlib import Path

from .config import config

# Setup logging
logging.basicConfig(
    level=getattr(logging, config.log_level),
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Rate limiting for Gemini API (10 requests per minute)
_last_request_time = 0
_request_interval = 4.0  # 60 seconds / 10 requests = 6 seconds between requests

def setup_gemini():
    """Setup Google Gemini API."""
    try:
        client = genai.Client(api_key=config.google_api_key)
        return client
    except Exception as e:
        logger.error(f"Failed to setup Gemini: {e}")
        raise

def _rate_limit():
    """Ensure we don't exceed rate limits."""
    global _last_request_time
    current_time = time.time()
    time_since_last = current_time - _last_request_time
    
    if time_since_last < _request_interval:
        sleep_time = _request_interval - time_since_last
        logger.debug(f"Rate limiting: sleeping for {sleep_time:.2f} seconds")
        time.sleep(sleep_time)
    
    _last_request_time = time.time()

def generate_embedding(text: str, model) -> np.ndarray:
    """Generate embedding for text using Gemini."""
    try:
        _rate_limit()
        # Use Gemini's embedding model
        response = model.models.embed_content(
            model="models/embedding-001",
            contents=text
        )
        # The response structure is different in the new library
        if hasattr(response, 'embeddings') and response.embeddings:
            return np.array(response.embeddings[0].values)
        elif hasattr(response, 'embedding') and response.embedding:
            return np.array(response.embedding.values[0].values)
        else:
            # Fallback: try to access the embedding data directly
            return np.array(response.values[0].values)
    except Exception as e:
        logger.error(f"Failed to generate embedding: {e}")
        # Return zero vector as fallback
        return np.zeros(768)

def call_gemini(prompt: str, model, temperature: float = 0.7) -> str:
    """Call Gemini model with a prompt."""
    try:
        _rate_limit()
        response = model.models.generate_content(
            model="models/gemini-2.5-flash-lite",
            contents=prompt,
            config={
                "temperature": temperature,
                "max_output_tokens": 8192,
            }
        )
        return response.candidates[0].content.parts[0].text
    except Exception as e:
        logger.error(f"Failed to call Gemini: {e}")
        return ""

def save_json(data: Dict[str, Any], filepath: str):
    """Save data to JSON file."""
    try:
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved data to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}")

def load_json(filepath: str) -> Dict[str, Any]:
    """Load data from JSON file."""
    try:
        with open(filepath, 'r') as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON from {filepath}: {e}")
        return {}

def extract_citations(text: str) -> List[str]:
    """Extract citation patterns from text."""
    import re
    # Common citation patterns
    patterns = [
        r'\[([^\]]+)\]',  # [Author et al., 2023]
        r'\(([^)]+)\)',   # (Author et al., 2023)
        r'Author et al\.\s+\d{4}',  # Author et al. 2023
    ]
    
    citations = []
    for pattern in patterns:
        matches = re.findall(pattern, text)
        citations.extend(matches)
    
    return list(set(citations))

def validate_math_expression(expression: str) -> bool:
    """Validate mathematical expression using SymPy."""
    try:
        import sympy as sp
        # Try to parse the expression
        sp.sympify(expression)
        return True
    except Exception:
        return False

def sanitize_filename(filename: str) -> str:
    """Sanitize filename for safe file operations."""
    import re
    # Remove or replace invalid characters
    sanitized = re.sub(r'[<>:"/\\|?*]', '_', filename)
    # Limit length
    if len(sanitized) > 100:
        sanitized = sanitized[:100]
    return sanitized

def create_timestamped_filename(prefix: str, extension: str) -> str:
    """Create a timestamped filename."""
    from datetime import datetime
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"

def format_section_text(section_name: str, content: str, level: int = 1) -> str:
    """Format section text with proper LaTeX formatting."""
    if level == 1:
        return f"\\section{{{section_name}}}\n\n{content}\n"
    elif level == 2:
        return f"\\subsection{{{section_name}}}\n\n{content}\n"
    elif level == 3:
        return f"\\subsubsection{{{section_name}}}\n\n{content}\n"
    else:
        return f"\\paragraph{{{section_name}}}\n\n{content}\n"

def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    """Extract keywords from text using simple heuristics."""
    import re
    from collections import Counter
    
    # Remove common words
    stop_words = {
        'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for',
        'of', 'with', 'by', 'is', 'are', 'was', 'were', 'be', 'been', 'being',
        'have', 'has', 'had', 'do', 'does', 'did', 'will', 'would', 'could',
        'should', 'may', 'might', 'can', 'this', 'that', 'these', 'those'
    }
    
    # Extract words
    words = re.findall(r'\b[a-zA-Z]{3,}\b', text.lower())
    words = [word for word in words if word not in stop_words]
    
    # Count and return most common
    word_counts = Counter(words)
    return [word for word, count in word_counts.most_common(max_keywords)]

def calculate_similarity(text1: str, text2: str) -> float:
    """Calculate similarity between two texts using embeddings."""
    try:
        client = setup_gemini()
        embedding1 = generate_embedding(text1, client)
        embedding2 = generate_embedding(text2, client)
        
        # Cosine similarity
        similarity = np.dot(embedding1, embedding2) / (np.linalg.norm(embedding1) * np.linalg.norm(embedding2))
        return float(similarity)
    except Exception as e:
        logger.error(f"Failed to calculate similarity: {e}")
        return 0.0

def log_agent_action(agent_name: str, action: str, details: Dict[str, Any] = None):
    """Log agent actions for debugging and monitoring."""
    log_entry = {
        "timestamp": str(datetime.now()),
        "agent": agent_name,
        "action": action,
        "details": details or {}
    }
    logger.info(f"Agent {agent_name}: {action}")
    if config.debug_mode:
        logger.debug(f"Details: {json.dumps(log_entry, indent=2)}") 