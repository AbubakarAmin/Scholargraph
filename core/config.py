"""
Configuration module for the multi-agent research system.
Handles environment variables, API keys, and system settings.
"""

import os
from typing import Optional
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

# Load environment variables
load_dotenv()

class Config(BaseSettings):
    """Configuration class for the research system."""
    
    # API Keys
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY", "")
    scite_api_key: Optional[str] = os.getenv("SCITE_API_KEY", "")
    
    # OpenAlex Configuration (no API key required, just email for rate limiting)
    openalex_email: str = os.getenv("OPENALEX_EMAIL", "darkdevil034477@gmail.com")
    
    # System Configuration
    research_domain: str = os.getenv("RESEARCH_DOMAIN", "computer_science")
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "10"))
    supervisor_threshold: float = float(os.getenv("SUPERVISOR_THRESHOLD", "8.5"))
    debug_mode: bool = os.getenv("DEBUG_MODE", "true").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")
    
    # Memory Configuration
    vector_db_path: str = os.getenv("VECTOR_DB_PATH", "./memory/vector_db")
    memory_size: int = int(os.getenv("MEMORY_SIZE", "10000"))
    
    # Output Configuration
    output_dir: str = os.getenv("OUTPUT_DIR", "./output")
    draft_versions_dir: str = os.getenv("DRAFT_VERSIONS_DIR", "./output/draft_versions")
    debate_log_path: str = os.getenv("DEBATE_LOG_PATH", "./output/debate_log.txt")
    feedback_log_path: str = os.getenv("FEEDBACK_LOG_PATH", "./output/feedback_log.json")
    
    class Config:
        env_file = ".env"

# Global configuration instance
config = Config()

def validate_config():
    """Validate that required API keys are present."""
    if not config.google_api_key:
        raise ValueError("GOOGLE_API_KEY is required. Get it from https://makersuite.google.com/app/apikey")
    
    if not config.openalex_email:
        raise ValueError("OPENALEX_EMAIL is required for rate limiting. Set it to your email address.")
    
    # Create output directories if they don't exist
    os.makedirs(config.output_dir, exist_ok=True)
    os.makedirs(config.draft_versions_dir, exist_ok=True)
    os.makedirs(os.path.dirname(config.vector_db_path), exist_ok=True)
    
    return True 