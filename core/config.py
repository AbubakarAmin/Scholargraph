"""
Configuration module for the multi-agent research system.
Handles environment variables, API keys, and system settings.
"""

import logging
import os
from typing import Optional
from pathlib import Path
from dotenv import load_dotenv
from pydantic_settings import BaseSettings

logger = logging.getLogger(__name__)

# Load environment variables from project-root .env (cwd-independent)
_PROJECT_ROOT = Path(__file__).resolve().parents[1]
load_dotenv(_PROJECT_ROOT / ".env")

# Force matplotlib to use non-interactive Agg backend process-wide to prevent
# Tcl_AsyncDelete: async handler deleted by the wrong thread on Windows.
os.environ.setdefault("MPLBACKEND", "Agg")

class Config(BaseSettings):
    """Configuration class for the research system."""

    # --- LLM Provider ---
    # gemini | openai | openai_compatible
    llm_provider: str = os.getenv("LLM_PROVIDER", "gemini")

    # Gemini
    google_api_key: str = os.getenv("GOOGLE_API_KEY", "")
    gemini_model: str = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
    gemini_embedding_model: str = os.getenv("GEMINI_EMBEDDING_MODEL", "gemini-embedding-2")
    embedding_dimension: int = int(os.getenv("EMBEDDING_DIMENSION", "3072"))

    # OpenAI / OpenAI-compatible (OpenRouter, Groq, Together, local vLLM, Ollama, etc.)
    openai_api_key: Optional[str] = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: Optional[str] = os.getenv("OPENAI_BASE_URL", "https://api.openai.com/v1")
    openai_model: str = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
    openai_embedding_model: str = os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")

    # Cost-aware routing: cheap model for lookups, strong for debate/writing/verify
    llm_model_cheap: str = os.getenv("LLM_MODEL_CHEAP", "")
    llm_model_strong: str = os.getenv("LLM_MODEL_STRONG", "")
    llm_model_judge: str = os.getenv("LLM_MODEL_JUDGE", "")
    llm_model_fallback: str = os.getenv("LLM_MODEL_FALLBACK", "")

    # Optional secondary judge models (comma-separated) for ensemble
    ensemble_judge_models: str = os.getenv("ENSEMBLE_JUDGE_MODELS", "")

    # External APIs
    scite_api_key: Optional[str] = os.getenv("SCITE_API_KEY", "")
    semantic_scholar_api_key: Optional[str] = os.getenv("SEMANTIC_SCHOLAR_API_KEY", "")
    openalex_email: str = os.getenv("OPENALEX_EMAIL", "researcher@example.com")

    # OpenReview
    openreview_username: str = os.getenv("OPENREVIEW_USERNAME", "")
    openreview_password: str = os.getenv("OPENREVIEW_PASSWORD", "")
    openreview_enabled: bool = os.getenv("OPENREVIEW_ENABLED", "true").lower() == "true"

    # HuggingFace (for dataset access)
    huggingface_token: str = os.getenv("HUGGINGFACE_TOKEN", "")
    huggingface_max_download_mb: int = int(os.getenv("HUGGINGFACE_MAX_DOWNLOAD_MB", "500"))
    openreview_venues: list = ["iclr2025", "iclr2026", "neurips2025"]
    arxiv_enabled: bool = os.getenv("ARXIV_ENABLED", "true").lower() == "true"
    openalex_enabled: bool = os.getenv("OPENALEX_ENABLED", "true").lower() == "true"

    # System
    research_domain: str = os.getenv("RESEARCH_DOMAIN", "computer_science")
    max_iterations: int = int(os.getenv("MAX_ITERATIONS", "10"))
    supervisor_threshold: float = float(os.getenv("SUPERVISOR_THRESHOLD", "8.5"))
    debate_pass_threshold: float = float(os.getenv("DEBATE_PASS_THRESHOLD", "7.0"))
    debate_min_rounds: int = int(os.getenv("DEBATE_MIN_ROUNDS", "2"))
    debate_max_rounds: int = int(os.getenv("DEBATE_MAX_ROUNDS", "5"))
    novelty_similarity_reject: float = float(os.getenv("NOVELTY_SIMILARITY_REJECT", "0.92"))
    experiment_seeds: int = int(os.getenv("EXPERIMENT_SEEDS", "3"))
    experiment_branch_count: int = int(os.getenv("EXPERIMENT_BRANCH_COUNT", "3"))
    debug_mode: bool = os.getenv("DEBUG_MODE", "true").lower() == "true"
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Memory / paths
    vector_db_path: str = os.getenv("VECTOR_DB_PATH", "./memory/vector_db")
    memory_size: int = int(os.getenv("MEMORY_SIZE", "10000"))
    cross_run_memory_path: str = os.getenv("CROSS_RUN_MEMORY_PATH", "./memory/cross_run.jsonl")
    run_log_path: str = os.getenv("RUN_LOG_PATH", "./output/run_scratchpad.jsonl")
    run_events_path: str = os.getenv("RUN_EVENTS_PATH", "./output/run_events.jsonl")
    elo_ratings_path: str = os.getenv("ELO_RATINGS_PATH", "./memory/elo_ratings.json")
    elo_prior_rating: float = float(os.getenv("ELO_PRIOR_RATING", "1500"))
    elo_shrinkage_k: float = float(os.getenv("ELO_SHRINKAGE_K", "8"))
    elo_min_observations: int = int(os.getenv("ELO_MIN_OBSERVATIONS", "5"))
    topic_exploration_every: int = int(os.getenv("TOPIC_EXPLORATION_EVERY", "4"))
    topic_exploration_seed: int = int(os.getenv("TOPIC_EXPLORATION_SEED", "42"))
    checkpoint_path: str = os.getenv("CHECKPOINT_PATH", "./memory/checkpoints.sqlite")
    research_db_path: str = os.getenv("RESEARCH_DB_PATH", "./memory/research_ledger.sqlite")

    output_dir: str = os.getenv("OUTPUT_DIR", "./output")
    draft_versions_dir: str = os.getenv("DRAFT_VERSIONS_DIR", "./output/draft_versions")
    debate_log_path: str = os.getenv("DEBATE_LOG_PATH", "./output/debate_log.json")
    feedback_log_path: str = os.getenv("FEEDBACK_LOG_PATH", "./output/feedback_log.json")
    raw_results_dir: str = os.getenv("RAW_RESULTS_DIR", "./output/raw_results")
    companion_repo_dir: str = os.getenv("COMPANION_REPO_DIR", "./output/companion_repo")

    # Sandbox
    sandbox_timeout_sec: int = int(os.getenv("SANDBOX_TIMEOUT_SEC", "120"))
    sandbox_max_output_bytes: int = int(os.getenv("SANDBOX_MAX_OUTPUT_BYTES", "1048576"))
    sandbox_backend: str = os.getenv("SANDBOX_BACKEND", "docker")  # "ast" | "docker"
    sandbox_docker_memory: str = os.getenv("SANDBOX_DOCKER_MEMORY", "2g")
    sandbox_docker_cpus: str = os.getenv("SANDBOX_DOCKER_CPUS", "1.0")

    # --- TopicHunter v2 feature flags ---
    openalex_concept_filtering_enabled: bool = os.getenv("OPENALEX_CONCEPT_FILTERING_ENABLED", "true").lower() == "true"
    hyde_enabled: bool = os.getenv("HYDE_ENABLED", "true").lower() == "true"
    hyde_max_chars: int = int(os.getenv("HYDE_MAX_CHARS", "600"))
    multi_hop_retrieval_enabled: bool = os.getenv("MULTI_HOP_RETRIEVAL_ENABLED", "true").lower() == "true"
    multi_hop_min_papers_threshold: int = int(os.getenv("MULTI_HOP_MIN_PAPERS_THRESHOLD", "12"))
    multi_hop_max_hops: int = int(os.getenv("MULTI_HOP_MAX_HOPS", "2"))
    frontier_seeding_enabled: bool = os.getenv("FRONTIER_SEEDING_ENABLED", "true").lower() == "true"
    frontier_refresh_every_n_runs: int = int(os.getenv("FRONTIER_REFRESH_EVERY_N_RUNS", "5"))
    frontier_sample_size: int = int(os.getenv("FRONTIER_SAMPLE_SIZE", "30"))
    frontier_terms_extracted: int = int(os.getenv("FRONTIER_TERMS_EXTRACTED", "8"))
    cross_seed_paper_cache_enabled: bool = os.getenv("CROSS_SEED_PAPER_CACHE_ENABLED", "true").lower() == "true"
    capability_first_dataset_scoping_enabled: bool = os.getenv("CAPABILITY_FIRST_DATASET_SCOPING_ENABLED", "true").lower() == "true"
    structural_gap_mining_enabled: bool = os.getenv("STRUCTURAL_GAP_MINING_ENABLED", "true").lower() == "true"
    structural_gap_max_pairs: int = int(os.getenv("STRUCTURAL_GAP_MAX_PAIRS", "8"))
    sparsity_matrix_enabled: bool = os.getenv("SPARSITY_MATRIX_ENABLED", "true").lower() == "true"
    contradiction_mining_enabled: bool = os.getenv("CONTRADICTION_MINING_ENABLED", "true").lower() == "true"
    replication_target_mining_enabled: bool = os.getenv("REPLICATION_TARGET_MINING_ENABLED", "true").lower() == "true"
    negative_result_seeding_enabled: bool = os.getenv("NEGATIVE_RESULT_SEEDING_ENABLED", "true").lower() == "true"
    persona_ensemble_enabled: bool = os.getenv("PERSONA_ENSEMBLE_ENABLED", "true").lower() == "true"
    persona_count: int = int(os.getenv("PERSONA_COUNT", "2"))
    seed_strategy_elo_enabled: bool = os.getenv("SEED_STRATEGY_ELO_ENABLED", "true").lower() == "true"

    # --- TopicHunter v3 ---
    llm_seed_generation_enabled: bool = os.getenv("LLM_SEED_GENERATION_ENABLED", "true").lower() == "true"

    # API gateway settings
    seed_sequential: bool = os.getenv("SEED_SEQUENTIAL", "true").lower() == "true"
    persona_sequential: bool = os.getenv("PERSONA_SEQUENTIAL", "true").lower() == "true"
    llm_budget_per_seed: int = int(os.getenv("LLM_BUDGET_PER_SEED", "30"))
    novelty_max_abstracts: int = int(os.getenv("NOVELTY_MAX_ABSTRACTS", "20"))

    # Web UI
    web_host: str = os.getenv("WEB_HOST", "127.0.0.1")
    web_port: int = int(os.getenv("WEB_PORT", "8765"))
    keys_store_path: str = os.getenv("KEYS_STORE_PATH", "./memory/keys.json")

    def get_ensemble_models(self) -> list[str]:
        if not self.ensemble_judge_models.strip():
            return []
        return [m.strip() for m in self.ensemble_judge_models.split(",") if m.strip()]

    def resolve_model(self, tier: str = "default") -> str:
        """Resolve model id for a cost tier: cheap | strong | judge | default."""
        provider = (self.llm_provider or "gemini").lower()
        if tier == "cheap" and self.llm_model_cheap:
            return self.llm_model_cheap
        if tier == "strong" and self.llm_model_strong:
            return self.llm_model_strong
        if tier == "judge" and self.llm_model_judge:
            return self.llm_model_judge
        if provider in ("openai", "openai_compatible"):
            return self.openai_model
        return self.gemini_model

    class Config:
        env_file = ".env"
        extra = "ignore"


# Global configuration instance
config = Config()

def apply_runtime_keys(keys: dict) -> None:
    """Apply keys from the web UI store into the live config + process env."""
    mapping = {
        "GOOGLE_API_KEY": "google_api_key",
        "OPENAI_API_KEY": "openai_api_key",
        "OPENAI_BASE_URL": "openai_base_url",
        "LLM_PROVIDER": "llm_provider",
        "GEMINI_MODEL": "gemini_model",
        "OPENAI_MODEL": "openai_model",
        "GEMINI_EMBEDDING_MODEL": "gemini_embedding_model",
        "OPENAI_EMBEDDING_MODEL": "openai_embedding_model",
        "EMBEDDING_DIMENSION": "embedding_dimension",
        "LLM_MODEL_CHEAP": "llm_model_cheap",
        "LLM_MODEL_STRONG": "llm_model_strong",
        "LLM_MODEL_JUDGE": "llm_model_judge",
        "LLM_MODEL_FALLBACK": "llm_model_fallback",
        "ENSEMBLE_JUDGE_MODELS": "ensemble_judge_models",
        "SEMANTIC_SCHOLAR_API_KEY": "semantic_scholar_api_key",
        "SCITE_API_KEY": "scite_api_key",
        "OPENALEX_EMAIL": "openalex_email",
        "RESEARCH_DOMAIN": "research_domain",
        "SUPERVISOR_THRESHOLD": "supervisor_threshold",
        "DEBATE_PASS_THRESHOLD": "debate_pass_threshold",
        "DEBATE_MIN_ROUNDS": "debate_min_rounds",
        "DEBATE_MAX_ROUNDS": "debate_max_rounds",
        "NOVELTY_SIMILARITY_REJECT": "novelty_similarity_reject",
        "EXPERIMENT_SEEDS": "experiment_seeds",
        "EXPERIMENT_BRANCH_COUNT": "experiment_branch_count",
        "MAX_ITERATIONS": "max_iterations",
        "DEBUG_MODE": "debug_mode",
        "LOG_LEVEL": "log_level",
        "VECTOR_DB_PATH": "vector_db_path",
        "MEMORY_SIZE": "memory_size",
        "CROSS_RUN_MEMORY_PATH": "cross_run_memory_path",
        "RUN_LOG_PATH": "run_log_path",
        "RUN_EVENTS_PATH": "run_events_path",
        "ELO_RATINGS_PATH": "elo_ratings_path",
        "ELO_PRIOR_RATING": "elo_prior_rating",
        "ELO_SHRINKAGE_K": "elo_shrinkage_k",
        "ELO_MIN_OBSERVATIONS": "elo_min_observations",
        "TOPIC_EXPLORATION_EVERY": "topic_exploration_every",
        "TOPIC_EXPLORATION_SEED": "topic_exploration_seed",
        "CHECKPOINT_PATH": "checkpoint_path",
        "RESEARCH_DB_PATH": "research_db_path",
        "OUTPUT_DIR": "output_dir",
        "DRAFT_VERSIONS_DIR": "draft_versions_dir",
        "DEBATE_LOG_PATH": "debate_log_path",
        "FEEDBACK_LOG_PATH": "feedback_log_path",
        "RAW_RESULTS_DIR": "raw_results_dir",
        "COMPANION_REPO_DIR": "companion_repo_dir",
        "SANDBOX_TIMEOUT_SEC": "sandbox_timeout_sec",
        "SANDBOX_MAX_OUTPUT_BYTES": "sandbox_max_output_bytes",
        "SANDBOX_BACKEND": "sandbox_backend",
        "SANDBOX_DOCKER_MEMORY": "sandbox_docker_memory",
        "SANDBOX_DOCKER_CPUS": "sandbox_docker_cpus",
        "WEB_HOST": "web_host",
        "WEB_PORT": "web_port",
        "KEYS_STORE_PATH": "keys_store_path",
        "OPENALEX_CONCEPT_FILTERING_ENABLED": "openalex_concept_filtering_enabled",
        "HYDE_ENABLED": "hyde_enabled",
        "HYDE_MAX_CHARS": "hyde_max_chars",
        "MULTI_HOP_RETRIEVAL_ENABLED": "multi_hop_retrieval_enabled",
        "MULTI_HOP_MIN_PAPERS_THRESHOLD": "multi_hop_min_papers_threshold",
        "MULTI_HOP_MAX_HOPS": "multi_hop_max_hops",
        "FRONTIER_SEEDING_ENABLED": "frontier_seeding_enabled",
        "FRONTIER_REFRESH_EVERY_N_RUNS": "frontier_refresh_every_n_runs",
        "FRONTIER_SAMPLE_SIZE": "frontier_sample_size",
        "FRONTIER_TERMS_EXTRACTED": "frontier_terms_extracted",
        "CROSS_SEED_PAPER_CACHE_ENABLED": "cross_seed_paper_cache_enabled",
        "CAPABILITY_FIRST_DATASET_SCOPING_ENABLED": "capability_first_dataset_scoping_enabled",
        "STRUCTURAL_GAP_MINING_ENABLED": "structural_gap_mining_enabled",
        "STRUCTURAL_GAP_MAX_PAIRS": "structural_gap_max_pairs",
        "SPARSITY_MATRIX_ENABLED": "sparsity_matrix_enabled",
        "CONTRADICTION_MINING_ENABLED": "contradiction_mining_enabled",
        "REPLICATION_TARGET_MINING_ENABLED": "replication_target_mining_enabled",
        "NEGATIVE_RESULT_SEEDING_ENABLED": "negative_result_seeding_enabled",
        "PERSONA_ENSEMBLE_ENABLED": "persona_ensemble_enabled",
        "PERSONA_COUNT": "persona_count",
        "SEED_STRATEGY_ELO_ENABLED": "seed_strategy_elo_enabled",
        "LLM_SEED_GENERATION_ENABLED": "llm_seed_generation_enabled",
        "OPENREVIEW_USERNAME": "openreview_username",
        "OPENREVIEW_PASSWORD": "openreview_password",
        "OPENREVIEW_ENABLED": "openreview_enabled",
        "ARXIV_ENABLED": "arxiv_enabled",
        "OPENALEX_ENABLED": "openalex_enabled",
        "HUGGINGFACE_TOKEN": "huggingface_token",
        "HUGGINGFACE_MAX_DOWNLOAD_MB": "huggingface_max_download_mb",
    }
    for env_key, attr in mapping.items():
        if env_key in keys and keys[env_key] not in (None, ""):
            value = keys[env_key]
            os.environ[env_key] = str(value)
            if hasattr(config, attr):
                current = getattr(config, attr)
                if isinstance(current, float):
                    setattr(config, attr, float(value))
                elif isinstance(current, int) and not isinstance(current, bool):
                    setattr(config, attr, int(value))
                elif isinstance(current, bool):
                    setattr(config, attr, str(value).lower() in {"1", "true", "yes", "on"})
                else:
                    setattr(config, attr, value)


def sync_env_file(keys: dict, env_path: Optional[Path] = None) -> Path:
    """Upsert UI-saved settings into project-root .env (preserves other lines/comments)."""
    path = Path(env_path) if env_path else (_PROJECT_ROOT / ".env")
    known = {
        "GOOGLE_API_KEY",
        "OPENAI_API_KEY",
        "OPENAI_BASE_URL",
        "LLM_PROVIDER",
        "GEMINI_MODEL",
        "OPENAI_MODEL",
        "GEMINI_EMBEDDING_MODEL",
        "OPENAI_EMBEDDING_MODEL",
        "EMBEDDING_DIMENSION",
        "LLM_MODEL_CHEAP",
        "LLM_MODEL_STRONG",
        "LLM_MODEL_JUDGE",
        "LLM_MODEL_FALLBACK",
        "ENSEMBLE_JUDGE_MODELS",
        "SEMANTIC_SCHOLAR_API_KEY",
        "SCITE_API_KEY",
        "OPENALEX_EMAIL",
        "RESEARCH_DOMAIN",
        "SUPERVISOR_THRESHOLD",
        "DEBATE_PASS_THRESHOLD",
        "DEBATE_MIN_ROUNDS",
        "DEBATE_MAX_ROUNDS",
        "NOVELTY_SIMILARITY_REJECT",
        "EXPERIMENT_SEEDS",
        "EXPERIMENT_BRANCH_COUNT",
        "MAX_ITERATIONS",
        "DEBUG_MODE",
        "LOG_LEVEL",
        "VECTOR_DB_PATH",
        "MEMORY_SIZE",
        "CROSS_RUN_MEMORY_PATH",
        "RUN_LOG_PATH",
        "RUN_EVENTS_PATH",
        "ELO_RATINGS_PATH",
        "CHECKPOINT_PATH",
        "RESEARCH_DB_PATH",
        "OUTPUT_DIR",
        "DRAFT_VERSIONS_DIR",
        "DEBATE_LOG_PATH",
        "FEEDBACK_LOG_PATH",
        "RAW_RESULTS_DIR",
        "COMPANION_REPO_DIR",
        "SANDBOX_TIMEOUT_SEC",
        "SANDBOX_MAX_OUTPUT_BYTES",
        "SANDBOX_BACKEND",
        "SANDBOX_DOCKER_MEMORY",
        "SANDBOX_DOCKER_CPUS",
        "WEB_HOST",
        "WEB_PORT",
        "KEYS_STORE_PATH",
        "OPENALEX_CONCEPT_FILTERING_ENABLED",
        "HYDE_ENABLED",
        "HYDE_MAX_CHARS",
        "MULTI_HOP_RETRIEVAL_ENABLED",
        "MULTI_HOP_MIN_PAPERS_THRESHOLD",
        "MULTI_HOP_MAX_HOPS",
        "FRONTIER_SEEDING_ENABLED",
        "FRONTIER_REFRESH_EVERY_N_RUNS",
        "FRONTIER_SAMPLE_SIZE",
        "FRONTIER_TERMS_EXTRACTED",
        "CROSS_SEED_PAPER_CACHE_ENABLED",
        "CAPABILITY_FIRST_DATASET_SCOPING_ENABLED",
        "STRUCTURAL_GAP_MINING_ENABLED",
        "STRUCTURAL_GAP_MAX_PAIRS",
        "SPARSITY_MATRIX_ENABLED",
        "CONTRADICTION_MINING_ENABLED",
        "REPLICATION_TARGET_MINING_ENABLED",
        "NEGATIVE_RESULT_SEEDING_ENABLED",
        "PERSONA_ENSEMBLE_ENABLED",
        "PERSONA_COUNT",
        "SEED_STRATEGY_ELO_ENABLED",
        "LLM_SEED_GENERATION_ENABLED",
        "OPENREVIEW_USERNAME",
        "OPENREVIEW_PASSWORD",
        "OPENREVIEW_ENABLED",
        "ARXIV_ENABLED",
        "OPENALEX_ENABLED",
        "HUGGINGFACE_TOKEN",
        "HUGGINGFACE_MAX_DOWNLOAD_MB",
    }
    updates = {
        str(k): str(v)
        for k, v in keys.items()
        if k in known and v not in (None, "")
    }
    if not updates:
        return path

    lines: list[str] = []
    if path.exists():
        lines = path.read_text(encoding="utf-8").splitlines()

    seen: set[str] = set()
    out: list[str] = []
    for line in lines:
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            out.append(line)
            continue
        key = stripped.split("=", 1)[0].strip()
        if key in updates:
            out.append(f"{key}={updates[key]}")
            seen.add(key)
        else:
            out.append(line)

    missing = [k for k in updates if k not in seen]
    if missing:
        if out and out[-1].strip():
            out.append("")
        out.append("# Synced from Control Deck UI")
        for key in missing:
            out.append(f"{key}={updates[key]}")

    path.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(out)
    if text and not text.endswith("\n"):
        text += "\n"
    path.write_text(text, encoding="utf-8")
    return path


def _resolve_keys_path() -> Path:
    """Resolve keys.json relative to project root (not process cwd)."""
    raw = Path(config.keys_store_path)
    if raw.is_absolute():
        return raw
    return (_PROJECT_ROOT / raw).resolve()


def _auto_load_keys():
    try:
        import json
        p = _resolve_keys_path()
        if p.exists():
            data = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(data, dict) and data:
                apply_runtime_keys(data)
    except Exception as e:
        logger.warning("Failed to auto-load keys from %s: %s", _resolve_keys_path(), e)


_auto_load_keys()


def validate_config():
    """Validate that required settings are present for the selected provider."""
    provider = (config.llm_provider or "gemini").lower()

    if provider == "gemini":
        if not config.google_api_key:
            raise ValueError(
                "GOOGLE_API_KEY is required when LLM_PROVIDER=gemini. "
                "Get it from https://aistudio.google.com/apikey"
            )
        if not config.gemini_embedding_model:
            raise ValueError(
                "GEMINI_EMBEDDING_MODEL is required when LLM_PROVIDER=gemini. "
                "Default is gemini-embedding-2."
            )
    elif provider in ("openai", "openai_compatible"):
        if not config.openai_api_key:
            raise ValueError(
                "OPENAI_API_KEY is required when LLM_PROVIDER is openai or "
                "openai_compatible (also used for OpenRouter/Groq/local endpoints). "
                "For local servers that ignore auth, set any non-empty value e.g. local."
            )
        if provider == "openai_compatible" and not (config.openai_base_url or "").strip():
            raise ValueError(
                "OPENAI_BASE_URL is required for openai_compatible "
                "(e.g. http://127.0.0.1:11434/v1 for Ollama)."
            )
    else:
        raise ValueError(
            f"Unknown LLM_PROVIDER={provider}. Use gemini, openai, or openai_compatible."
        )

    if not config.openalex_email:
        raise ValueError(
            "OPENALEX_EMAIL is required for rate limiting. Set it to your email."
        )

    for path in (
        config.output_dir,
        config.draft_versions_dir,
        config.raw_results_dir,
        config.companion_repo_dir,
        os.path.dirname(config.vector_db_path) or "./memory",
        "./memory",
    ):
        os.makedirs(path, exist_ok=True)

    return True
 