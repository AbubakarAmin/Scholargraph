"""Single-point dispatch for sandbox backend selection.

Routes ``execute`` and ``run_multi_seed`` calls to either the AST/builtins
sandbox (fast, in-process) or the Docker container sandbox (hard isolation)
based on ``config.sandbox_backend``.

All callers that need to respect the backend switch should call these
functions instead of importing from ``core.sandbox`` or
``core.container_sandbox`` directly.
"""

from __future__ import annotations

from typing import Any, Dict, List, Optional

from .config import config
from .sandbox import (
    execute_sandboxed,
    run_multi_seed,
    validate_code,
    run_known_answer_check,
)


def _use_docker() -> bool:
    return (config.sandbox_backend or "ast").lower() == "docker"


def execute(
    code: str,
    timeout_sec: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute one seed run using the configured backend."""
    if _use_docker():
        from .container_sandbox import execute_containerized

        return execute_containerized(
            code,
            timeout_sec=timeout_sec,
            seed=seed,
            memory=config.sandbox_docker_memory,
            cpus=config.sandbox_docker_cpus,
        )
    return execute_sandboxed(code, timeout_sec=timeout_sec, seed=seed)


def execute_multi_seed(
    code: str,
    n_seeds: Optional[int] = None,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """Run N seeds and aggregate metrics using the configured backend."""
    if _use_docker():
        from .container_sandbox import container_multi_seed

        return container_multi_seed(
            code,
            n_seeds=n_seeds,
            base_seed=base_seed,
            memory=config.sandbox_docker_memory,
            cpus=config.sandbox_docker_cpus,
        )
    return run_multi_seed(code, n_seeds=n_seeds, base_seed=base_seed)
