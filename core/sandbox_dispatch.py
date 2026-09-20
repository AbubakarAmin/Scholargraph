"""Single-point dispatch for sandbox backend selection.

Routes ``execute`` and ``run_multi_seed`` calls to either the AST/builtins
sandbox (fast, in-process) or the Docker container sandbox (hard isolation)
based on ``config.sandbox_backend``.

All callers that need to respect the backend switch should call these
functions instead of importing from ``core.sandbox`` or
``core.container_sandbox`` directly.

Robustness (v3): when the Docker backend is configured but the daemon is
unreachable (e.g. Docker Desktop not running), dispatch falls back to the
AST sandbox with a one-time warning instead of failing every run.
"""

from __future__ import annotations

import logging
import shutil
import subprocess
import threading
from typing import Any, Dict, List, Optional

from .config import config
from .sandbox import (
    execute_sandboxed,
    run_multi_seed,
    validate_code,
    run_known_answer_check,
)

logger = logging.getLogger(__name__)

_docker_probe_result: bool | None = None
_docker_probe_lock = threading.Lock()
_docker_fallback_warned = False


def _docker_daemon_reachable() -> bool:
    """Probe the Docker daemon once per process; cache the verdict."""
    global _docker_probe_result, _docker_fallback_warned
    if _docker_probe_result is not None:
        return _docker_probe_result
    with _docker_probe_lock:
        if _docker_probe_result is not None:
            return _docker_probe_result
        docker_exe = shutil.which("docker")
        if not docker_exe:
            _docker_probe_result = False
        else:
            try:
                probe = subprocess.run(
                    [docker_exe, "info", "--format", "{{.ServerVersion}}"],
                    capture_output=True,
                    timeout=5,
                )
                _docker_probe_result = probe.returncode == 0
            except Exception:
                _docker_probe_result = False
        if not _docker_probe_result and not _docker_fallback_warned:
            _docker_fallback_warned = True
            logger.warning(
                "SANDBOX_BACKEND=docker but the Docker daemon is unreachable — "
                "falling back to the AST sandbox for this process."
            )
    return _docker_probe_result


def _use_docker() -> bool:
    if (config.sandbox_backend or "ast").lower() != "docker":
        return False
    return _docker_daemon_reachable()


def execute(
    code: str,
    timeout_sec: Optional[int] = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """Execute one seed run using the configured backend."""
    use_docker = _use_docker()
    backend = "docker" if use_docker else "ast"
    logger.info("Sandbox execute started (backend=%s, seed=%s, timeout=%s)", backend, seed, timeout_sec)

    try:
        if use_docker:
            from .container_sandbox import execute_containerized

            result = execute_containerized(
                code,
                timeout_sec=timeout_sec,
                seed=seed,
                memory=config.sandbox_docker_memory,
                cpus=config.sandbox_docker_cpus,
            )
        else:
            result = execute_sandboxed(code, timeout_sec=timeout_sec, seed=seed)
    except TimeoutError:
        logger.info("Sandbox execution timed out (backend=%s, seed=%s, timeout=%s)", backend, seed, timeout_sec)
        raise
    except Exception as exc:
        logger.info("Sandbox execution failed (backend=%s, seed=%s, error=%s)", backend, seed, exc)
        raise

    status = result.get("status", "unknown") if isinstance(result, dict) else "unknown"
    logger.info("Sandbox execution finished (backend=%s, seed=%s, status=%s)", backend, seed, status)
    return result


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
