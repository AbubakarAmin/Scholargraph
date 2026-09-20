"""Docker-based execution backend for Engineer experiments.

Provides stronger isolation than the AST/builtins sandbox by running
generated code inside a container with no network access, configurable
memory/CPU limits, and automatic cleanup.

The returned dict shapes are identical to ``core.sandbox.execute_sandboxed``
and ``core.sandbox.run_multi_seed`` so callers do not need to change.
"""

from __future__ import annotations

import json
import logging
import os
import shutil
import subprocess
import tempfile
import time
import traceback
from pathlib import Path
from typing import Any, Dict, List, Optional

from .config import config

logger = logging.getLogger(__name__)

IMAGE_NAME = "scholargraph-sandbox:latest"

DEFAULT_MEMORY_LIMIT = "2g"
DEFAULT_CPU_LIMIT = "1.0"


def _resource_limits(
    memory: Optional[str] = None,
    cpus: Optional[str] = None,
) -> Dict[str, str]:
    return {
        "memory": memory or DEFAULT_MEMORY_LIMIT,
        "cpus": cpus or DEFAULT_CPU_LIMIT,
    }


def _build_preamble(seed: int) -> str:
    return (
        f"import random as _sg_random\n"
        f"import numpy as _sg_np\n"
        f"import numpy as np\n"
        f"import pandas as pd\n"
        f"import matplotlib\n"
        f"matplotlib.use('Agg')\n"
        f"import matplotlib.pyplot as plt\n"
        f"_sg_random.seed({seed})\n"
        f"_sg_np.random.seed({seed})\n"
    )


def _parse_json_from_stdout(stdout: str) -> Dict[str, Any]:
    """Same logic as core.sandbox._parse_json_from_stdout."""
    text = (stdout or "").strip()
    if not text:
        return {}
    for line in reversed(text.splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    last_close = text.rfind("}")
    if last_close >= 0:
        depth = 0
        for i in range(last_close, -1, -1):
            if text[i] == "}":
                depth += 1
            elif text[i] == "{":
                depth -= 1
                if depth == 0:
                    candidate = text[i : last_close + 1]
                    try:
                        return json.loads(candidate)
                    except json.JSONDecodeError:
                        break
    if text.startswith("{"):
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            pass
    return {}


def _container_running(container_id: str) -> bool:
    try:
        r = subprocess.run(
            ["docker", "inspect", "-f", "{{.State.Running}}", container_id],
            capture_output=True, text=True, timeout=5,
        )
        return "true" in (r.stdout or "").lower()
    except Exception:
        return False


def _force_remove(container_id: str) -> None:
    try:
        subprocess.run(
            ["docker", "rm", "-f", container_id],
            capture_output=True, timeout=10,
        )
    except Exception:
        pass


def execute_containerized(
    code: str,
    timeout_sec: Optional[int] = None,
    seed: int = 42,
    memory: Optional[str] = None,
    cpus: Optional[str] = None,
) -> Dict[str, Any]:
    """Execute code inside a Docker container.

    Returns the same dict shape as ``core.sandbox.execute_sandboxed``.
    """
    timeout_sec = timeout_sec or config.sandbox_timeout_sec
    limits = _resource_limits(memory, cpus)
    preamble = _build_preamble(seed)
    full_code = preamble + "\n" + code

    tmpdir = tempfile.mkdtemp(prefix="sg_container_")
    code_path = os.path.join(tmpdir, "experiment.py")
    output_dir = os.path.join(tmpdir, "output")
    os.makedirs(output_dir, exist_ok=True)

    try:
        Path(code_path).write_text(full_code, encoding="utf-8")
        container_id = None
        try:
            cmd = [
                "docker", "run",
                "--rm",
                "--network", "none",
                "--memory", limits["memory"],
                "--cpus", limits["cpus"],
                "--read-only",
                "--tmpfs", "/tmp:size=64m",
                "-v", f"{code_path}:/sandbox/experiment.py:ro",
                "-v", f"{output_dir}:/sandbox/output",
                IMAGE_NAME,
                "python", "/sandbox/experiment.py",
            ]
            t0 = time.monotonic()
            proc = subprocess.Popen(
                cmd,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True,
            )
            try:
                stdout, stderr = proc.communicate(timeout=timeout_sec)
                elapsed = time.monotonic() - t0
                exit_code = proc.returncode
            except subprocess.TimeoutExpired:
                elapsed = time.monotonic() - t0
                # Kill the docker run process (this stops the container)
                proc.kill()
                try:
                    stdout, stderr = proc.communicate(timeout=10)
                except Exception:
                    stdout = ""
                    stderr = ""
                # Clean up any leftover containers from this image
                try:
                    inspect = subprocess.run(
                        ["docker", "ps", "-q", "--filter", "ancestor=" + IMAGE_NAME],
                        capture_output=True, text=True, timeout=5,
                    )
                    for cid in (inspect.stdout or "").strip().splitlines():
                        if cid:
                            _force_remove(cid)
                except Exception:
                    pass
                return {
                    "success": False,
                    "error": f"Container timeout after {timeout_sec}s",
                    "stdout": stdout,
                    "stderr": f"timeout:{timeout_sec}",
                    "seed": seed,
                    "timeout": True,
                }

            if exit_code != 0:
                return {
                    "success": False,
                    "error": f"Container exited with code {exit_code}",
                    "stdout": stdout,
                    "stderr": stderr,
                    "seed": seed,
                }

            stdout_truncated = stdout
            max_bytes = config.sandbox_max_output_bytes
            if len(stdout_truncated) > max_bytes:
                stdout_truncated = stdout_truncated[:max_bytes] + "\n...[truncated]"

            parsed = _parse_json_from_stdout(stdout_truncated)
            return {
                "success": True,
                "stdout": stdout_truncated,
                "stderr": stderr,
                "parsed": parsed,
                "seed": seed,
            }

        except FileNotFoundError:
            return {
                "success": False,
                "error": "Docker not found on PATH — install Docker and ensure daemon is running",
                "stdout": "",
                "stderr": "docker binary not found",
                "seed": seed,
            }
        except Exception as e:
            if container_id and _container_running(container_id):
                _force_remove(container_id)
            return {
                "success": False,
                "error": str(e),
                "traceback": traceback.format_exc(),
                "stdout": "",
                "stderr": str(e),
                "seed": seed,
            }
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)

    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "stdout": "",
            "stderr": str(e),
            "seed": seed,
        }


def container_multi_seed(
    code: str,
    n_seeds: Optional[int] = None,
    base_seed: int = 42,
    memory: Optional[str] = None,
    cpus: Optional[str] = None,
) -> Dict[str, Any]:
    """Run the same experiment with multiple seeds inside containers;
    aggregate mean +/- std. Returns the same dict shape as
    ``core.sandbox.run_multi_seed``.
    """
    import numpy as np

    n_seeds = n_seeds or config.experiment_seeds
    runs: List[Dict[str, Any]] = []
    for i in range(n_seeds):
        seed = base_seed + i * 1009
        result = execute_containerized(code, seed=seed, memory=memory, cpus=cpus)
        runs.append(result)

    successes = [r for r in runs if r.get("success")]
    if not successes:
        error_details = []
        for i, r in enumerate(runs):
            err = r.get("error") or "unknown"
            stderr = r.get("stderr") or ""
            detail = f"seed_{i}: {err}"
            if stderr:
                detail += f" | stderr: {stderr[:300]}"
            error_details.append(detail)
        combined_error = "All seeded runs failed. Details:\n" + "\n".join(error_details)
        return {
            "success": False,
            "error": combined_error,
            "runs": runs,
            "aggregate_metrics": {},
        }

    metric_series: Dict[str, List[float]] = {}
    for r in successes:
        metrics = (r.get("parsed") or {}).get("metrics") or {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                metric_series.setdefault(k, []).append(float(v))

    if not metric_series:
        return {
            "success": False,
            "error": "Code executed but produced no parseable metrics - check stdout JSON format",
            "runs": runs,
            "aggregate_metrics": {},
        }

    aggregate = {}
    for k, vals in metric_series.items():
        arr = np.array(vals, dtype=float)
        aggregate[k] = {
            "mean": float(arr.mean()),
            "std": float(arr.std(ddof=1)) if len(arr) > 1 else 0.0,
            "values": vals,
            "n": len(vals),
        }

    return {
        "success": True,
        "n_seeds": n_seeds,
        "n_success": len(successes),
        "runs": runs,
        "aggregate_metrics": aggregate,
        "raw_results": [r.get("parsed") for r in successes],
    }
