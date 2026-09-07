"""Replay companion experiments from a reproducibility manifest."""

from __future__ import annotations

import json
import os
import subprocess
import sys
import venv
from pathlib import Path
from typing import Any, Dict, List, Optional

from .sandbox import execute_sandboxed


def replay_companion(companion_dir: str, manifest_name: str = "reproducibility_manifest.json", clean_env: bool = False) -> Dict[str, Any]:
    root = Path(companion_dir).resolve()
    manifest_path = root / manifest_name
    if not manifest_path.is_file():
        return {"passed": False, "error": "missing reproducibility manifest", "experiments": []}
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    experiments: List[Dict[str, Any]] = []
    python_executable = sys.executable
    environment = {"mode": "current_process", "python": sys.version}
    if clean_env:
        env_dir = root / ".replay_venv"
        if not env_dir.exists():
            venv.EnvBuilder(with_pip=False, clear=False).create(env_dir)
        python_executable = str(env_dir / ("Scripts/python.exe" if os.name == "nt" else "bin/python"))
        environment = {"mode": "clean_venv", "python": python_executable}
    for code_path in sorted((root / "experiments").glob("*.py")):
        if clean_env:
            completed = subprocess.run([python_executable, str(code_path)], cwd=root, capture_output=True, text=True, timeout=120)
            result = {"success": completed.returncode == 0, "stdout": completed.stdout, "stderr": completed.stderr, "returncode": completed.returncode}
        else:
            result = execute_sandboxed(code_path.read_text(encoding="utf-8"), seed=0)
        experiments.append({"name": code_path.stem, "success": bool(result.get("success")), "result": result})
    passed = bool(experiments) and all(item["success"] for item in experiments)
    return {
        "passed": passed,
        "manifest": manifest,
        "environment": environment,
        "experiments": experiments,
        "error": None if passed else "one or more companion experiments failed replay",
    }
