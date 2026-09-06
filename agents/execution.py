"""Independent reproducible experiment execution worker."""

from __future__ import annotations

import hashlib
import json
import platform
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Optional

from core.capabilities import DEFAULT_MANIFESTS
from core.context import RunContext, get_active_context
from core.contracts import CodeArtifact, ExecutionArtifact, ExecutionRequest
from core.sandbox import run_multi_seed, validate_code


class ExecutionAgent:
    """Execute approved code and preserve raw, replayable outputs."""

    capability_manifest = DEFAULT_MANIFESTS["ExecutionAgent"]

    def __init__(self, context: Optional[RunContext] = None):
        self.context = context or get_active_context()
        runtime_config = self.context.config if self.context else None
        output_dir = runtime_config.raw_results_dir if runtime_config else "./output/raw_results"
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def execute(
        self,
        code_artifact: CodeArtifact,
        request: ExecutionRequest,
    ) -> ExecutionArtifact:
        code = code_artifact.get("source_code") or code_artifact.get("code") or ""
        experiment_name = request.get("experiment_name") or code_artifact.get("experiment_name", "experiment")
        seeds = request.get("seeds") or self._default_seeds()
        now = datetime.now(timezone.utc).isoformat()
        environment = {
            "python": sys.version,
            "platform": platform.platform(),
            "seeds": seeds,
            "sandbox": "core.sandbox.run_multi_seed",
        }
        ok, error = validate_code(code)
        if not ok:
            return self._artifact(
                experiment_name,
                request,
                environment,
                "failed",
                error=f"Code rejected before execution: {error}",
                created_at=now,
            )

        result = run_multi_seed(code, n_seeds=len(seeds), base_seed=seeds[0])
        if result.get("success") and len(seeds) != result.get("n_seeds"):
            result["requested_seeds"] = seeds
        status = "completed" if result.get("success") else "failed"
        return self._artifact(
            experiment_name,
            request,
            environment,
            status,
            result=result,
            error=result.get("error"),
            created_at=now,
        )

    def _default_seeds(self) -> list[int]:
        runtime_config = self.context.config if self.context else None
        count = runtime_config.experiment_seeds if runtime_config else 3
        return [42 + index * 1009 for index in range(count)]

    def _artifact(
        self,
        experiment_name: str,
        request: ExecutionRequest,
        environment: dict[str, Any],
        status: str,
        *,
        result: Optional[dict[str, Any]] = None,
        error: Optional[str] = None,
        created_at: str,
    ) -> ExecutionArtifact:
        payload = result or {"error": error}
        encoded = json.dumps(payload, sort_keys=True, default=str).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        raw_path = self.output_dir / f"{experiment_name}_{digest[:12]}.json"
        raw_path.write_bytes(encoded)
        return {
            "request": request,
            "raw_results_path": str(raw_path),
            "environment": environment,
            "seed_results": payload,
            "status": status,
            "error": error or "",
            "content_hash": digest,
            "provenance": {
                "artifact_type": "execution",
                "producer": "ExecutionAgent",
                "content_hash": digest,
                "created_at": created_at,
                "status": status,
                "environment": environment,
                "limitations": ["Execution uses the existing in-process sandbox."],
            },
        }