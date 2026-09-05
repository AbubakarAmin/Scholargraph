"""
Restricted Python sandbox for Engineer experiments.
Blocks exit(), subprocess, os.system, and other dangerous calls.
Does NOT rely on the model behaving — AST + builtins lockdown.
"""

from __future__ import annotations

import ast
import builtins
import io
import json
import traceback
from contextlib import redirect_stdout, redirect_stderr
from typing import Any, Dict, List, Set, Tuple

from .config import config

FORBIDDEN_NAMES: Set[str] = {
    "exit",
    "quit",
    "open",  # replaced with safe_open for read-only of allowed paths if needed
    "exec",
    "eval",
    "compile",
    "__import__",
    "breakpoint",
    "input",
    "help",
}

FORBIDDEN_MODULES: Set[str] = {
    "subprocess",
    "multiprocessing",
    "ctypes",
    "socket",
    "http",
    "urllib",
    "requests",
    "os",
    "sys",
    "shutil",
    "pathlib",
    "importlib",
    "pty",
    "fcntl",
    "signal",
    "pickle",
    "shelve",
    "tempfile",
}

ALLOWED_IMPORT_ROOTS: Set[str] = {
    "numpy",
    "np",
    "pandas",
    "pd",
    "matplotlib",
    "plt",
    "sklearn",
    "scipy",
    "math",
    "statistics",
    "random",
    "json",
    "re",
    "collections",
    "itertools",
    "functools",
    "typing",
    "dataclasses",
    "copy",
    "time",
    "datetime",
    "hashlib",
    "decimal",
    "fractions",
    "string",
    "warnings",
    "textwrap",
    "heapq",
    "bisect",
    "array",
    "struct",
    "operator",
    "seaborn",
    "networkx",
    "sympy",
}


class SandboxViolation(Exception):
    pass


class SandboxASTValidator(ast.NodeVisitor):
    """Reject dangerous AST patterns before execution."""

    def visit_Import(self, node: ast.Import):
        for alias in node.names:
            root = alias.name.split(".")[0]
            if root in FORBIDDEN_MODULES or root not in ALLOWED_IMPORT_ROOTS:
                raise SandboxViolation(f"Import blocked: {alias.name}")
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        if node.module:
            root = node.module.split(".")[0]
            if root in FORBIDDEN_MODULES or root not in ALLOWED_IMPORT_ROOTS:
                raise SandboxViolation(f"Import blocked: {node.module}")
        self.generic_visit(node)

    def visit_Call(self, node: ast.Call):
        # Block exit()/quit() and attribute calls like os.system / subprocess.run
        if isinstance(node.func, ast.Name) and node.func.id in ("exit", "quit", "exec", "eval", "compile", "__import__"):
            raise SandboxViolation(f"Call blocked: {node.func.id}()")
        if isinstance(node.func, ast.Attribute):
            if node.func.attr in ("system", "popen", "remove", "rmdir", "unlink", "chdir", "kill"):
                raise SandboxViolation(f"Attribute call blocked: .{node.func.attr}()")
            if isinstance(node.func.value, ast.Name) and node.func.value.id in FORBIDDEN_MODULES:
                raise SandboxViolation(f"Module call blocked: {node.func.value.id}.{node.func.attr}")
        self.generic_visit(node)

    def visit_Attribute(self, node: ast.Attribute):
        if node.attr.startswith("__") and node.attr.endswith("__"):
            if node.attr not in ("__name__", "__doc__", "__class__", "__dict__", "__len__", "__iter__", "__getitem__"):
                # Allow common dunders used by libraries but block __builtins__ etc.
                if node.attr in ("__builtins__", "__import__", "__subclasses__", "__globals__", "__code__"):
                    raise SandboxViolation(f"Dunder access blocked: {node.attr}")
        self.generic_visit(node)


def validate_code(code: str) -> Tuple[bool, str]:
    try:
        tree = ast.parse(code)
        SandboxASTValidator().visit(tree)
        return True, ""
    except SandboxViolation as e:
        return False, str(e)
    except SyntaxError as e:
        return False, f"SyntaxError: {e}"


def _safe_builtins() -> Dict[str, Any]:
    allowed = {
        "abs", "all", "any", "bool", "bytes", "callable", "chr", "complex",
        "dict", "divmod", "enumerate", "filter", "float", "format", "frozenset",
        "hasattr", "hash", "hex", "int", "isinstance", "issubclass", "iter",
        "len", "list", "map", "max", "min", "next", "oct", "ord", "pow",
        "print", "range", "repr", "reversed", "round", "set", "slice",
        "sorted", "str", "sum", "tuple", "type", "zip", "True", "False", "None",
        "Exception", "ValueError", "TypeError", "RuntimeError", "AssertionError",
        "StopIteration", "KeyError", "IndexError", "AttributeError",
    }
    ns = {name: getattr(builtins, name) for name in allowed if hasattr(builtins, name)}
    # Provide a restricted __import__ that only allows allowlisted modules
    def _restricted_import(name, globals=None, locals=None, fromlist=(), level=0):
        root = name.split(".")[0]
        if root in FORBIDDEN_MODULES or root not in ALLOWED_IMPORT_ROOTS:
            raise SandboxViolation(f"Import blocked at runtime: {name}")
        return builtins.__import__(name, globals, locals, fromlist, level)

    ns["__import__"] = _restricted_import
    return ns


def execute_sandboxed(
    code: str,
    timeout_sec: int = None,
    seed: int = 42,
) -> Dict[str, Any]:
    """
    Execute code in-process with restricted builtins and AST validation.
    Note: true OS-level isolation would require containers; this blocks the
    documented Agent Laboratory failure modes (exit, subprocess, host installs).
    """
    timeout_sec = timeout_sec or config.sandbox_timeout_sec
    ok, err = validate_code(code)
    if not ok:
        return {"success": False, "error": f"Sandbox rejection: {err}", "stdout": "", "stderr": err}

    # Inject seed preamble
    preamble = (
        f"import random as _sg_random\n"
        f"import numpy as _sg_np\n"
        f"_sg_random.seed({seed})\n"
        f"_sg_np.random.seed({seed})\n"
    )
    full_code = preamble + "\n" + code

    stdout_buf = io.StringIO()
    stderr_buf = io.StringIO()
    local_ns: Dict[str, Any] = {"__name__": "__sandbox__"}
    global_ns: Dict[str, Any] = {"__builtins__": _safe_builtins()}

    try:
        # Soft wall-clock timeout (threads cannot be hard-killed on Windows, but
        # the graph stops waiting so engineering no longer hangs forever).
        compiled = compile(full_code, "<sandbox>", "exec")

        def _run():
            with redirect_stdout(stdout_buf), redirect_stderr(stderr_buf):
                exec(compiled, global_ns, local_ns)

        from concurrent.futures import ThreadPoolExecutor, TimeoutError as FuturesTimeout

        with ThreadPoolExecutor(max_workers=1) as pool:
            fut = pool.submit(_run)
            try:
                fut.result(timeout=timeout_sec)
            except FuturesTimeout:
                return {
                    "success": False,
                    "error": f"Sandbox timeout after {timeout_sec}s — experiment code ran too long",
                    "stdout": stdout_buf.getvalue(),
                    "stderr": f"timeout:{timeout_sec}",
                    "seed": seed,
                    "timeout": True,
                }

        stdout = stdout_buf.getvalue()
        stderr = stderr_buf.getvalue()
        if len(stdout) > config.sandbox_max_output_bytes:
            stdout = stdout[: config.sandbox_max_output_bytes] + "\n...[truncated]"
        parsed = _parse_json_from_stdout(stdout)
        return {
            "success": True,
            "stdout": stdout,
            "stderr": stderr,
            "parsed": parsed,
            "seed": seed,
        }
    except SandboxViolation as e:
        return {
            "success": False,
            "error": f"Sandbox violation: {e}",
            "stdout": stdout_buf.getvalue(),
            "stderr": str(e),
            "seed": seed,
        }
    except Exception as e:
        return {
            "success": False,
            "error": str(e),
            "traceback": traceback.format_exc(),
            "stdout": stdout_buf.getvalue(),
            "stderr": stderr_buf.getvalue() + "\n" + traceback.format_exc(),
            "seed": seed,
        }


def _parse_json_from_stdout(stdout: str) -> Dict[str, Any]:
    for line in reversed(stdout.strip().splitlines()):
        line = line.strip()
        if line.startswith("{") and line.endswith("}"):
            try:
                return json.loads(line)
            except json.JSONDecodeError:
                continue
    return {}


def run_multi_seed(
    code: str,
    n_seeds: int = None,
    base_seed: int = 42,
) -> Dict[str, Any]:
    """Run the same experiment with multiple seeds; aggregate mean ± std."""
    import numpy as np

    n_seeds = n_seeds or config.experiment_seeds
    runs: List[Dict[str, Any]] = []
    for i in range(n_seeds):
        seed = base_seed + i * 1009
        result = execute_sandboxed(code, seed=seed)
        runs.append(result)

    successes = [r for r in runs if r.get("success")]
    if not successes:
        return {
            "success": False,
            "error": "All seeded runs failed",
            "runs": runs,
            "aggregate_metrics": {},
        }

    # Aggregate numeric metrics from parsed JSON
    metric_series: Dict[str, List[float]] = {}
    for r in successes:
        metrics = (r.get("parsed") or {}).get("metrics") or {}
        for k, v in metrics.items():
            if isinstance(v, (int, float)):
                metric_series.setdefault(k, []).append(float(v))

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
