"""
Shared utilities and common operations.
Use core.llm for LLM provider access and compatibility aliases.
"""

import json
import logging
import re
from collections import Counter
from datetime import datetime
from typing import List, Dict, Any, Optional

import numpy as np

from .config import config
from .llm import generate_embedding

logging.basicConfig(
    level=getattr(logging, config.log_level),
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def save_json(data: Dict[str, Any], filepath: str):
    try:
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved data to {filepath}")
    except Exception as e:
        logger.error(f"Failed to save JSON to {filepath}: {e}")


def load_json(filepath: str) -> Dict[str, Any]:
    try:
        with open(filepath, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load JSON from {filepath}: {e}")
        return {}


def extract_citations(text: str) -> List[str]:
    patterns = [
        r"\[([^\]]+)\]",
        r"\(([^)]+)\)",
        r"Author et al\.\s+\d{4}",
    ]
    citations = []
    for pattern in patterns:
        citations.extend(re.findall(pattern, text))
    return list(set(citations))


def validate_math_expression(expression: str) -> bool:
    try:
        import sympy as sp

        sp.sympify(expression)
        return True
    except Exception:
        return False


def verify_math_derivation(steps: List[str], start: str, end: str) -> Dict[str, Any]:
    """SymPy check that symbolic simplification of start reaches end (when parseable)."""
    try:
        import sympy as sp

        start_e = sp.simplify(sp.sympify(start))
        end_e = sp.simplify(sp.sympify(end))
        equal = sp.simplify(start_e - end_e) == 0
        return {"passed": bool(equal), "start": str(start_e), "end": str(end_e)}
    except Exception as e:
        return {"passed": False, "error": str(e)}


def sanitize_filename(filename: str) -> str:
    sanitized = re.sub(r'[<>:"/\\|?*]', "_", filename)
    return sanitized[:100]


def create_timestamped_filename(prefix: str, extension: str) -> str:
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{prefix}_{timestamp}.{extension}"


def format_section_text(section_name: str, content: str, level: int = 1) -> str:
    if level == 1:
        return f"\\section{{{section_name}}}\n\n{content}\n"
    if level == 2:
        return f"\\subsection{{{section_name}}}\n\n{content}\n"
    if level == 3:
        return f"\\subsubsection{{{section_name}}}\n\n{content}\n"
    return f"\\paragraph{{{section_name}}}\n\n{content}\n"


def extract_keywords(text: str, max_keywords: int = 10) -> List[str]:
    stop_words = {
        "the", "a", "an", "and", "or", "but", "in", "on", "at", "to", "for",
        "of", "with", "by", "is", "are", "was", "were", "be", "been", "being",
        "have", "has", "had", "do", "does", "did", "will", "would", "could",
        "should", "may", "might", "can", "this", "that", "these", "those",
    }
    words = re.findall(r"\b[a-zA-Z]{3,}\b", text.lower())
    words = [w for w in words if w not in stop_words]
    return [w for w, _ in Counter(words).most_common(max_keywords)]


def calculate_similarity(text1: str, text2: str) -> float:
    try:
        embedding1 = generate_embedding(text1)
        embedding2 = generate_embedding(text2)
        denom = np.linalg.norm(embedding1) * np.linalg.norm(embedding2)
        if denom == 0:
            return 0.0
        return float(np.dot(embedding1, embedding2) / denom)
    except Exception as e:
        logger.error(f"Failed to calculate similarity: {e}")
        return 0.0


def parse_json_from_llm(response: str) -> Optional[Any]:
    """Extract JSON object/array from an LLM response."""
    if not response:
        return None
    try:
        if "[" in response and response.find("[") < (response.find("{") if "{" in response else 10**9):
            start, end = response.find("["), response.rfind("]") + 1
            return json.loads(response[start:end])
        if "{" in response:
            start, end = response.find("{"), response.rfind("}") + 1
            return json.loads(response[start:end])
    except json.JSONDecodeError:
        return None
    return None


# Safety refusals / empty stubs that must never be treated as valid agent output.
_DEGENERATE_LLM_PATTERNS = (
    re.compile(r"^\s*user\s+safety\s*:\s*safe\s*$", re.I),
    re.compile(r"^\s*safe\s*$", re.I),
    re.compile(r"i\s+can'?t\s+(help|assist)\s+with\s+that", re.I),
    re.compile(r"as\s+an\s+ai\s+(language\s+)?model", re.I),
)


def strip_markdown_headers(text: str) -> str:
    """Remove leading markdown headings so length checks measure body prose."""
    lines = []
    for line in str(text or "").splitlines():
        if re.match(r"^\s*#{1,6}\s+", line):
            continue
        lines.append(line)
    return "\n".join(lines).strip()


def is_degenerate_llm_output(text: Any, *, min_chars: int = 40) -> bool:
    """True when an LLM response is empty, a safety stub, or otherwise non-substantive."""
    if text is None:
        return True
    if not isinstance(text, str):
        text = str(text)
    body = strip_markdown_headers(text)
    if len(body) < min_chars:
        return True
    compact = re.sub(r"\s+", " ", body).strip()
    for pattern in _DEGENERATE_LLM_PATTERNS:
        if pattern.search(compact):
            return True
    # Repeated header-only stubs like "# Related Work\\n\\n# Related Work\\n\\nUser Safety: safe"
    if re.search(r"user\s+safety\s*:\s*safe", compact, re.I) and len(compact) < 120:
        return True
    return False


def title_token_overlap(a: str, b: str) -> float:
    """Jaccard overlap of title tokens; used to de-duplicate failed-debate topics."""
    def tokens(value: str) -> set:
        return {t for t in re.findall(r"[a-z0-9]+", (value or "").lower()) if len(t) > 2}

    left, right = tokens(a), tokens(b)
    if not left or not right:
        return 0.0
    return len(left & right) / len(left | right)


def log_agent_action(agent_name: str, action: str, details: Dict[str, Any] = None):
    from .run_log import emit_event, get_tracker

    details = details or {}
    logger.info(f"Agent {agent_name}: {action}")
    if config.debug_mode:
        logger.debug(f"Details: {json.dumps(details, default=str)[:500]}")
    tracker = get_tracker()
    run_id = tracker.run_id if tracker else None
    emit_event(
        "agent_action",
        {"action": action, "details": details},
        run_id=run_id,
        agent=agent_name,
    )
