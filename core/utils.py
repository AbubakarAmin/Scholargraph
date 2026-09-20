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
    """Extract JSON object/array from an LLM response.

    Returns dict OR list — callers must type-check the result before calling
    dict-only methods like ``.get`` (a bare JSON array is a valid model output).
    Strips `` blocks before extraction to prevent thinking-trace contamination.
    Prefers the LAST complete JSON block (reasoning models emit thinking first).
    """
    if not isinstance(response, str) or not response:
        return None
    response = strip_thinking_tags(response)
    # After stripping thinking tags, also strip any residual chain-of-thought
    # that may not be inside formal <think> tags (reasoning models sometimes
    # emit plain-text reasoning before the JSON).
    cleaned = response
    # Find all JSON blocks (object or array) in the response.
    # Track (start, end, parsed_value) so we can pick the outermost block.
    json_blocks: list = []
    # Try objects
    for m in re.finditer(r'\{', cleaned):
        start = m.start()
        # Find matching closing brace from the end
        end = cleaned.rfind('}', start)
        if end > start:
            candidate = cleaned[start:end + 1]
            try:
                parsed = json.loads(candidate)
                json_blocks.append((start, end, parsed))
            except Exception:
                pass
    # Try arrays
    for m in re.finditer(r'\[', cleaned):
        start = m.start()
        end = cleaned.rfind(']', start)
        if end > start:
            candidate = cleaned[start:end + 1]
            try:
                parsed = json.loads(candidate)
                json_blocks.append((start, end, parsed))
            except Exception:
                pass
    if json_blocks:
        # Prefer the outermost block (smallest start, largest end span).
        # When spans are equal (same block found as both dict and array),
        # prefer the one found later (array closing bracket tends to be
        # more reliable for arrays wrapping dicts).
        outermost = max(json_blocks, key=lambda b: (b[1] - b[0], -b[0]))
        return outermost[2]
    return None


def call_llm_json(
    prompt: str,
    *,
    attempts: int = 3,
    temperature: float = 0.3,
    tier: str = "default",
    client: Any = None,
    model: Any = None,
    system: Optional[str] = None,
    max_tokens: int = 8192,
    call_fn: Optional[Any] = None,
) -> Optional[Any]:
    """Self-correcting structured-output call.

    Parses the LLM response as JSON; when parsing fails, re-asks with the
    parse error and the offending response excerpt appended so the model can
    repair its own malformed output (control-data flow separation: the parse
    protocol stays deterministic, only the content is re-generated).
    Returns None only after exhausting attempts.
    """
    if call_fn is None:
        from .llm import call_llm as call_fn

    last_response = ""
    last_error = "empty response"
    for attempt in range(max(1, attempts)):
        if attempt == 0:
            current_prompt = prompt
        else:
            current_prompt = (
                f"{prompt}\n\nYour previous response could not be parsed as JSON."
                f"\nParse problem: {last_error}"
                f"\nPrevious response (truncated): {str(last_response)[:400]}"
                "\nReturn ONLY the valid JSON object or array now — no prose, no markdown fences."
            )
        try:
            last_response = call_fn(
                prompt=current_prompt,
                client=client,
                temperature=max(0.0, temperature - 0.1 * attempt),
                model=model,
                tier=tier,
                system=system,
                max_tokens=max_tokens,
            )
        except TypeError:
            # Injectable stubs may not accept the full keyword set.
            last_response = call_fn(current_prompt)
        try:
            parsed = parse_json_from_llm(last_response)
        except Exception as exc:  # defensive: parse must never raise
            parsed = None
            last_error = str(exc)
        else:
            if parsed is not None:
                return parsed
            last_error = "no JSON object or array found in response"
    return None


# Safety refusals / empty stubs that must never be treated as valid agent output.
_DEGENERATE_LLM_PATTERNS = (
    re.compile(r"^\s*user\s+safety\s*:\s*safe\s*$", re.I),
    re.compile(r"^\s*safe\s*$", re.I),
    re.compile(r"i\s+can'?t\s+(help|assist)\s+with\s+that", re.I),
    re.compile(r"as\s+an\s+ai\s+(language\s+)?model", re.I),
)


_THINKING_TAG_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


def strip_thinking_tags(text: str) -> str:
    """Remove LLM chain-of-thought `` blocks from output.

    Models like Qwen/DeepSeek wrap internal reasoning in `` tags.
    These leak into stored arguments and corrupt JSON extraction.
    """
    if not text:
        return text
    return _THINKING_TAG_RE.sub("", text).strip()


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
    text = strip_thinking_tags(text)
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
