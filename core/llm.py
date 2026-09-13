"""
Multi-provider LLM client: Gemini, OpenAI, and any OpenAI-compatible endpoint.
Agents should use call_llm / generate_embedding / get_llm_client — not provider SDKs directly.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from typing import Any, Optional

import numpy as np

from .config import config

logger = logging.getLogger(__name__)

_last_request_time = 0.0
_request_interval = float(os.getenv("LLM_REQUEST_INTERVAL", "0.15"))
_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
_RETRY_BACKOFF = float(os.getenv("LLM_RETRY_BACKOFF", "2.0"))
_rate_lock = threading.Lock()


class LLMClient:
    """Thin wrapper around Gemini or OpenAI-compatible chat + embeddings."""

    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or config.llm_provider or "gemini").lower()
        self._gemini = None
        self._openai = None
        self._init_clients()

    def _init_clients(self):
        if self.provider == "gemini":
            from google import genai

            if not config.google_api_key:
                raise ValueError("GOOGLE_API_KEY required for Gemini provider")
            self._gemini = genai.Client(api_key=config.google_api_key)
        elif self.provider in ("openai", "openai_compatible"):
            from openai import OpenAI

            if not config.openai_api_key:
                raise ValueError("OPENAI_API_KEY required for OpenAI-compatible provider")
            kwargs = {"api_key": config.openai_api_key}
            if config.openai_base_url:
                kwargs["base_url"] = config.openai_base_url
            self._openai = OpenAI(**kwargs)
        else:
            raise ValueError(f"Unknown LLM provider: {self.provider}")

    def chat(
        self,
        prompt: str,
        temperature: float = 0.7,
        model: Optional[str] = None,
        max_tokens: int = 8192,
        system: Optional[str] = None,
    ) -> str:
        model_id = model or config.resolve_model("default")
        last_error = None
        for attempt in range(_MAX_RETRIES):
            _rate_limit()
            try:
                if self.provider == "gemini":
                    return self._chat_gemini(prompt, temperature, model_id, max_tokens, system)
                return self._chat_openai(prompt, temperature, model_id, max_tokens, system)
            except Exception as e:
                last_error = e
                wait = _RETRY_BACKOFF * (2 ** attempt)
                logger.warning(f"LLM chat failed ({self.provider}/{model_id}), attempt {attempt+1}/{_MAX_RETRIES}: {e}. Retrying in {wait:.1f}s")
                time.sleep(wait)
        logger.error(f"LLM chat failed after {_MAX_RETRIES} attempts ({self.provider}/{model_id}): {last_error}")
        return ""

    def _chat_gemini(
        self,
        prompt: str,
        temperature: float,
        model_id: str,
        max_tokens: int,
        system: Optional[str],
    ) -> str:
        contents = prompt
        gen_config: dict = {
            "temperature": temperature,
            "max_output_tokens": max_tokens,
        }
        if system:
            gen_config["system_instruction"] = system
        # Accept both "gemini-2.5-flash" and "models/gemini-2.5-flash"
        model_name = model_id if model_id.startswith("models/") else f"models/{model_id}"
        response = self._gemini.models.generate_content(
            model=model_name,
            contents=contents,
            config=gen_config,
        )
        try:
            return response.candidates[0].content.parts[0].text
        except Exception:
            return getattr(response, "text", "") or ""

    def _chat_openai(
        self,
        prompt: str,
        temperature: float,
        model_id: str,
        max_tokens: int,
        system: Optional[str],
    ) -> str:
        messages = []
        if system:
            messages.append({"role": "system", "content": system})
        messages.append({"role": "user", "content": prompt})
        response = self._openai.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
        )
        return response.choices[0].message.content or ""

    def embed(self, text: str, model: Optional[str] = None) -> np.ndarray:
        last_error = None
        for attempt in range(_MAX_RETRIES):
            _rate_limit()
            try:
                if self.provider == "gemini":
                    return self._embed_gemini(text, model)
                return self._embed_openai(text, model)
            except Exception as e:
                last_error = e
                wait = _RETRY_BACKOFF * (2 ** attempt)
                logger.warning(f"Embedding failed, attempt {attempt+1}/{_MAX_RETRIES}: {e}. Retrying in {wait:.1f}s")
                time.sleep(wait)
        logger.error(f"Embedding failed after {_MAX_RETRIES} attempts: {last_error}")
        return np.zeros(config.embedding_dimension)

    def _embed_gemini(self, text: str, model: Optional[str]) -> np.ndarray:
        model_id = model or config.gemini_embedding_model
        model_name = model_id if model_id.startswith("models/") else f"models/{model_id}"
        response = self._gemini.models.embed_content(
            model=model_name,
            contents=text,
            output_dimensionality=config.embedding_dimension,
        )
        if hasattr(response, "embeddings") and response.embeddings:
            vec = np.array(response.embeddings[0].values, dtype=float)
        elif hasattr(response, "embedding") and response.embedding:
            vals = response.embedding.values
            vec = np.array(vals[0].values if hasattr(vals[0], "values") else vals, dtype=float)
        else:
            vec = np.array(response.values[0].values, dtype=float)
        actual_dim = vec.shape[0]
        if actual_dim != config.embedding_dimension:
            logger.warning(
                f"Gemini embedding returned {actual_dim} dims, expected {config.embedding_dimension}. "
                f"Adjusting to match FAISS index."
            )
            if actual_dim < config.embedding_dimension:
                vec = np.pad(vec, (0, config.embedding_dimension - actual_dim))
            else:
                vec = vec[:config.embedding_dimension]
        return vec

    def _embed_openai(self, text: str, model: Optional[str]) -> np.ndarray:
        model_id = model or config.openai_embedding_model
        response = self._openai.embeddings.create(model=model_id, input=text)
        vec = np.array(response.data[0].embedding, dtype=float)
        actual_dim = vec.shape[0]
        if actual_dim != config.embedding_dimension:
            logger.warning(
                f"OpenAI embedding returned {actual_dim} dims, expected {config.embedding_dimension}. "
                f"Adjusting to match FAISS index."
            )
            if actual_dim < config.embedding_dimension:
                vec = np.pad(vec, (0, config.embedding_dimension - actual_dim))
            else:
                vec = vec[:config.embedding_dimension]
        return vec


_client: Optional[LLMClient] = None


def get_llm_client(force_new: bool = False) -> LLMClient:
    global _client
    if _client is None or force_new:
        _client = LLMClient()
    return _client


def reset_llm_client():
    """Call after runtime key changes from the web UI."""
    global _client
    _client = None


def _rate_limit():
    global _last_request_time
    with _rate_lock:
        now = time.time()
        delta = now - _last_request_time
        if delta < _request_interval:
            sleep_time = _request_interval - delta
        else:
            sleep_time = 0.0
        _last_request_time = time.time()
    if sleep_time > 0.0:
        time.sleep(sleep_time)


def call_llm(
    prompt: str,
    client: Any = None,
    temperature: float = 0.7,
    model: Optional[str] = None,
    tier: str = "default",
    system: Optional[str] = None,
    max_tokens: int = 8192,
) -> str:
    """Primary LLM entry point used by all agents."""
    llm = client if isinstance(client, LLMClient) else get_llm_client()
    model_id = model or config.resolve_model(tier)
    result = llm.chat(
        prompt,
        temperature=temperature,
        model=model_id,
        max_tokens=max_tokens,
        system=system,
    )
    try:
        from .run_log import get_tracker

        tracker = get_tracker()
        if tracker:
            tracker.bump("llm_calls")
    except Exception:
        pass
    return result


def generate_embedding(text: str, model: Any = None) -> np.ndarray:
    """Generate embedding; `model` may be unused legacy gemini client."""
    llm = model if isinstance(model, LLMClient) else get_llm_client()
    return llm.embed(text)


# --- Backward-compatible aliases (old agents import these) ---

def setup_gemini():
    """Legacy: returns LLMClient (works with call_gemini / generate_embedding)."""
    return get_llm_client()


def call_gemini(prompt: str, model: Any = None, temperature: float = 0.7) -> str:
    """Legacy alias for call_llm."""
    return call_llm(prompt, client=model, temperature=temperature)
