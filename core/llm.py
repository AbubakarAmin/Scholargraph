"""
Multi-provider LLM client: Gemini, OpenAI, and any OpenAI-compatible endpoint.
Agents should use call_llm / generate_embedding / get_llm_client — not provider SDKs directly.

Rate limiting and circuit breaking are handled by core.api_gateway.
"""

from __future__ import annotations

import logging
import os
import re
import time
from typing import Any, Optional

import numpy as np

from .api_gateway import get_gateway
from .config import config

logger = logging.getLogger(__name__)

_MAX_RETRIES = int(os.getenv("LLM_MAX_RETRIES", "3"))
_RETRY_BACKOFF = float(os.getenv("LLM_RETRY_BACKOFF", "2.0"))
_THINKING_TAG_RE = re.compile(r"<think>[\s\S]*?</think>", re.IGNORECASE)


class LLMClient:
    """Thin wrapper around Gemini or OpenAI-compatible chat + embeddings."""

    def __init__(self, provider: Optional[str] = None):
        self.provider = (provider or config.llm_provider or "gemini").lower()
        self._gemini = None
        self._openai = None
        self._init_clients()

    def _init_clients(self):
        # Always init Gemini client — embeddings always use Gemini
        from google import genai
        if config.google_api_key:
            self._gemini = genai.Client(api_key=config.google_api_key)
        else:
            self._gemini = None

        if self.provider == "gemini":
            if not self._gemini:
                raise ValueError("GOOGLE_API_KEY required for Gemini provider")
        elif self.provider in ("openai", "openai_compatible"):
            from openai import OpenAI

            if not config.openai_api_key:
                raise ValueError("OPENAI_API_KEY required for OpenAI-compatible provider")
            kwargs = {"api_key": config.openai_api_key, "max_retries": 0, "timeout": 300.0}
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
        gateway = get_gateway()
        try:
            result = gateway.request(
                "llm",
                self._chat_raw,
                prompt, temperature, model_id, max_tokens, system,
            )
            if result:
                result = _THINKING_TAG_RE.sub("", result).strip()
            return result
        except Exception as e:
            logger.warning("LLM chat failed (%s/%s): %s", self.provider, model_id, e)
            try:
                from .run_log import get_tracker

                tracker = get_tracker()
                if tracker:
                    tracker.bump("llm_failures")
                    tracker.message(f"LLM call failed ({self.provider}/{model_id}): {e}", level="error")
            except Exception:
                pass
            return ""

    def _chat_raw(
        self,
        prompt: str,
        temperature: float,
        model_id: str,
        max_tokens: int,
        system: Optional[str],
    ) -> str:
        if self.provider == "gemini":
            return self._chat_gemini(prompt, temperature, model_id, max_tokens, system)
        return self._chat_openai(prompt, temperature, model_id, max_tokens, system)

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
        stream = self._openai.chat.completions.create(
            model=model_id,
            messages=messages,
            temperature=temperature,
            max_tokens=max_tokens,
            stream=True,
        )
        collected = []
        for chunk in stream:
            if not chunk.choices:
                continue
            delta = chunk.choices[0].delta
            if delta and delta.content is not None:
                collected.append(delta.content)
        return "".join(collected)

    def embed(self, text: str, model: Optional[str] = None) -> np.ndarray:
        """Generate embedding. Always uses Gemini — regardless of LLM provider."""
        try:
            return self._embed_gemini(text, model)
        except Exception as e:
            logger.warning("Gemini embedding failed: %s", e)
            return np.zeros(config.embedding_dimension)

    def _embed_raw(self, text: str, model: Optional[str]) -> np.ndarray:
        if self.provider == "gemini":
            return self._embed_gemini(text, model)
        return self._embed_openai(text, model)

    def _embed_gemini(self, text: str, model: Optional[str]) -> np.ndarray:
        model_id = model or config.gemini_embedding_model
        model_name = model_id if model_id.startswith("models/") else f"models/{model_id}"
        response = self._gemini.models.embed_content(
            model=model_name,
            contents=text,
            config={"output_dimensionality": config.embedding_dimension},
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
                "Gemini embedding returned %d dims, expected %d. Adjusting.",
                actual_dim, config.embedding_dimension,
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
                "OpenAI embedding returned %d dims, expected %d. Adjusting.",
                actual_dim, config.embedding_dimension,
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
    # Estimate input tokens (~4 chars per token)
    input_tokens_est = len(prompt) // 4
    if system:
        input_tokens_est += len(system) // 4
    result = llm.chat(
        prompt,
        temperature=temperature,
        model=model_id,
        max_tokens=max_tokens,
        system=system,
    )
    # Estimate output tokens (~4 chars per token)
    output_tokens_est = len(result) // 4 if result else 0
    try:
        from .run_log import get_tracker

        tracker = get_tracker()
        if tracker:
            tracker.bump("llm_calls")
            tracker.bump("llm_tokens_in", amount=input_tokens_est)
            tracker.bump("llm_tokens_out", amount=output_tokens_est)
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
