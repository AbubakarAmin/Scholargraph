"""
Multi-provider LLM client: Gemini, OpenAI, and any OpenAI-compatible endpoint.
Agents should use call_llm / generate_embedding / get_llm_client — not provider SDKs directly.
"""

from __future__ import annotations

import logging
import os
import time
from typing import Any, Optional

import numpy as np

from .config import config

logger = logging.getLogger(__name__)

_last_request_time = 0.0
_request_interval = float(os.getenv("LLM_REQUEST_INTERVAL", "1.0"))


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
        _rate_limit()
        model_id = model or config.resolve_model("default")
        try:
            if self.provider == "gemini":
                return self._chat_gemini(prompt, temperature, model_id, max_tokens, system)
            return self._chat_openai(prompt, temperature, model_id, max_tokens, system)
        except Exception as e:
            logger.error(f"LLM chat failed ({self.provider}/{model_id}): {e}")
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
        _rate_limit()
        try:
            if self.provider == "gemini":
                return self._embed_gemini(text, model)
            return self._embed_openai(text, model)
        except Exception as e:
            logger.error(f"Embedding failed: {e}")
            return np.zeros(768)

    def _embed_gemini(self, text: str, model: Optional[str]) -> np.ndarray:
        model_id = model or config.gemini_embedding_model
        model_name = model_id if model_id.startswith("models/") else f"models/{model_id}"
        response = self._gemini.models.embed_content(
            model=model_name,
            contents=text,
        )
        if hasattr(response, "embeddings") and response.embeddings:
            return np.array(response.embeddings[0].values)
        if hasattr(response, "embedding") and response.embedding:
            vals = response.embedding.values
            return np.array(vals[0].values if hasattr(vals[0], "values") else vals)
        return np.array(response.values[0].values)

    def _embed_openai(self, text: str, model: Optional[str]) -> np.ndarray:
        model_id = model or config.openai_embedding_model
        response = self._openai.embeddings.create(model=model_id, input=text)
        vec = np.array(response.data[0].embedding, dtype=float)
        # Pad/truncate to 768 for FAISS compatibility with existing indexes
        if vec.shape[0] < 768:
            vec = np.pad(vec, (0, 768 - vec.shape[0]))
        elif vec.shape[0] > 768:
            vec = vec[:768]
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
    now = time.time()
    delta = now - _last_request_time
    if delta < _request_interval:
        time.sleep(_request_interval - delta)
    _last_request_time = time.time()


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
    return llm.chat(
        prompt,
        temperature=temperature,
        model=model_id,
        max_tokens=max_tokens,
        system=system,
    )


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
