"""Reliable, replayable access to allowlisted scholarly sources."""

from __future__ import annotations

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Dict, Mapping, Optional
from urllib.parse import urlparse

import requests

from .contracts import SourceArtifact

logger = logging.getLogger(__name__)


DEFAULT_SOURCE_BASES = {
    "openalex": "https://api.openalex.org/",
    "crossref": "https://api.crossref.org/",
    "arxiv": "https://export.arxiv.org/",
    "semantic_scholar": "https://api.semanticscholar.org/",
}


@dataclass(frozen=True)
class SourcePolicy:
    timeout_seconds: float = 15.0
    retries: int = 2
    max_response_bytes: int = 5_000_000
    allow_full_text: bool = True


class SourceClient:
    """Fetch JSON from approved sources and preserve replayable raw responses."""

    def __init__(
        self,
        cache_dir: str,
        session: Optional[requests.Session] = None,
        source_bases: Optional[Mapping[str, str]] = None,
        policy: Optional[SourcePolicy] = None,
    ):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.session = session or requests.Session()
        self.source_bases = dict(source_bases or DEFAULT_SOURCE_BASES)
        self.policy = policy or SourcePolicy()

    def fetch_json(
        self,
        source: str,
        url: str,
        *,
        params: Optional[Mapping[str, Any]] = None,
        headers: Optional[Mapping[str, str]] = None,
        validator: Optional[Callable[[Any], bool]] = None,
        cache_key: Optional[str] = None,
    ) -> SourceArtifact:
        self._validate_url(source, url)
        key = cache_key or self._cache_key(source, url, params)
        cache_path = self.cache_dir / f"{key}.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached["status"] = "cached"
            return cached

        last_error = "unavailable"
        for attempt in range(self.policy.retries + 1):
            try:
                response = self.session.get(
                    url,
                    params=dict(params or {}),
                    headers=dict(headers or {}),
                    timeout=self.policy.timeout_seconds,
                )
                if response.status_code == 429:
                    retry_after = float(response.headers.get("Retry-After", "5"))
                    wait = min(30.0, max(retry_after, 5.0 * (2 ** attempt)))
                    if attempt < self.policy.retries:
                        logger.warning(
                            "SourceClient 429 from %s — backing off %.1fs (attempt %d/%d)",
                            source, wait, attempt + 1, self.policy.retries + 1,
                        )
                        time.sleep(wait)
                        continue
                response.raise_for_status()
                content_length = len(response.content)
                if content_length > self.policy.max_response_bytes:
                    raise ValueError("response exceeds configured size limit")
                content = response.json()
                if validator and not validator(content):
                    raise ValueError("response failed source validation")
                artifact = self._artifact(source, url, content, "verified")
                cache_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
                return artifact
            except (requests.RequestException, ValueError, json.JSONDecodeError) as exc:
                last_error = str(exc)
                if attempt == self.policy.retries:
                    break

        return self._artifact(source, url, {}, "unavailable", [last_error])

    def fetch_text(self, source: str, url: str, *, headers: Optional[Mapping[str, str]] = None) -> SourceArtifact:
        """Fetch allowlisted text while preserving the same cache/provenance contract."""
        self._validate_url(source, url)
        key = self._cache_key(source, url, None)
        cache_path = self.cache_dir / f"{key}.text.json"
        if cache_path.exists():
            cached = json.loads(cache_path.read_text(encoding="utf-8"))
            cached["status"] = "cached"
            return cached
        try:
            response = self.session.get(url, headers=dict(headers or {}), timeout=self.policy.timeout_seconds)
            response.raise_for_status()
            if len(response.content) > self.policy.max_response_bytes:
                raise ValueError("response exceeds configured size limit")
            artifact = self._artifact(source, url, {"text": response.text}, "verified")
            cache_path.write_text(json.dumps(artifact, indent=2, default=str), encoding="utf-8")
            return artifact
        except (requests.RequestException, ValueError) as exc:
            return self._artifact(source, url, {"text": ""}, "unavailable", [str(exc)])

    def fetch_open_access_text(
        self,
        source: str,
        url: str,
        *,
        license_name: Optional[str] = None,
        headers: Optional[Mapping[str, str]] = None,
    ) -> SourceArtifact:
        """Fetch full text only when the caller has an explicit OA/license signal."""
        if not self.policy.allow_full_text:
            return self._artifact(source, url, {"text": ""}, "unavailable", ["full-text retrieval disabled by policy"])
        if not license_name or license_name.lower() in {"unknown", "closed", "paywalled"}:
            return self._artifact(source, url, {"text": ""}, "unavailable", ["license or open-access permission was not supplied"])
        artifact = self.fetch_text(source, url, headers=headers)
        artifact.setdefault("provenance", {})["license"] = license_name
        artifact.setdefault("provenance", {})["access_policy"] = "open-access-only"
        return artifact

    def _validate_url(self, source: str, url: str) -> None:
        base = self.source_bases.get(source)
        parsed = urlparse(url)
        if not base or parsed.scheme != "https" or not url.startswith(base):
            raise PermissionError(f"URL is not allowlisted for source: {source}")

    @staticmethod
    def _cache_key(source: str, url: str, params: Optional[Mapping[str, Any]]) -> str:
        payload = json.dumps(
            {"source": source, "url": url, "params": dict(params or {})},
            sort_keys=True,
            default=str,
        ).encode("utf-8")
        return hashlib.sha256(payload).hexdigest()

    @staticmethod
    def _artifact(
        source: str,
        url: str,
        content: Dict[str, Any],
        status: str,
        warnings: Optional[list[str]] = None,
    ) -> SourceArtifact:
        encoded = json.dumps(content, sort_keys=True, default=str).encode("utf-8")
        digest = hashlib.sha256(encoded).hexdigest()
        retrieved_at = datetime.now(timezone.utc).isoformat()
        return {
            "source": source,
            "url": url,
            "retrieved_at": retrieved_at,
            "status": status,
            "response_hash": digest,
            "content": content,
            "warnings": warnings or [],
            "provenance": {
                "artifact_type": "source_response",
                "source": url,
                "content_hash": digest,
                "created_at": retrieved_at,
                "status": status,
            },
        }