"""Auditable execution boundary for agent capabilities."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import datetime, timezone
from typing import Any, Callable, Dict, Mapping

from .capabilities import authorize


@dataclass(frozen=True)
class ToolCall:
    agent: str
    capability: str
    tool: str
    status: str
    reason: str
    started_at: str


class CapabilityBroker:
    """Dispatch registered tools only after manifest authorization."""

    def __init__(self, tools: Mapping[str, Callable[..., Any]] | None = None):
        self._tools: Dict[str, Callable[..., Any]] = dict(tools or {})
        self.calls: list[ToolCall] = []

    def register(self, tool: str, handler: Callable[..., Any]) -> None:
        if not tool.strip():
            raise ValueError("Tool name cannot be empty")
        self._tools[tool] = handler

    def call(
        self,
        manifest: Mapping[str, object],
        capability: str,
        tool: str,
        **kwargs: Any,
    ) -> Any:
        decision = authorize(manifest, capability)
        started_at = datetime.now(timezone.utc).isoformat()
        if not decision.allowed:
            self.calls.append(ToolCall(
                decision.agent, capability, tool, "denied", decision.reason, started_at,
            ))
            raise PermissionError(f"{decision.agent} cannot use {capability}: {decision.reason}")
        if tool not in self._tools:
            self.calls.append(ToolCall(
                decision.agent, capability, tool, "missing", "tool not registered", started_at,
            ))
            raise KeyError(f"Tool is not registered: {tool}")
        self.calls.append(ToolCall(
            decision.agent, capability, tool, "started", "authorized", started_at,
        ))
        return self._tools[tool](**kwargs)
