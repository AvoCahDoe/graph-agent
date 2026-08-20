"""Tool registry with risk-level metadata for Ask/Agent routing and HITL."""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any, Callable

from graph_agent.policy import get_policy


class RiskLevel(str, Enum):
    READ_ONLY = "read_only"
    HITL = "hitl"
    MUTATING = "mutating"
    DESTRUCTIVE = "destructive"


@dataclass
class ToolMetadata:
    risk_level: RiskLevel
    supports_dry_run: bool = False
    description: str = ""


class ToolRegistry:
    """Tracks LangChain tools plus risk metadata."""

    def __init__(self) -> None:
        self._tools: dict[str, tuple[Any, ToolMetadata]] = {}

    def register(self, tool: Any, metadata: ToolMetadata) -> Any:
        name = getattr(tool, "name", None)
        if not name:
            raise ValueError("Tool must have a name")
        self._tools[name] = (tool, metadata)
        return tool

    def get_all_tools(self) -> list[Any]:
        return [tool for tool, _ in self._tools.values()]

    def get_tools_by_risk(self, risk: str | RiskLevel) -> list[Any]:
        value = risk.value if isinstance(risk, RiskLevel) else str(risk).lower()
        return [tool for tool in self.get_all_tools() if self.get_risk(tool.name).value == value]

    def get_tools_for_mode(self, mode: str) -> list[Any]:
        policy = get_policy()
        extra = set(policy.mode(mode).extra_tools)
        selected: list[Any] = []
        for tool in self.get_all_tools():
            name = tool.name
            risk = self.get_risk(name).value
            if name in extra or policy.allows_risk(mode, risk):
                selected.append(tool)
        return selected

    def get_tools_by_names(self, names: set[str] | frozenset[str] | list[str]) -> list[Any]:
        wanted = set(names)
        return [tool for tool in self.get_all_tools() if tool.name in wanted]

    def get_tools_for_mode_and_names(self, mode: str, names: set[str] | frozenset[str]) -> list[Any]:
        wanted = set(names)
        return [tool for tool in self.get_tools_for_mode(mode) if tool.name in wanted]

    def get_metadata(self, name: str) -> ToolMetadata | None:
        entry = self._tools.get(name)
        return entry[1] if entry else None

    def get_risk(self, name: str) -> RiskLevel:
        override = get_policy().tool(name).risk
        if override:
            try:
                return RiskLevel(override)
            except ValueError:
                pass
        meta = self.get_metadata(name)
        return meta.risk_level if meta else RiskLevel.READ_ONLY

    def names(self) -> list[str]:
        return list(self._tools.keys())


tool_registry = ToolRegistry()


def register_tool(
    risk_level: RiskLevel,
    supports_dry_run: bool = False,
    description: str = "",
) -> Callable[[Any], Any]:
    """Decorator: register an already-built LangChain tool."""

    def decorator(tool: Any) -> Any:
        tool_registry.register(
            tool,
            ToolMetadata(
                risk_level=risk_level,
                supports_dry_run=supports_dry_run,
                description=description or getattr(tool, "description", ""),
            ),
        )
        return tool

    return decorator
