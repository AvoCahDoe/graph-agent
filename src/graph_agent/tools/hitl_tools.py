"""Built-in HITL tools."""

from __future__ import annotations

from typing import Any

from langchain_core.tools import tool

from graph_agent.agent.hitl import ask_human
from graph_agent.tools.registry import RiskLevel, register_tool


@register_tool(risk_level=RiskLevel.HITL, description="Ask the user a clarifying question")
@tool
def ask_user(
    question: str,
    options: list[str] | None = None,
) -> str:
    """Pause and ask the human a question. Optionally provide clickable options."""
    return ask_human(question, options=options)


# Ensure module side-effects run on import.
CORE_TOOLS: list[Any] = [ask_user]
