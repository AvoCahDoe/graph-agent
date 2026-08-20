"""Extensible LangGraph agent state."""

from __future__ import annotations

from enum import Enum
from typing import Any, Optional

from langgraph.graph import MessagesState


class InteractionMode(str, Enum):
    ASK = "ask"
    AGENT = "agent"


class AgentState(MessagesState):
    """Messages plus supervisor hop bookkeeping (planner loop)."""

    mode: InteractionMode = InteractionMode.ASK
    user_confirmed: bool = False
    dry_run_mode: bool = False
    pending_action: Optional[dict[str, Any]] = None
    pack_hops: int = 0
    visited_packs: list[str] = []
