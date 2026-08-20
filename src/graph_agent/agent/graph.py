"""LangGraph agent factory: undo gate → LLM planner → pack subgraphs (loop).

Pack names and behavior come from config/agent.yaml.
Pass checkpointer=None for external persistence (Studio / Agent Server).
"""

from __future__ import annotations

import logging
from typing import Any, Callable, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, SystemMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import END, START, StateGraph

from graph_agent.agent.commands import (
    conversation_note_for_undo,
    format_undo_reply,
    is_undo_message,
)
from graph_agent.agent.planner import make_planner_node
from graph_agent.agent.routing import last_human_text
from graph_agent.agent.specialists import build_specialist_react
from graph_agent.agent.state import AgentState, InteractionMode
from graph_agent.policy import get_policy

logger = logging.getLogger(__name__)

UndoHandler = Callable[[], dict[str, Any] | None]

_undo_handler: UndoHandler | None = None


def set_undo_handler(handler: UndoHandler | None) -> None:
    """Optional integrator hook: return undo result dict or None."""
    global _undo_handler
    _undo_handler = handler


def route_entry(state: AgentState) -> str:
    if is_undo_message(last_human_text(list(state.get("messages") or []))):
        return "undo"
    return "planner"


def undo_node(state: AgentState) -> dict[str, list[BaseMessage]]:
    result = None
    if _undo_handler is not None:
        try:
            result = _undo_handler()
        except Exception as exc:
            logger.warning("Undo handler failed: %s", exc)
            result = None
    messages: list[BaseMessage] = []
    if result:
        messages.append(SystemMessage(content=conversation_note_for_undo(result)))
    messages.append(AIMessage(content=format_undo_reply(result)))
    return {"messages": messages}


def create_agent(
    llm: BaseChatModel,
    mode: InteractionMode = InteractionMode.ASK,
    checkpointer: MemorySaver | None | Literal["auto"] = "auto",
):
    """Build supervisor: undo gate + planner + N specialist subgraphs from config."""
    import graph_agent.tools  # noqa: F401

    policy = get_policy()
    pack_names = policy.pack_names()
    if not pack_names:
        raise RuntimeError("No packs defined in agent.yaml — add at least one under packs:")

    specialists = {
        name: build_specialist_react(llm, mode, name) for name in pack_names
    }
    planner = make_planner_node(llm)

    logger.info("Creating %s agent with packs: %s", mode.value, ", ".join(pack_names))

    builder = StateGraph(AgentState)
    builder.add_node("undo", undo_node)
    builder.add_node("planner", planner)
    for name, subgraph in specialists.items():
        builder.add_node(name, subgraph)

    builder.add_conditional_edges(
        START,
        route_entry,
        {
            "undo": "undo",
            "planner": "planner",
        },
    )
    builder.add_edge("undo", END)
    for name in pack_names:
        builder.add_edge(name, "planner")

    kwargs: dict[str, Any] = {}
    if checkpointer == "auto":
        kwargs["checkpointer"] = MemorySaver()
    elif checkpointer is not None:
        kwargs["checkpointer"] = checkpointer
    return builder.compile(**kwargs)


def agent_factory(llm: BaseChatModel, checkpointer: MemorySaver | None | Literal["auto"] = "auto"):
    """Return a callable that builds an agent for a given mode."""

    def create(mode: InteractionMode):
        return create_agent(llm, mode=mode, checkpointer=checkpointer)

    return create
