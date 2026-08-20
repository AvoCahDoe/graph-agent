"""Supervisor planner: pick next pack or finish (packs from config)."""

from __future__ import annotations

import json
import logging
import os
import re
from typing import Any

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, SystemMessage
from langgraph.graph import END
from langgraph.types import Command
from pydantic import BaseModel, Field

from graph_agent.agent.routing import (
    keyword_route_confident,
    last_human_text,
    pack_scores,
    pending_keyword_packs,
    route_pack_excluding,
    route_pack_from_text,
)
from graph_agent.agent.state import AgentState
from graph_agent.policy import get_policy

logger = logging.getLogger(__name__)

PLANNER_EXCERPT_CHARS = 400


class PlannerDecision(BaseModel):
    next: str = Field(description="Next pack or finish")
    reason: str = ""


def _message_text(msg: BaseMessage | Any) -> str:
    content = getattr(msg, "content", msg)
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return str(content or "")


def _last_is_human(messages: list | None) -> bool:
    if not messages:
        return True
    last = messages[-1]
    if isinstance(last, HumanMessage):
        return True
    return getattr(last, "type", None) == "human"


def _last_ai_excerpt(messages: list | None, limit: int = PLANNER_EXCERPT_CHARS) -> str:
    for msg in reversed(messages or []):
        if isinstance(msg, AIMessage) or getattr(msg, "type", None) == "ai":
            text = _message_text(msg).strip()
            if text:
                return text[:limit]
    return ""


def _parse_decision_json(raw: str, pack_values: set[str]) -> PlannerDecision | None:
    text = (raw or "").strip()
    if not text:
        return None
    fence = re.search(r"```(?:json)?\s*(\{.*?\})\s*```", text, re.DOTALL)
    blob = fence.group(1) if fence else text
    match = re.search(r"\{[^{}]*\}", blob)
    if not match:
        return None
    try:
        data = json.loads(match.group(0))
        nxt = str(data.get("next") or "").strip().lower()
        if nxt == "end":
            nxt = "finish"
        if nxt not in pack_values and nxt != "finish":
            return None
        return PlannerDecision(next=nxt, reason=str(data.get("reason") or ""))
    except (json.JSONDecodeError, ValueError):
        return None


def _planner_keyword_only() -> bool:
    return os.getenv("PLANNER_KEYWORD_ONLY", "").lower() in {"1", "true", "yes"}


def _bind_planner_llm(llm: BaseChatModel) -> BaseChatModel:
    cap = get_policy().llm.planner_max_tokens
    bind = getattr(llm, "bind", None)
    if callable(bind):
        return bind(max_tokens=cap)
    return llm


def _invoke_planner_llm(
    llm: BaseChatModel,
    user_text: str,
    visited: list[str],
    notes: str,
    pack_names: list[str],
) -> PlannerDecision | None:
    remaining = [p for p in pack_names if p not in visited]
    system = get_policy().planner.system_prompt
    user = (
        f"User question: {user_text or '(empty)'}\n"
        f"Already visited: {', '.join(visited) or '(none)'}\n"
        f"Unused packs: {', '.join(remaining) or '(none)'}\n"
        f"Last specialist notes: {notes or '(none)'}"
    )
    messages = [SystemMessage(content=system), HumanMessage(content=user)]
    planner_llm = _bind_planner_llm(llm)
    try:
        response = planner_llm.invoke(messages)
        return _parse_decision_json(_message_text(response), set(pack_names))
    except Exception as exc:
        logger.warning("Planner LLM invoke failed: %s", exc)
        return None


def _fallback_pack(user_text: str, visited: list[str], *, new_turn: bool) -> str:
    picked = route_pack_excluding(user_text, exclude=set(visited))
    if picked is not None:
        return picked
    if new_turn:
        return route_pack_from_text(user_text)
    return "finish"


def _keyword_first_pack(user_text: str, visited: list[str]) -> str | None:
    if _planner_keyword_only():
        return route_pack_excluding(user_text, exclude=set(visited))
    return keyword_route_confident(user_text, exclude=set(visited))


def decide_next_pack(
    llm: BaseChatModel | None,
    state: dict[str, Any] | AgentState,
) -> tuple[str, dict[str, Any]]:
    """Return (goto_target, state_update). goto_target is a pack name or END."""
    policy = get_policy()
    pack_names = policy.pack_names()
    max_hops = policy.agent.max_pack_hops

    messages = list((state or {}).get("messages") or [])
    new_turn = _last_is_human(messages)
    user_text = last_human_text(messages)
    if new_turn:
        hops = 0
        visited: list[str] = []
    else:
        hops = int((state or {}).get("pack_hops") or 0)
        visited = [str(p) for p in ((state or {}).get("visited_packs") or []) if p]

    if hops >= max_hops or len(visited) >= max_hops:
        logger.info("Planner finish: hop cap (hops=%s, visited=%s)", hops, visited)
        return END, {"pack_hops": hops, "visited_packs": visited}

    remaining = [p for p in pack_names if p not in visited]
    if not remaining:
        return END, {"pack_hops": hops, "visited_packs": visited}

    scores = pack_scores(user_text)
    if (
        not new_turn
        and "analytics" in visited
        and scores.get("analytics", 0) > 0
    ):
        logger.info("Planner finish: analytics already handled chart/KPI")
        return END, {"pack_hops": hops, "visited_packs": visited}

    confident = _keyword_first_pack(user_text, visited)
    if confident is not None and confident in remaining:
        logger.info("Planner keyword-first: %s", confident)
        return confident, {"pack_hops": hops + 1, "visited_packs": [*visited, confident]}

    if not new_turn and visited:
        pending = pending_keyword_packs(user_text, visited)
        excerpt = _last_ai_excerpt(messages)
        if not pending and excerpt.strip():
            logger.info("Planner finish heuristic: no pending keyword packs")
            return END, {"pack_hops": hops, "visited_packs": visited}

    decision: PlannerDecision | None = None
    if llm is not None:
        decision = _invoke_planner_llm(
            llm, user_text, visited, _last_ai_excerpt(messages), pack_names
        )

    nxt: str
    if decision is None:
        nxt = _fallback_pack(user_text, visited, new_turn=new_turn)
        logger.info("Planner keyword fallback: %s", nxt)
    else:
        nxt = decision.next
        logger.info("Planner LLM: %s (%s)", nxt, decision.reason)

    if nxt == "finish" or nxt not in remaining:
        if new_turn:
            nxt = _fallback_pack(user_text, visited, new_turn=True)
            if nxt == "finish" or nxt not in remaining:
                nxt = remaining[0]
        else:
            return END, {"pack_hops": hops, "visited_packs": visited}

    next_visited = [*visited, nxt]
    return nxt, {"pack_hops": hops + 1, "visited_packs": next_visited}


def make_planner_node(llm: BaseChatModel):
    """LangGraph node: Command(goto=pack|END) with hop updates."""

    def planner(state: AgentState) -> Command:
        target, update = decide_next_pack(llm, state)
        return Command(goto=target, update=update)

    planner.__name__ = "planner"
    return planner
