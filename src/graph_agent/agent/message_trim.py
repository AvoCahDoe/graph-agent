"""Trim message history for specialist LLM calls without mutating graph state."""

from __future__ import annotations

from typing import Any

from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage

from graph_agent.policy import get_policy

_CONTENT_CHAR_BUDGET = 6000


def _is_human(msg: BaseMessage | Any) -> bool:
    if isinstance(msg, HumanMessage):
        return True
    return getattr(msg, "type", None) == "human"


def _content_chars(msg: BaseMessage | Any) -> int:
    content = getattr(msg, "content", "")
    if isinstance(content, str):
        return len(content)
    if isinstance(content, list):
        return sum(len(str(block)) for block in content)
    return len(str(content or ""))


def _first_human_index(messages: list[BaseMessage]) -> int | None:
    for idx, msg in enumerate(messages):
        if _is_human(msg):
            return idx
    return None


def _last_human_index(messages: list[BaseMessage]) -> int | None:
    for idx in range(len(messages) - 1, -1, -1):
        if _is_human(messages[idx]):
            return idx
    return None


def _segment_end(messages: list[BaseMessage], start: int) -> int:
    if start >= len(messages):
        return start
    msg = messages[start]
    if _is_human(msg):
        return start + 1
    if isinstance(msg, AIMessage):
        end = start + 1
        if msg.tool_calls:
            while end < len(messages) and isinstance(messages[end], ToolMessage):
                end += 1
        return end
    if isinstance(msg, ToolMessage):
        end = start + 1
        while end < len(messages) and isinstance(messages[end], ToolMessage):
            end += 1
        return end
    return start + 1


def _drop_oldest_segment_after_anchor(windowed: list[BaseMessage]) -> list[BaseMessage]:
    if len(windowed) <= 1:
        return windowed
    cut = _segment_end(windowed, 1)
    if cut <= 1:
        cut = 2 if len(windowed) > 2 else 1
    return [windowed[0], *windowed[cut:]]


def trim_messages_for_llm(messages: list[BaseMessage] | None) -> list[BaseMessage]:
    if not messages:
        return []
    window_max = get_policy().llm.message_window_max
    msgs = list(messages)

    last_human = _last_human_index(msgs)
    if last_human is None:
        return msgs[-window_max:]

    first_human = _first_human_index(msgs)
    if first_human is None:
        first_human = last_human
    start = min(first_human, last_human)
    windowed = msgs[start:]

    while len(windowed) > window_max:
        shorter = _drop_oldest_segment_after_anchor(windowed)
        if shorter == windowed:
            break
        windowed = shorter

    while sum(_content_chars(m) for m in windowed) > _CONTENT_CHAR_BUDGET and len(windowed) > 2:
        shorter = _drop_oldest_segment_after_anchor(windowed)
        if shorter == windowed:
            break
        windowed = shorter

    return windowed


def trim_for_llm(state: dict[str, Any]) -> dict[str, Any]:
    messages = list(state.get("messages") or [])
    return {"llm_input_messages": trim_messages_for_llm(messages)}
