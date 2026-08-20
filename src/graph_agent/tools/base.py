"""Dry-run helpers and JSON helpers for tools."""

from __future__ import annotations

import json
from contextvars import ContextVar
from typing import Any

from graph_agent.policy import get_policy

dry_run_context: ContextVar[bool] = ContextVar("agent_dry_run", default=False)


def is_dry_run(explicit: bool | None = None) -> bool:
    if explicit:
        return True
    return dry_run_context.get()


def set_dry_run(enabled: bool) -> None:
    dry_run_context.set(enabled)


def preview(action: str, **kwargs: Any) -> str:
    payload = {k: v for k, v in kwargs.items() if v is not None}
    return json.dumps(
        {"dry_run": True, "would_execute": action, "args": payload},
        ensure_ascii=False,
        separators=(",", ":"),
        default=str,
    )


def tool_result_max_chars() -> int:
    return get_policy().llm.tool_result_max_chars


def dumps(data: Any, *, max_chars: int | None = None) -> str:
    text = json.dumps(data, ensure_ascii=False, separators=(",", ":"), default=str)
    if max_chars is not None and max_chars > 0 and len(text) > max_chars:
        return json.dumps(
            {"truncated": True, "preview": text[: max(0, max_chars - 80)]},
            ensure_ascii=False,
            separators=(",", ":"),
        )
    return text


def error(message: str) -> str:
    return json.dumps({"error": message}, ensure_ascii=False, separators=(",", ":"))
