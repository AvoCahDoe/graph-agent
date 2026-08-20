"""Slash / undo command helpers."""

from __future__ import annotations

import re
from typing import Any

_SLASHES = str.maketrans(
    {
        "\u2215": "/",
        "\uff0f": "/",
        "\u2044": "/",
    }
)

_UNDO_ALIASES = frozenset(
    {
        "/undo",
        "/u",
        "undo",
        "u",
        "undo last",
        "undo last write",
        "undo last operation",
        "undo that",
        "undo this",
        "please undo",
    }
)

_UNDO_MESSAGE = re.compile(
    r"^"
    r"(?:please\s+)?"
    r"/?"
    r"\s*"
    r"(?:undo|u)"
    r"(?:\s+(?:last(?:\s+(?:write|operation|change|action))?|that|this))?"
    r"\s*[.!]?"
    r"$",
    re.IGNORECASE,
)


def normalize_command_text(raw: str) -> str:
    return (raw or "").translate(_SLASHES).strip().lower()


def is_undo_message(raw: str) -> bool:
    text = normalize_command_text(raw)
    return text in _UNDO_ALIASES or bool(_UNDO_MESSAGE.match(text))


def format_undo_reply(result: dict[str, Any] | None) -> str:
    if not result:
        return "Nothing to undo."
    label = str(result.get("label") or "").strip() or "last write"
    return f"Undid {label}."


def conversation_note_for_undo(result: dict[str, Any] | None) -> str:
    return (
        f"{format_undo_reply(result)} "
        "Earlier tool results for those records are stale. "
        "You MUST call get/list/search again. "
        "If a record is not found, say it no longer exists. "
        "Never repeat previous field dumps."
    )
