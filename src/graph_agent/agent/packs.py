"""Pack helpers: tools/hints/prompts from centralized policy."""

from __future__ import annotations

from typing import Any

from graph_agent.agent.state import InteractionMode
from graph_agent.policy import get_policy
from graph_agent.tools.registry import tool_registry


def tools_for_pack(mode: InteractionMode | str, pack: str) -> list[Any]:
    """Mode-filtered tools intersected with the pack allow-list from config."""
    import graph_agent.tools  # noqa: F401

    mode_key = mode.value if isinstance(mode, InteractionMode) else str(mode)
    pack_policy = get_policy().pack(pack)
    names = set(pack_policy.tools) if pack_policy else set()
    return tool_registry.get_tools_for_mode_and_names(mode_key, names)


def pack_hint(pack: str) -> str:
    pack_policy = get_policy().pack(pack)
    return pack_policy.hint if pack_policy else ""


def prompt_for_pack(mode: InteractionMode, pack: str) -> str:
    policy = get_policy()
    pack_policy = policy.pack(pack)
    name = policy.agent.name
    mode_label = "ASK MODE (read-only)" if mode == InteractionMode.ASK else "AGENT MODE"
    header = f"You are {name}, {mode_label}."
    body = pack_policy.prompt if pack_policy else ""
    return f"{header}\n\n{body}".strip()
