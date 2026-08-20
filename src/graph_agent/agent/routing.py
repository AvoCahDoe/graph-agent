"""Keyword router: last user message → domain pack (keywords from config)."""

from __future__ import annotations

import re
from typing import Any

from langchain_core.messages import HumanMessage

from graph_agent.policy import get_policy


def _content_text(content: Any) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
        return " ".join(parts)
    return str(content or "")


def last_human_text(messages: list | None) -> str:
    for msg in reversed(messages or []):
        if isinstance(msg, HumanMessage):
            return _content_text(msg.content)
        if isinstance(msg, tuple) and len(msg) >= 2 and str(msg[0]) in {"user", "human"}:
            return str(msg[1])
        if getattr(msg, "type", None) == "human":
            return _content_text(getattr(msg, "content", ""))
    return ""


def _keyword_matches(text: str, keyword: str) -> bool:
    """Match multi-word phrases as substrings; single words on word boundaries."""
    kw = (keyword or "").strip().lower()
    if not kw:
        return False
    if " " in kw:
        return kw in text
    return bool(re.search(rf"\b{re.escape(kw)}\b", text))


def _score(text: str, keywords: list[str]) -> int:
    score = 0
    for kw in keywords:
        if not _keyword_matches(text, kw):
            continue
        parts = len(kw.split())
        # Prefer multi-word phrases over single-token hits.
        score += 1 + min(parts, 3) + (4 if parts > 1 else 0)
    return score


def pack_scores(text: str) -> dict[str, int]:
    """Keyword match scores per pack (higher = stronger signal)."""
    policy = get_policy()
    normalized = f" {(text or '').strip().lower()} "
    return {
        name: _score(normalized, pack.keywords)
        for name, pack in policy.packs.items()
    }


def pending_keyword_packs(text: str, visited: list[str] | set[str]) -> list[str]:
    blocked = {str(x) for x in visited}
    return [pack for pack, score in pack_scores(text).items() if pack not in blocked and score > 0]


def keyword_route_confident(text: str, exclude: set[str] | frozenset[str] | None = None) -> str | None:
    """Return a pack only when exactly one non-excluded pack matches keywords."""
    blocked = {str(x) for x in (exclude or set())}
    positive = {pack: score for pack, score in pack_scores(text).items() if pack not in blocked and score > 0}
    if len(positive) == 1:
        return next(iter(positive))
    return None


def route_pack_from_text(text: str) -> str:
    """Pick a pack from free text. Falls back to configured default_pack."""
    picked = route_pack_excluding(text, exclude=set())
    if picked is not None:
        return picked
    policy = get_policy()
    if policy.agent.default_pack in policy.packs:
        return policy.agent.default_pack
    names = policy.pack_names()
    return names[0] if names else "billing"


def route_pack_excluding(text: str, exclude: set[str] | frozenset[str] | None = None) -> str | None:
    """Best keyword pack not in exclude. None if every remaining pack scores 0."""
    blocked = {str(x) for x in (exclude or set())}
    scores = {pack: score for pack, score in pack_scores(text).items() if pack not in blocked}
    if not scores:
        return None
    best = max(scores.values())
    if best == 0:
        return None
    winners = [p for p, s in scores.items() if s == best]
    # Prefer analytics on ties when present (chart/KPI questions).
    if "analytics" in winners:
        return "analytics"
    return sorted(winners)[0]


def route_pack(state: dict[str, Any] | None) -> str:
    messages = (state or {}).get("messages") if isinstance(state, dict) else None
    return route_pack_from_text(last_human_text(list(messages or [])))
