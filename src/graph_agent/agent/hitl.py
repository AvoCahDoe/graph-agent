"""Human-in-the-loop helpers via langgraph.types.interrupt."""

from __future__ import annotations

from typing import Any

from langgraph.types import interrupt

from graph_agent.policy import get_policy
from graph_agent.tools.base import is_dry_run
from graph_agent.tools.registry import tool_registry

CANCELLED_MESSAGE = "Action cancelled by the user."

_APPROVE_VALUES = frozenset(
    {"approve", "approved", "accept", "accepted", "yes", "y", "true", "1"}
)


def is_approved(decision: Any) -> bool:
    """Interpret a resume value as approve vs reject."""
    if decision is True:
        return True
    if decision is False or decision is None:
        return False
    if isinstance(decision, (int, float)) and not isinstance(decision, bool):
        return decision == 1
    if isinstance(decision, str):
        return decision.strip().lower() in _APPROVE_VALUES
    if isinstance(decision, dict):
        kind = str(
            decision.get("type")
            or decision.get("decision")
            or decision.get("action")
            or ""
        ).strip().lower()
        if kind in _APPROVE_VALUES:
            return True
        if kind in {"reject", "rejected", "deny", "denied", "no", "cancel", "cancelled"}:
            return False
        if "approved" in decision:
            return bool(decision["approved"])
    return False


def humanize_action(action: str) -> str:
    cleaned = (action or "").replace("_", " ").strip()
    if not cleaned:
        return "Action"
    return cleaned[0].upper() + cleaned[1:]


def interrupt_payload(item: Any) -> dict[str, Any]:
    value = getattr(item, "value", item)
    if isinstance(value, dict):
        return value
    return {"type": "ask_user", "question": str(value)}


def graph_state(agent: Any, config: dict[str, Any]) -> Any:
    try:
        return agent.get_state(config, subgraphs=True)
    except TypeError:
        return agent.get_state(config)


def pending_interrupts(state: Any) -> list[Any]:
    found: list[Any] = []
    seen: set[int] = set()

    def add(items: Any) -> None:
        if items is None:
            return
        if not isinstance(items, (list, tuple)):
            items = (items,)
        for item in items:
            ident = id(item)
            if ident in seen:
                continue
            seen.add(ident)
            found.append(item)

    if state is None:
        return found
    add(getattr(state, "interrupts", None))
    for task in getattr(state, "tasks", None) or ():
        add(getattr(task, "interrupts", None))
        nested = getattr(task, "state", None)
        if nested is None or isinstance(nested, dict):
            continue
        for item in pending_interrupts(nested):
            add((item,))
    return found


def require_confirmation(*, action: str, args: dict[str, Any], message: str) -> bool:
    if is_dry_run():
        return True
    risk = tool_registry.get_risk(action).value
    if not get_policy().should_confirm(action, risk):
        return True
    decision = interrupt(
        {
            "type": "confirm",
            "action": action,
            "args": args,
            "message": message,
            "allowed": ["approve", "reject"],
        }
    )
    return is_approved(decision)


def ask_human(
    question: str,
    options: list[str] | None = None,
) -> str:
    payload: dict[str, Any] = {"type": "ask_user", "question": question}
    if options:
        payload["options"] = list(options)
    answer = interrupt(payload)
    if answer is None:
        return ""
    if isinstance(answer, dict):
        for key in ("answer", "response", "value", "text"):
            if key in answer and answer[key] is not None:
                return str(answer[key])
        return str(answer)
    return str(answer)
