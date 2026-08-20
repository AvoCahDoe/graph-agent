"""Optional LangSmith tracing: configure env + build LangGraph run configs.

LangSmith is observability only. Conversation / tenant identity is passed as
metadata so integrators can find a specific customer run. Product history stays
in the integrator's store (OpenSearch, logs, etc.).
"""

from __future__ import annotations

import logging
import os
from dataclasses import dataclass, field
from typing import Any

from graph_agent.config import sync_langsmith_env
from graph_agent.policy import get_policy

logger = logging.getLogger(__name__)

_configured: bool | None = None


@dataclass
class TraceContext:
    """Caller identity forwarded into LangSmith metadata (filtered by policy)."""

    conversation_id: str | None = None
    tenant_id: str | None = None
    user_id: str | None = None
    agent_id: str | None = None
    session_id: str | None = None
    extra: dict[str, Any] = field(default_factory=dict)

    def as_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "conversation_id": self.conversation_id,
            "tenant_id": self.tenant_id,
            "user_id": self.user_id,
            "agent_id": self.agent_id,
            "session_id": self.session_id,
        }
        data.update(self.extra or {})
        return {k: v for k, v in data.items() if v is not None and str(v).strip() != ""}


def is_tracing_active() -> bool:
    """True after a successful configure_tracing() that enabled export."""
    return _configured is True


def reset_tracing_state() -> None:
    """Tests only: clear configure cache."""
    global _configured
    _configured = None


def _api_key() -> str:
    return (
        os.getenv("LANGCHAIN_API_KEY")
        or os.getenv("LANGSMITH_API_KEY")
        or ""
    ).strip()


def _resolve_workspace() -> str | None:
    existing = os.getenv("LANGSMITH_WORKSPACE_ID") or os.getenv("LANGCHAIN_WORKSPACE_ID")
    if existing:
        os.environ["LANGSMITH_WORKSPACE_ID"] = existing
        os.environ.setdefault("LANGCHAIN_WORKSPACE_ID", existing)
        return existing
    try:
        from langsmith import Client
    except ImportError:
        return None
    try:
        probe = Client()
        if probe.workspace_id:
            os.environ["LANGSMITH_WORKSPACE_ID"] = probe.workspace_id
            os.environ.setdefault("LANGCHAIN_WORKSPACE_ID", probe.workspace_id)
            return probe.workspace_id
        response = probe.request_with_retries("GET", "/workspaces")
        workspaces = response.json() or []
        if len(workspaces) == 1:
            wid = str(workspaces[0]["id"])
            os.environ["LANGSMITH_WORKSPACE_ID"] = wid
            os.environ.setdefault("LANGCHAIN_WORKSPACE_ID", wid)
            return wid
        if len(workspaces) > 1:
            choices = ", ".join(
                f"{item.get('display_name')} ({item.get('id')})" for item in workspaces
            )
            logger.warning(
                "Multiple LangSmith workspaces; set LANGSMITH_WORKSPACE_ID to one of: %s",
                choices,
            )
    except Exception as exc:
        logger.warning("LangSmith workspace lookup failed: %s", exc)
    return None


def configure_tracing() -> bool:
    """Enable LangSmith export when policy/env say so. Idempotent; fail-soft."""
    global _configured
    if _configured is not None:
        return _configured

    sync_langsmith_env()
    policy = get_policy().tracing
    if not policy.enabled:
        _configured = False
        logger.debug("LangSmith tracing disabled (set LANGSMITH_TRACING=true to enable)")
        return False

    if not _api_key():
        logger.warning("LangSmith tracing enabled but LANGSMITH_API_KEY is missing; tracing off")
        _configured = False
        return False

    try:
        import langsmith  # noqa: F401
    except ImportError:
        logger.warning(
            "LangSmith tracing enabled but langsmith is not installed; "
            "pip install 'graph-agent[tracing]'"
        )
        _configured = False
        return False

    os.environ["LANGCHAIN_TRACING_V2"] = "true"
    os.environ["LANGSMITH_TRACING"] = "true"
    project = policy.project or "graph-agent"
    os.environ["LANGCHAIN_PROJECT"] = project
    os.environ.setdefault("LANGSMITH_PROJECT", project)
    _resolve_workspace()

    _configured = True
    logger.info("LangSmith tracing active (project=%s)", project)
    return True


def filter_metadata(context: TraceContext | None, allowed_keys: list[str] | None = None) -> dict[str, Any]:
    """Keep only policy-allowlisted keys from TraceContext (+ extras)."""
    if context is None:
        return {}
    keys = list(allowed_keys if allowed_keys is not None else get_policy().tracing.metadata_keys)
    allowed = set(keys)
    raw = context.as_dict()
    return {k: v for k, v in raw.items() if k in allowed}


def build_run_config(
    thread_id: str,
    mode: str,
    *,
    context: TraceContext | None = None,
    recursion_limit: int | None = None,
) -> dict[str, Any]:
    """LangGraph runnable config with optional LangSmith tags/metadata."""
    policy = get_policy()
    tracing = policy.tracing
    agent_name = policy.agent.name
    limit = recursion_limit if recursion_limit is not None else policy.agent.recursion_limit

    try:
        run_name = tracing.run_name_template.format(
            agent=agent_name,
            mode=mode,
            thread_id=thread_id,
        )
    except (KeyError, ValueError):
        run_name = f"{agent_name}/{mode}/{thread_id}"

    tags = list(tracing.tags)
    mode_tag = f"mode:{mode}"
    if mode_tag not in tags and mode not in tags:
        tags.append(mode_tag)

    metadata: dict[str, Any] = {
        "thread_id": thread_id,
        "mode": mode,
        "agent_name": agent_name,
    }
    metadata.update(filter_metadata(context, tracing.metadata_keys))

    return {
        "configurable": {"thread_id": thread_id},
        "recursion_limit": limit,
        "run_name": run_name,
        "tags": tags,
        "metadata": metadata,
    }


def trace_context_from_mapping(
    payload: dict[str, Any] | None = None,
    headers: dict[str, str] | None = None,
) -> TraceContext:
    """Build TraceContext from HTTP JSON body and/or headers (body wins)."""
    body = payload or {}
    hdrs = {str(k).lower(): str(v) for k, v in (headers or {}).items()}

    def pick(*names: str) -> str | None:
        for name in names:
            if name in body and body[name] is not None and str(body[name]).strip():
                return str(body[name]).strip()
        for name in names:
            key = name.lower().replace("_", "-")
            header_keys = (
                name.lower(),
                f"x-{key}",
                name.replace("_", "-").lower(),
            )
            for hk in header_keys:
                if hk in hdrs and hdrs[hk].strip():
                    return hdrs[hk].strip()
        return None

    session = pick("session_id", "sessionId", "thread_id", "threadId")
    return TraceContext(
        conversation_id=pick("conversation_id", "conversationId"),
        tenant_id=pick("tenant_id", "tenantId", "mandant"),
        user_id=pick("user_id", "userId"),
        agent_id=pick("agent_id", "agentId"),
        session_id=session,
    )
