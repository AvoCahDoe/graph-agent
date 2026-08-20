"""Product policy loaded from config/agent.yaml (modes, packs, LLM, tools)."""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from graph_agent.config import get_settings

_RISK_RANK = {
    "read_only": 0,
    "hitl": 1,
    "mutating": 2,
    "destructive": 3,
}

_DEFAULT_CONFIRM = {"mutating": True, "destructive": True}


@dataclass
class ModePolicy:
    max_risk: str = "read_only"
    extra_tools: list[str] = field(default_factory=list)


@dataclass
class LlmPolicy:
    provider: str = "auto"
    specialist_max_tokens_ask: int = 768
    specialist_max_tokens_agent: int = 768
    planner_max_tokens: int = 64
    message_window_max: int = 14
    tool_result_max_chars: int = 3500


@dataclass
class ToolPolicy:
    risk: str | None = None
    confirm: bool | None = None


@dataclass
class PackPolicy:
    tools: list[str] = field(default_factory=list)
    keywords: list[str] = field(default_factory=list)
    hint: str = ""
    prompt: str = ""


@dataclass
class AgentMeta:
    name: str = "Standalone Agent"
    default_mode: str = "ask"
    recursion_limit: int = 32
    max_pack_hops: int = 3
    default_pack: str = "billing"


@dataclass
class ServerPolicy:
    host: str = "0.0.0.0"
    port: int = 6969


@dataclass
class PlannerPolicy:
    system_prompt: str = (
        'You route questions to one specialist pack, or finish.\n'
        'Reply with JSON only: {"next":"<pack_name>|finish","reason":"short"}'
    )


_DEFAULT_METADATA_KEYS = (
    "conversation_id",
    "tenant_id",
    "user_id",
    "agent_id",
    "session_id",
)


@dataclass
class TracingPolicy:
    """LangSmith observability knobs. Secrets stay in env."""

    enabled: bool = False
    project: str = "graph-agent"
    run_name_template: str = "{agent}/{mode}/{thread_id}"
    tags: list[str] = field(default_factory=lambda: ["graph-agent"])
    metadata_keys: list[str] = field(default_factory=lambda: list(_DEFAULT_METADATA_KEYS))


@dataclass
class Policy:
    agent: AgentMeta = field(default_factory=AgentMeta)
    modes: dict[str, ModePolicy] = field(default_factory=dict)
    confirm: dict[str, bool] = field(default_factory=lambda: dict(_DEFAULT_CONFIRM))
    tools: dict[str, ToolPolicy] = field(default_factory=dict)
    packs: dict[str, PackPolicy] = field(default_factory=dict)
    llm: LlmPolicy = field(default_factory=LlmPolicy)
    planner: PlannerPolicy = field(default_factory=PlannerPolicy)
    server: ServerPolicy = field(default_factory=ServerPolicy)
    tracing: TracingPolicy = field(default_factory=TracingPolicy)

    def mode(self, name: str) -> ModePolicy:
        return self.modes.get(name) or ModePolicy(
            max_risk="read_only" if name == "ask" else "destructive",
            extra_tools=["ask_user"] if name == "ask" else [],
        )

    def tool(self, name: str) -> ToolPolicy:
        return self.tools.get(name) or ToolPolicy()

    def pack(self, name: str) -> PackPolicy | None:
        return self.packs.get(name)

    def pack_names(self) -> list[str]:
        return list(self.packs.keys())

    def allows_risk(self, mode: str, risk: str) -> bool:
        max_rank = _RISK_RANK.get(self.mode(mode).max_risk, 0)
        return _RISK_RANK.get(risk, 99) <= max_rank

    def should_confirm(self, action: str, risk: str | None) -> bool:
        entry = self.tool(action)
        if entry.confirm is not None:
            return entry.confirm
        if not risk:
            return False
        if risk in {"read_only", "hitl"}:
            return False
        return bool(self.confirm.get(risk, False))


_policy: Policy | None = None


def _as_int(value: Any, default: int, *, minimum: int = 1) -> int:
    if value is None:
        return default
    try:
        parsed = int(value)
    except (TypeError, ValueError):
        return default
    return max(minimum, parsed)


def _as_bool(value: Any, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    return str(value).strip().lower() in {"1", "true", "yes", "on"}


def _env_int(name: str) -> int | None:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return None
    try:
        return int(str(raw).strip())
    except ValueError:
        return None


def _parse_tool(raw: Any) -> ToolPolicy:
    if raw is None:
        return ToolPolicy()
    if isinstance(raw, str):
        return ToolPolicy(risk=raw.strip().lower())
    if not isinstance(raw, dict):
        return ToolPolicy()
    risk = raw.get("risk")
    confirm = raw.get("confirm")
    return ToolPolicy(
        risk=str(risk).strip().lower() if risk else None,
        confirm=None if confirm is None else _as_bool(confirm, False),
    )


def _parse_pack(raw: Any) -> PackPolicy:
    raw = raw if isinstance(raw, dict) else {}
    tools = raw.get("tools") or []
    keywords = raw.get("keywords") or []
    if isinstance(tools, str):
        tools = [tools]
    if isinstance(keywords, str):
        keywords = [keywords]
    return PackPolicy(
        tools=[str(t) for t in tools],
        keywords=[str(k).strip().lower() for k in keywords if str(k).strip()],
        hint=str(raw.get("hint") or "").strip(),
        prompt=str(raw.get("prompt") or "").strip(),
    )


def _parse_llm(raw: Any) -> LlmPolicy:
    raw = raw if isinstance(raw, dict) else {}
    ask = _env_int("SPECIALIST_MAX_TOKENS_ASK") or _as_int(
        raw.get("specialist_max_tokens_ask"), 768
    )
    agent = _env_int("SPECIALIST_MAX_TOKENS_AGENT") or _as_int(
        raw.get("specialist_max_tokens_agent"),
        _as_int(raw.get("specialist_max_tokens"), 768),
    )
    planner = _env_int("PLANNER_MAX_TOKENS") or _as_int(raw.get("planner_max_tokens"), 64)
    window = _env_int("MESSAGE_WINDOW_MAX") or _as_int(raw.get("message_window_max"), 14)
    tool_chars = _env_int("TOOL_RESULT_MAX_CHARS") or _as_int(
        raw.get("tool_result_max_chars"), 3500, minimum=256
    )
    provider = str(raw.get("provider") or os.getenv("LLM_PROVIDER") or "auto").strip()
    return LlmPolicy(
        provider=provider,
        specialist_max_tokens_ask=ask,
        specialist_max_tokens_agent=agent,
        planner_max_tokens=planner,
        message_window_max=window,
        tool_result_max_chars=tool_chars,
    )


def _env_bool_override(name: str) -> bool | None:
    raw = os.getenv(name)
    if raw is None or not str(raw).strip():
        return None
    return str(raw).strip().lower() in {"1", "true", "yes", "on"}


def _parse_tracing(raw: Any) -> TracingPolicy:
    raw = raw if isinstance(raw, dict) else {}
    tags = raw.get("tags") or ["graph-agent"]
    if isinstance(tags, str):
        tags = [tags]
    keys = raw.get("metadata_keys") or list(_DEFAULT_METADATA_KEYS)
    if isinstance(keys, str):
        keys = [keys]

    yaml_enabled = _as_bool(raw.get("enabled"), False)
    env_enabled = _env_bool_override("LANGSMITH_TRACING")
    enabled = yaml_enabled if env_enabled is None else env_enabled

    project = (
        os.getenv("LANGSMITH_PROJECT")
        or os.getenv("LANGCHAIN_PROJECT")
        or str(raw.get("project") or "graph-agent")
    ).strip()

    template = str(raw.get("run_name_template") or "{agent}/{mode}/{thread_id}").strip()
    return TracingPolicy(
        enabled=enabled,
        project=project or "graph-agent",
        run_name_template=template or "{agent}/{mode}/{thread_id}",
        tags=[str(t).strip() for t in tags if str(t).strip()],
        metadata_keys=[str(k).strip() for k in keys if str(k).strip()],
    )


def policy_from_dict(data: dict[str, Any] | None) -> Policy:
    data = data or {}

    agent_raw = data.get("agent") or {}
    if not isinstance(agent_raw, dict):
        agent_raw = {}
    agent = AgentMeta(
        name=str(agent_raw.get("name") or "Standalone Agent"),
        default_mode=str(agent_raw.get("default_mode") or "ask"),
        recursion_limit=_as_int(agent_raw.get("recursion_limit"), 32),
        max_pack_hops=_as_int(agent_raw.get("max_pack_hops"), 3),
        default_pack=str(agent_raw.get("default_pack") or "billing"),
    )

    modes: dict[str, ModePolicy] = {}
    for name, raw in (data.get("modes") or {}).items():
        raw = raw or {}
        extra = raw.get("extra_tools") or []
        if isinstance(extra, str):
            extra = [extra]
        modes[str(name)] = ModePolicy(
            max_risk=str(raw.get("max_risk") or ("read_only" if name == "ask" else "destructive")),
            extra_tools=[str(t) for t in extra],
        )

    confirm_raw = data.get("confirm") or {}
    confirm = {
        "mutating": _as_bool(confirm_raw.get("mutating"), True),
        "destructive": _as_bool(confirm_raw.get("destructive"), True),
    }

    tools = {str(k): _parse_tool(v) for k, v in (data.get("tools") or {}).items()}
    packs = {str(k): _parse_pack(v) for k, v in (data.get("packs") or {}).items()}

    planner_raw = data.get("planner") or {}
    if not isinstance(planner_raw, dict):
        planner_raw = {}
    planner = PlannerPolicy(
        system_prompt=str(planner_raw.get("system_prompt") or PlannerPolicy().system_prompt).strip()
    )

    server_raw = data.get("server") or {}
    if not isinstance(server_raw, dict):
        server_raw = {}
    server = ServerPolicy(
        host=str(server_raw.get("host") or "0.0.0.0"),
        port=_as_int(server_raw.get("port"), 6969, minimum=1),
    )

    return Policy(
        agent=agent,
        modes=modes,
        confirm=confirm,
        tools=tools,
        packs=packs,
        llm=_parse_llm(data.get("llm")),
        planner=planner,
        server=server,
        tracing=_parse_tracing(data.get("tracing")),
    )


def load_policy(path: Path | None = None) -> Policy:
    if path is None:
        path = get_settings().policy_path
    if not path.is_file():
        return policy_from_dict(None)
    with path.open(encoding="utf-8") as handle:
        data = yaml.safe_load(handle) or {}
    if not isinstance(data, dict):
        return policy_from_dict(None)
    return policy_from_dict(data)


def get_policy() -> Policy:
    global _policy
    if _policy is None:
        _policy = load_policy()
    return _policy


def set_policy(policy: Policy | None) -> None:
    """Tests and reload: pass None to re-read agent.yaml."""
    global _policy
    _policy = policy
