"""LangSmith tracing config and metadata filtering."""

from __future__ import annotations

import os

from graph_agent.policy import TracingPolicy, get_policy, load_policy, policy_from_dict, set_policy
from graph_agent.tracing import (
    TraceContext,
    build_run_config,
    configure_tracing,
    filter_metadata,
    reset_tracing_state,
    trace_context_from_mapping,
)


def test_tracing_policy_from_yaml_defaults() -> None:
    policy = get_policy()
    assert isinstance(policy.tracing, TracingPolicy)
    assert policy.tracing.enabled is False
    assert "conversation_id" in policy.tracing.metadata_keys
    assert policy.tracing.project


def test_env_overrides_tracing_enabled(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.setenv("LANGSMITH_PROJECT", "from-env")
    policy = policy_from_dict(
        {
            "tracing": {
                "enabled": False,
                "project": "from-yaml",
                "tags": ["t1"],
                "metadata_keys": ["tenant_id"],
            }
        }
    )
    assert policy.tracing.enabled is True
    assert policy.tracing.project == "from-env"
    assert policy.tracing.metadata_keys == ["tenant_id"]


def test_filter_metadata_respects_keys() -> None:
    ctx = TraceContext(
        conversation_id="c1",
        tenant_id="t1",
        user_id="u1",
        agent_id="a1",
        session_id="s1",
        extra={"ignored": "x"},
    )
    filtered = filter_metadata(ctx, ["conversation_id", "tenant_id"])
    assert filtered == {"conversation_id": "c1", "tenant_id": "t1"}
    assert "ignored" not in filtered


def test_build_run_config_includes_thread_tags_metadata() -> None:
    ctx = TraceContext(conversation_id="conv-9", tenant_id="mandant-a", session_id="thr-1")
    cfg = build_run_config("thr-1", "ask", context=ctx)
    assert cfg["configurable"]["thread_id"] == "thr-1"
    assert cfg["recursion_limit"] >= 1
    assert "ask" in cfg["run_name"] or "thr-1" in cfg["run_name"]
    assert any("mode:ask" == t or t == "ask" for t in cfg["tags"]) or "graph-agent" in cfg["tags"]
    assert cfg["metadata"]["thread_id"] == "thr-1"
    assert cfg["metadata"]["mode"] == "ask"
    assert cfg["metadata"]["conversation_id"] == "conv-9"
    assert cfg["metadata"]["tenant_id"] == "mandant-a"


def test_configure_tracing_off_without_flag(monkeypatch) -> None:
    monkeypatch.delenv("LANGSMITH_TRACING", raising=False)
    monkeypatch.delenv("LANGCHAIN_TRACING_V2", raising=False)
    reset_tracing_state()
    set_policy(policy_from_dict({"tracing": {"enabled": False}}))
    assert configure_tracing() is False
    reset_tracing_state()
    set_policy(None)


def test_configure_tracing_off_without_api_key(monkeypatch) -> None:
    monkeypatch.setenv("LANGSMITH_TRACING", "true")
    monkeypatch.delenv("LANGSMITH_API_KEY", raising=False)
    monkeypatch.delenv("LANGCHAIN_API_KEY", raising=False)
    reset_tracing_state()
    set_policy(policy_from_dict({"tracing": {"enabled": True}}))
    assert configure_tracing() is False
    reset_tracing_state()
    set_policy(None)


def test_trace_context_from_mapping_body_wins_over_headers() -> None:
    ctx = trace_context_from_mapping(
        {"tenant_id": "from-body", "conversation_id": "c-body", "session_id": "s1"},
        {"X-Tenant-Id": "from-header", "X-Conversation-Id": "c-header"},
    )
    assert ctx.tenant_id == "from-body"
    assert ctx.conversation_id == "c-body"
    assert ctx.session_id == "s1"


def test_trace_context_from_headers_fallback() -> None:
    ctx = trace_context_from_mapping(
        {},
        {"X-Tenant-Id": "mandant-x", "X-User-Id": "user-1", "X-Conversation-Id": "conv-h"},
    )
    assert ctx.tenant_id == "mandant-x"
    assert ctx.user_id == "user-1"
    assert ctx.conversation_id == "conv-h"


def test_policy_reload_tracing_section(tmp_path) -> None:
    yaml_path = tmp_path / "agent.yaml"
    yaml_path.write_text(
        """
agent:
  name: TraceBot
tracing:
  enabled: false
  project: custom-project
  tags: [alpha]
  metadata_keys: [tenant_id, conversation_id]
packs:
  alpha:
    tools: [ask_user]
    keywords: [hello]
""",
        encoding="utf-8",
    )
    # Clear env so yaml project is used
    os.environ.pop("LANGSMITH_PROJECT", None)
    os.environ.pop("LANGCHAIN_PROJECT", None)
    set_policy(None)
    policy = load_policy(yaml_path)
    set_policy(policy)
    assert get_policy().tracing.project == "custom-project"
    assert get_policy().tracing.tags == ["alpha"]
    set_policy(None)
