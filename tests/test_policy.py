"""Policy / config loading."""

from __future__ import annotations

from graph_agent.policy import get_policy


def test_modes_and_confirm() -> None:
    policy = get_policy()
    assert policy.mode("ask").max_risk == "read_only"
    assert "ask_user" in policy.mode("ask").extra_tools
    assert policy.confirm["mutating"] is True
    assert policy.should_confirm("upsert_report_definition", "mutating") is True
    assert policy.allows_risk("ask", "read_only") is True
    assert policy.allows_risk("ask", "destructive") is False
