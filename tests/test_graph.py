"""Graph factory and pack wiring."""

from __future__ import annotations

from graph_agent.agent.packs import pack_hint, prompt_for_pack, tools_for_pack
from graph_agent.agent.state import InteractionMode
from graph_agent.policy import get_policy, load_policy, set_policy


def test_packs_loaded_from_yaml() -> None:
    policy = get_policy()
    assert "billing" in policy.packs
    assert "catalog" in policy.packs
    assert "analytics" in policy.packs
    assert "get_invoice" in policy.packs["billing"].tools
    assert "invoice" in policy.packs["billing"].keywords


def test_prompt_for_pack_uses_config() -> None:
    blob = prompt_for_pack(InteractionMode.ASK, "billing").lower()
    assert "billing" in blob or "invoice" in blob
    assert pack_hint("billing")


def test_tools_for_pack_intersects_registry() -> None:
    # Only ask_user is registered by default in the standalone core.
    tools = tools_for_pack(InteractionMode.ASK, "billing")
    names = {t.name for t in tools}
    assert "ask_user" in names


def test_policy_reload(monkeypatch, tmp_path) -> None:
    yaml_path = tmp_path / "agent.yaml"
    yaml_path.write_text(
        """
agent:
  name: TestBot
  default_pack: alpha
packs:
  alpha:
    tools: [ask_user]
    keywords: [hello]
    prompt: Alpha pack
    hint: hint-alpha
""",
        encoding="utf-8",
    )
    set_policy(None)
    policy = load_policy(yaml_path)
    set_policy(policy)
    assert get_policy().agent.name == "TestBot"
    assert get_policy().pack_names() == ["alpha"]
    # Restore default project policy for other tests.
    set_policy(None)
