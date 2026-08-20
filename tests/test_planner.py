"""Supervisor planner hop logic."""

from __future__ import annotations

from langchain_core.messages import AIMessage, HumanMessage
from langgraph.graph import END

from graph_agent.agent.planner import decide_next_pack
from graph_agent.policy import get_policy


def test_new_turn_purchases_chart_goes_analytics() -> None:
    target, update = decide_next_pack(
        None,
        {"messages": [HumanMessage(content="chart of purchases per supplier")]},
    )
    assert target == "analytics"
    assert update["visited_packs"] == ["analytics"]


def test_finish_after_analytics_on_chart_question(monkeypatch) -> None:
    from graph_agent.agent import planner as planner_mod

    called: list[bool] = []

    def _fail_if_called(*args, **kwargs):
        called.append(True)
        raise AssertionError("planner LLM should not hop after analytics on a chart question")

    monkeypatch.setattr(planner_mod, "_invoke_planner_llm", _fail_if_called)
    target, update = decide_next_pack(
        object(),
        {
            "messages": [
                HumanMessage(content="chart of purchases per supplier"),
                AIMessage(content=""),
            ],
            "pack_hops": 1,
            "visited_packs": ["analytics"],
        },
    )
    assert target == END
    assert update["visited_packs"] == ["analytics"]
    assert not called


def test_planner_prompt_mentions_packs() -> None:
    text = get_policy().planner.system_prompt.lower()
    assert "finish" in text
    assert "billing" in text or "pack" in text
