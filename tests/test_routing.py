"""Domain pack routing tests (keywords from agent.yaml)."""

from __future__ import annotations

from graph_agent.agent.routing import route_pack, route_pack_from_text


def test_analytics_purchases_per_supplier() -> None:
    assert route_pack_from_text("chart of purchases per supplier") == "analytics"


def test_analytics_keywords() -> None:
    assert route_pack_from_text("top clients by CA") == "analytics"
    assert route_pack_from_text("monthly revenue please") == "analytics"


def test_default_is_billing() -> None:
    assert route_pack_from_text("") == "billing"
    assert route_pack_from_text("hello") == "billing"


def test_catalog_keywords() -> None:
    assert route_pack_from_text("check stock for SKU") == "catalog"
    assert route_pack_from_text("product inventory warehouse") == "catalog"


def test_billing_keywords() -> None:
    assert route_pack_from_text("list unpaid invoices") == "billing"


def test_route_pack_from_state() -> None:
    from langchain_core.messages import HumanMessage

    assert route_pack({"messages": [HumanMessage(content="show me a revenue chart")]}) == "analytics"
