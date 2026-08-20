"""Build per-pack ReAct specialists with trimmed tool sets."""

from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from langgraph.prebuilt import create_react_agent

from graph_agent.agent.message_trim import trim_for_llm
from graph_agent.agent.packs import pack_hint, prompt_for_pack, tools_for_pack
from graph_agent.agent.state import InteractionMode
from graph_agent.policy import get_policy

logger = logging.getLogger(__name__)


def _specialist_max_tokens(mode: InteractionMode) -> int:
    llm_policy = get_policy().llm
    if mode == InteractionMode.AGENT:
        return llm_policy.specialist_max_tokens_agent
    return llm_policy.specialist_max_tokens_ask


def _bind_specialist_llm(llm: BaseChatModel, mode: InteractionMode) -> BaseChatModel:
    cap = _specialist_max_tokens(mode)
    bind = getattr(llm, "bind", None)
    if callable(bind):
        return bind(max_tokens=cap)
    return llm


def build_specialist_react(llm: BaseChatModel, mode: InteractionMode, pack: str):
    """Compiled ReAct subgraph for one pack (no checkpointer; parent owns it)."""
    tools = tools_for_pack(mode, pack)
    prompt = f"{prompt_for_pack(mode, pack)}\n\n{pack_hint(pack)}".strip()
    model = _bind_specialist_llm(llm, mode)
    logger.info(
        "Specialist %s/%s with %s tools: %s",
        mode.value,
        pack,
        len(tools),
        sorted(t.name for t in tools),
    )
    return create_react_agent(
        model=model,
        tools=tools,
        prompt=prompt,
        name=pack,
        pre_model_hook=trim_for_llm,
    )
