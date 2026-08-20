"""High-level agent runner: invoke / stream / resume."""

from __future__ import annotations

import logging
import re
import uuid
from dataclasses import dataclass, field
from typing import Any, Iterator, Literal

from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage
from langgraph.checkpoint.memory import MemorySaver
from langgraph.types import Command

from graph_agent.agent.graph import agent_factory
from graph_agent.agent.hitl import graph_state, humanize_action, interrupt_payload, pending_interrupts
from graph_agent.agent.state import InteractionMode
from graph_agent.config import Settings, get_settings
from graph_agent.llm.factory import LLMFactory
from graph_agent.policy import get_policy
from graph_agent.tracing import TraceContext, build_run_config, configure_tracing, is_tracing_active

logger = logging.getLogger(__name__)

EventType = Literal["token", "tool", "interrupt", "done", "error", "system"]


@dataclass
class StreamEvent:
    type: EventType
    data: dict[str, Any] = field(default_factory=dict)

    @property
    def text(self) -> str:
        return str(self.data.get("text") or self.data.get("reply") or "")


def _chunk_text(content: object) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for block in content:
            if isinstance(block, dict) and block.get("type") == "text":
                parts.append(str(block.get("text") or ""))
            elif isinstance(block, str):
                parts.append(block)
        return "".join(parts)
    return ""


def _unpack_stream_item(item: object) -> tuple[object, str, object]:
    if not isinstance(item, tuple):
        return (), "updates", item
    if len(item) == 3:
        return item[0], str(item[1]), item[2]
    if len(item) == 2:
        left, right = item
        if isinstance(left, str) and left in {"messages", "updates", "values", "custom"}:
            return (), left, right
        return left, "updates", right
    return (), "updates", item


def _token_from_messages_chunk(data: object) -> str:
    pair = data if isinstance(data, (tuple, list)) and data else None
    msg = pair[0] if pair else data
    meta = pair[1] if pair and len(pair) > 1 and isinstance(pair[1], dict) else {}
    if getattr(msg, "type", None) in {"tool", "human"}:
        return ""
    if getattr(msg, "tool_call_chunks", None) or getattr(msg, "tool_calls", None):
        return ""
    node = str(meta.get("langgraph_node") or "")
    if "tools" in node or node == "planner" or node.endswith(":planner") or node.endswith("/planner"):
        return ""
    return _chunk_text(getattr(msg, "content", ""))


_PLANNER_JSON = re.compile(
    r'\{\s*"next"\s*:\s*"[^"]+"\s*,\s*"reason"\s*:\s*".*?"\s*\}',
    re.DOTALL,
)


def _strip_planner_json(text: str) -> str:
    out = text or ""
    out = _PLANNER_JSON.sub("", out)
    return out.lstrip()


def _ai_text(result: dict) -> str:
    messages = (result or {}).get("messages") or []
    for msg in reversed(messages):
        if isinstance(msg, AIMessage) and not getattr(msg, "tool_calls", None):
            text = _strip_planner_json(_chunk_text(msg.content))
            if text.strip():
                return text
    return ""


def _flatten_update(event: object) -> list[dict]:
    if isinstance(event, tuple) and len(event) >= 2:
        event = event[-1]
    if not isinstance(event, dict):
        return []
    out = [event]
    for key, value in event.items():
        if str(key).startswith("__"):
            continue
        if isinstance(value, dict):
            out.extend(_flatten_update(value))
    return out


def _interrupts_in_update(event: object) -> list:
    found: list = []
    for item in _flatten_update(event):
        raw = item.get("__interrupt__")
        if not raw:
            continue
        found.extend(list(raw) if isinstance(raw, (list, tuple)) else [raw])
    return found


def _public_interrupt(item: Any) -> dict[str, Any]:
    payload = interrupt_payload(item)
    action = payload.get("action")
    if action:
        payload["label"] = humanize_action(str(action))
    return payload


class AgentRunner:
    """Integratable facade over the LangGraph supervisor."""

    def __init__(
        self,
        *,
        llm: BaseChatModel | None = None,
        settings: Settings | None = None,
        mode: InteractionMode | str | None = None,
        checkpointer: MemorySaver | None | Literal["auto"] = "auto",
    ) -> None:
        self.settings = settings or get_settings()
        configure_tracing()
        self.llm = llm or LLMFactory.create(self.settings)
        self.factory = agent_factory(self.llm, checkpointer=checkpointer)
        mode_raw = mode or self.settings.default_mode or get_policy().agent.default_mode
        if isinstance(mode_raw, InteractionMode):
            self.mode = mode_raw
        else:
            self.mode = InteractionMode.AGENT if str(mode_raw).lower() == "agent" else InteractionMode.ASK
        self.agent = self.factory(self.mode)
        self._msg_counts: dict[str, int] = {}
        self.tracing_enabled = is_tracing_active()

    @property
    def name(self) -> str:
        return get_policy().agent.name

    def set_mode(self, mode: InteractionMode | str) -> None:
        if isinstance(mode, str):
            mode = InteractionMode.AGENT if mode.lower() == "agent" else InteractionMode.ASK
        self.mode = mode
        self.agent = self.factory(self.mode)

    def _config(self, thread_id: str, context: TraceContext | None = None) -> dict[str, Any]:
        return build_run_config(thread_id, self.mode.value, context=context)

    def _ensure_thread(self, thread_id: str | None) -> str:
        return thread_id or str(uuid.uuid4())

    def clear(self, thread_id: str | None = None) -> str:
        """Start a fresh thread (new MemorySaver thread id)."""
        new_id = str(uuid.uuid4())
        if thread_id:
            self._msg_counts.pop(thread_id, None)
        return new_id

    def invoke(
        self,
        text: str,
        *,
        thread_id: str | None = None,
        context: TraceContext | None = None,
    ) -> dict[str, Any]:
        tid = self._ensure_thread(thread_id)
        result = self.agent.invoke(
            {"messages": [("user", text)]},
            self._config(tid, context),
        )
        return self._after(result, tid, context)

    def resume(
        self,
        decision: Any,
        *,
        thread_id: str,
        context: TraceContext | None = None,
    ) -> dict[str, Any]:
        result = self.agent.invoke(
            Command(resume=decision),
            self._config(thread_id, context),
        )
        return self._after(result, thread_id, context)

    def stream(
        self,
        text: str | None = None,
        *,
        thread_id: str | None = None,
        resume: Any = None,
        context: TraceContext | None = None,
    ) -> Iterator[StreamEvent]:
        tid = self._ensure_thread(thread_id)
        cfg = self._config(tid, context)
        if resume is not None:
            graph_input: object = Command(resume=resume)
        else:
            if not text:
                yield StreamEvent("done", {"reply": "", "thread_id": tid, "mode": self.mode.value})
                return
            graph_input = {"messages": [("user", text)]}

        collected = ""
        raw = ""
        try:
            hit_interrupt = False
            for item in self.agent.stream(
                graph_input,
                cfg,
                stream_mode=["messages", "updates"],
                subgraphs=True,
            ):
                _ns, mode, data = _unpack_stream_item(item)
                if mode == "messages":
                    token = _token_from_messages_chunk(data)
                    if not token:
                        continue
                    raw += token
                    visible = _strip_planner_json(raw)
                    delta = visible[len(collected) :] if visible.startswith(collected) else visible
                    collected = visible
                    if delta:
                        yield StreamEvent("token", {"text": delta})
                elif mode == "updates":
                    for event in _flatten_update(data):
                        if _interrupts_in_update(event) or event.get("__interrupt__"):
                            hit_interrupt = True
                            break
                    if hit_interrupt:
                        break
        except Exception as exc:
            state = None
            try:
                state = graph_state(self.agent, cfg)
            except Exception:
                state = None
            interrupts = pending_interrupts(state) if state is not None else []
            if interrupts:
                values = state.values if state and isinstance(state.values, dict) else {}
                collected = _strip_planner_json(collected) or _strip_planner_json(_ai_text(values))
                payload = _public_interrupt(interrupts[0])
                yield StreamEvent("interrupt", payload)
                yield StreamEvent(
                    "done",
                    {"reply": collected, "thread_id": tid, "mode": self.mode.value},
                )
                return
            logger.exception("stream failed")
            yield StreamEvent("error", {"error": str(exc)})
            yield StreamEvent("done", {"reply": "", "thread_id": tid, "mode": self.mode.value})
            return

        state = graph_state(self.agent, cfg)
        values = state.values if isinstance(state.values, dict) else {}
        interrupts = pending_interrupts(state)
        collected = _strip_planner_json(collected) or _strip_planner_json(_ai_text(values))
        if interrupts:
            payload = _public_interrupt(interrupts[0])
            yield StreamEvent("interrupt", payload)
            yield StreamEvent(
                "done",
                {"reply": collected, "thread_id": tid, "mode": self.mode.value},
            )
            return
        yield StreamEvent(
            "done",
            {
                "reply": collected or "(no reply)",
                "thread_id": tid,
                "mode": self.mode.value,
            },
        )

    def _after(
        self,
        result: dict,
        thread_id: str,
        context: TraceContext | None = None,
    ) -> dict[str, Any]:
        interrupts = pending_interrupts(graph_state(self.agent, self._config(thread_id, context)))
        reply = _strip_planner_json(_ai_text(result))
        out: dict[str, Any] = {
            "reply": reply or "(no reply)",
            "thread_id": thread_id,
            "mode": self.mode.value,
            "name": self.name,
        }
        if interrupts:
            out["interrupt"] = _public_interrupt(interrupts[0])
            out["reply"] = reply
        return out
