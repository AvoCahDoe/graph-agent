"""Headless HTTP/SSE API — JSON only, no HTML/CSS.

Sidecar contract (your BFF-compatible):
  POST /api/chat          — message or resume (SSE)
  POST /api/chat/clear    — reset session
  GET  /health            — health check
  POST /chat              — alias of /api/chat (legacy Farid path)
"""

from __future__ import annotations

import json
import logging
import socket
import sys
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any

from graph_agent.config import get_settings
from graph_agent.policy import get_policy
from graph_agent.runner import AgentRunner
from graph_agent.tracing import (
    configure_tracing,
    is_tracing_active,
    trace_context_from_mapping,
)

logger = logging.getLogger(__name__)

RUNNER: AgentRunner | None = None
_CURRENT_THREAD: str | None = None


def _json_bytes(payload: dict | list) -> bytes:
    return json.dumps(payload, ensure_ascii=False, default=str).encode("utf-8")


def _request_headers(handler: BaseHTTPRequestHandler) -> dict[str, str]:
    return {str(k): str(v) for k, v in handler.headers.items()}


class Handler(BaseHTTPRequestHandler):
    def log_message(self, fmt: str, *args) -> None:
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _cors(self) -> None:
        self.send_header("Access-Control-Allow-Origin", "*")
        self.send_header(
            "Access-Control-Allow-Headers",
            "Content-Type, Cookie, cauth, X-Forwarded-Host, "
            "X-Tenant-Id, X-Conversation-Id, X-User-Id, X-Agent-Id",
        )
        self.send_header("Access-Control-Allow-Methods", "GET, POST, DELETE, OPTIONS")

    def _route_path(self) -> str:
        return (self.path.split("?", 1)[0] or "/").rstrip("/") or "/"

    def _write(self, code: int, body: bytes, content_type: str) -> None:
        self.send_response(code)
        self.send_header("Content-Type", content_type)
        self.send_header("Content-Length", str(len(body)))
        self.send_header("Cache-Control", "no-store")
        self._cors()
        self.end_headers()
        self.wfile.write(body)

    def _write_json(self, code: int, payload: dict | list) -> None:
        self._write(code, _json_bytes(payload), "application/json; charset=utf-8")

    def do_OPTIONS(self) -> None:
        self.send_response(204)
        self._cors()
        self.end_headers()

    def do_GET(self) -> None:
        path = self._route_path()
        if path in {"/health", "/api/health"}:
            policy = get_policy()
            self._write_json(
                200,
                {
                    "ok": RUNNER is not None,
                    "name": policy.agent.name,
                    "mode": RUNNER.mode.value if RUNNER else None,
                    "thread_id": _CURRENT_THREAD,
                    "tracing": is_tracing_active(),
                    "tracing_project": policy.tracing.project if is_tracing_active() else None,
                },
            )
            return
        self._write_json(404, {"error": "not_found"})

    def do_POST(self) -> None:
        global _CURRENT_THREAD
        if RUNNER is None:
            self._write_json(503, {"error": "starting"})
            return
        path = self._route_path()
        length = int(self.headers.get("Content-Length") or 0)
        raw = self.rfile.read(length).decode("utf-8") if length else "{}"
        try:
            payload: dict[str, Any] = json.loads(raw) if raw else {}
        except json.JSONDecodeError:
            payload = {}

        if path in {"/api/chat/clear", "/chat/clear"}:
            _CURRENT_THREAD = RUNNER.clear(_CURRENT_THREAD)
            self._write_json(200, {"ok": True, "thread_id": _CURRENT_THREAD})
            return

        if path not in {"/api/chat", "/chat"}:
            self._write_json(404, {"error": "not_found"})
            return

        try:
            self.connection.setsockopt(socket.IPPROTO_TCP, socket.TCP_NODELAY, 1)
        except OSError:
            pass

        self.send_response(200)
        self.send_header("Content-Type", "text/event-stream; charset=utf-8")
        self.send_header("Cache-Control", "no-cache, no-store")
        self.send_header("Connection", "close")
        self.send_header("X-Accel-Buffering", "no")
        self._cors()
        self.end_headers()
        self.close_connection = True

        def emit(event: str, data: dict) -> None:
            body = json.dumps(data, ensure_ascii=False, default=str)
            self.wfile.write(f"event: {event}\ndata: {body}\n\n".encode("utf-8"))
            self.wfile.flush()

        context = trace_context_from_mapping(payload, _request_headers(self))
        thread_id = str(
            payload.get("thread_id")
            or payload.get("sessionId")
            or payload.get("session_id")
            or context.session_id
            or _CURRENT_THREAD
            or ""
        )
        resume = payload.get("resume")
        text = str(payload.get("text") or payload.get("message") or "").strip()

        # Mode switches via slash commands (no LLM).
        if not resume and text.lower() in {"/ask", "/agent"}:
            RUNNER.set_mode("agent" if text.lower() == "/agent" else "ask")
            emit("system", {"text": f"{RUNNER.mode.value.upper()} mode."})
            emit(
                "done",
                {
                    "reply": "",
                    "mode": RUNNER.mode.value,
                    "thread_id": thread_id or _CURRENT_THREAD,
                },
            )
            return

        try:
            for event in RUNNER.stream(
                text or None,
                thread_id=thread_id or None,
                resume=resume,
                context=context,
            ):
                if event.type == "done" and event.data.get("thread_id"):
                    _CURRENT_THREAD = str(event.data["thread_id"])
                emit(event.type, event.data)
        except Exception as exc:
            logger.exception("chat failed")
            emit("error", {"error": str(exc)})
            emit(
                "done",
                {
                    "reply": "",
                    "mode": RUNNER.mode.value,
                    "thread_id": _CURRENT_THREAD,
                },
            )


def create_runner() -> AgentRunner:
    settings = get_settings()
    logging.basicConfig(
        level=getattr(logging, settings.log_level.upper(), logging.INFO),
        format="%(asctime)s %(levelname)s %(name)s: %(message)s",
    )
    configure_tracing()
    return AgentRunner(settings=settings)


def main() -> None:
    global RUNNER
    policy = get_policy()
    RUNNER = create_runner()
    host = policy.server.host
    port = policy.server.port
    print(f"{policy.agent.name} API: http://127.0.0.1:{port}", flush=True)
    print("  POST /api/chat | POST /api/chat/clear | GET /health", flush=True)
    if is_tracing_active():
        print(f"  LangSmith tracing: on (project={policy.tracing.project})", flush=True)
    else:
        print("  LangSmith tracing: off", flush=True)
    server = ThreadingHTTPServer((host, port), Handler)
    server.serve_forever()


if __name__ == "__main__":
    main()
