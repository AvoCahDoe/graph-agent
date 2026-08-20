# Graph Agent

**Status: Ready for prod**

Config-driven LangGraph agent core with **no UI**. Integrate via Python API or the headless HTTP/SSE server.

## Install

```bash
pip install -e ".[dev]"
cp .env.example .env   # fill API keys
```

For LangSmith tracing:

```bash
pip install -e ".[tracing]"
# set LANGSMITH_TRACING=true and LANGSMITH_API_KEY in .env
```

## Central config

All behavior is controlled by [`config/agent.yaml`](config/agent.yaml):

- Agent name, modes (`ask` / `agent`), risk policy
- LLM token limits
- Pack definitions (tools, keywords, prompts, hints)
- Planner system prompt
- Server host/port
- Per-tool risk overrides
- **Tracing** (`tracing.enabled`, project, tags, metadata keys)

Secrets stay in `.env`. Precedence for tracing: **env override > yaml > defaults**.

## LangSmith (optional)

LangGraph runs the agent. LangSmith is observability only (nodes, tools, LLM calls, latency, tokens, errors). Product conversation history stays in your store.

Enable:

```bash
LANGSMITH_TRACING=true
LANGSMITH_API_KEY=lsv2_...
LANGSMITH_PROJECT=graph-agent
```

Pass identity so you can find a customer run:

```python
from graph_agent import AgentRunner, TraceContext

runner = AgentRunner()
ctx = TraceContext(
    conversation_id="conv-123",
    tenant_id="acme",
    user_id="u-9",
    session_id="thread-abc",
)
runner.invoke("Show unpaid invoices", thread_id="thread-abc", context=ctx)
```

HTTP body / headers accepted by `/api/chat`:

```json
{
  "text": "...",
  "session_id": "<LangGraph thread>",
  "conversation_id": "<OpenSearch conversation id>",
  "tenant_id": "<mandant>",
  "user_id": "<user>",
  "agent_id": "agent-standalone"
}
```

Headers (optional; body wins): `X-Tenant-Id`, `X-Conversation-Id`, `X-User-Id`, `X-Agent-Id`.

`GET /health` includes `"tracing": true|false`.

## Python API

```python
from langchain_core.tools import tool
from graph_agent import AgentRunner
from graph_agent.tools.registry import RiskLevel, register_tool

@register_tool(risk_level=RiskLevel.READ_ONLY)
@tool
def get_invoice(invoice_id: str) -> str:
    """Fetch an invoice by id."""
    return f'{{"invoiceId":"{invoice_id}"}}'

runner = AgentRunner()
result = runner.invoke("Show invoice INV-1", thread_id="demo")
print(result["reply"])

for event in runner.stream("What can you do?", thread_id="demo"):
    if event.type == "token":
        print(event.text, end="")
```

## HTTP server (sidecar contract)

```bash
graph-agent
# or: python -m graph_agent.server.app
```

| Endpoint | Method | Purpose |
|----------|--------|---------|
| `/api/chat` | POST | `{ "text": "…" }` or `{ "resume": "approve" }` → SSE |
| `/api/chat/clear` | POST | Reset thread |
| `/health` | GET | Health (+ tracing flag) |

SSE events: `token`, `interrupt`, `error`, `system`, `done`.

Compatible with a BFF sidecar pattern (`Cookie` / `X-Forwarded-Host` forwarded by the BFF).

## Integrate with your backend

1. Register domain tools with `@register_tool` / `@tool`.
2. Customize `config/agent.yaml` packs for your domain.
3. Deploy as a Docker service and point your BFF at this service’s `/api/chat`.
4. Optionally send `conversation_id` / `tenant_id` / `user_id` for LangSmith metadata.

## Project layout

```
src/graph_agent/
  config.py / policy.py   # env + YAML
  tracing.py              # LangSmith configure + run config metadata
  agent/                  # LangGraph graph, planner, specialists, HITL
  llm/                    # provider factory
  tools/                  # registry + built-in ask_user
  runner.py               # AgentRunner
  server/app.py           # headless HTTP/SSE
config/agent.yaml         # centralized behavior
```

## TODO

- [ ] Add Docker image + healthcheck for sidecar deploy
- [ ] Wire ERP tools for production tenants
- [ ] Harden auth / tenant isolation on `/api/chat`
- [ ] Publish package version and changelog for release
- [ ] Add CI (pytest + lint) on push
- [ ] Document production env vars and secrets rotation
