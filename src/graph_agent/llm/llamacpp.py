"""llama.cpp (llama-server) helpers: health check and ChatOpenAI client."""

from __future__ import annotations

import json
import urllib.error
import urllib.request

from langchain_openai import ChatOpenAI


def normalize_base_url(base_url: str) -> str:
    url = (base_url or "").strip().rstrip("/")
    if url.endswith("/v1"):
        url = url[:-3].rstrip("/")
    return url


def openai_base_url(base_url: str) -> str:
    return normalize_base_url(base_url) + "/v1"


def server_available(base_url: str, timeout: float = 3.0) -> bool:
    root = normalize_base_url(base_url)
    for path in ("/v1/models", "/health"):
        url = root + path
        try:
            with urllib.request.urlopen(url, timeout=timeout) as response:
                if response.status != 200:
                    continue
                if path == "/health":
                    return True
                payload = json.loads(response.read().decode("utf-8"))
                if isinstance(payload, dict) and ("data" in payload or "models" in payload):
                    return True
                return True
        except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
            continue
    return False


def list_models(base_url: str, timeout: float = 3.0) -> list[str]:
    url = openai_base_url(base_url) + "/models"
    try:
        with urllib.request.urlopen(url, timeout=timeout) as response:
            payload = json.loads(response.read().decode("utf-8"))
    except (urllib.error.URLError, TimeoutError, OSError, json.JSONDecodeError):
        return []
    names: list[str] = []
    for item in payload.get("data") or []:
        if isinstance(item, dict):
            mid = item.get("id") or item.get("model")
            if mid:
                names.append(str(mid))
    return names


def build_llamacpp(model: str, base_url: str) -> ChatOpenAI:
    if not model:
        raise ValueError("LLAMA_CPP_MODEL is not set")
    return ChatOpenAI(
        model=model,
        api_key="llama",
        base_url=openai_base_url(base_url),
        temperature=0,
        streaming=True,
        max_tokens=512,
        extra_body={
            "chat_template_kwargs": {"enable_thinking": False},
        },
    )
