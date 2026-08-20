"""DeepSeek chat model via the OpenAI-compatible API."""

from __future__ import annotations

from langchain_openai import ChatOpenAI

DEFAULT_BASE_URL = "https://api.deepseek.com"


def build_deepseek(model: str, api_key: str, base_url: str = DEFAULT_BASE_URL) -> ChatOpenAI:
    if not api_key:
        raise ValueError("DEEPSEEK_API_KEY is not set")
    return ChatOpenAI(
        model=model,
        api_key=api_key,
        base_url=base_url.rstrip("/"),
        temperature=0,
        streaming=True,
        max_tokens=512,
    )
