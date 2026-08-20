"""Runtime settings from environment / .env."""

from __future__ import annotations

import os
from pathlib import Path

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict

PACKAGE_ROOT = Path(__file__).resolve().parent
PROJECT_ROOT = PACKAGE_ROOT.parent.parent

load_dotenv(PROJECT_ROOT / ".env", override=False)


def sync_langsmith_env() -> None:
    """Mirror LANGSMITH_* → LANGCHAIN_* so LangChain tracing picks them up."""
    if os.getenv("LANGSMITH_API_KEY") and not os.getenv("LANGCHAIN_API_KEY"):
        os.environ["LANGCHAIN_API_KEY"] = os.environ["LANGSMITH_API_KEY"]
    if os.getenv("LANGSMITH_TRACING", "").lower() in {"1", "true", "yes", "on"}:
        os.environ.setdefault("LANGCHAIN_TRACING_V2", "true")
    if os.getenv("LANGSMITH_PROJECT") and not os.getenv("LANGCHAIN_PROJECT"):
        os.environ["LANGCHAIN_PROJECT"] = os.environ["LANGSMITH_PROJECT"]
    if os.getenv("LANGSMITH_WORKSPACE_ID") and not os.getenv("LANGCHAIN_WORKSPACE_ID"):
        os.environ["LANGCHAIN_WORKSPACE_ID"] = os.environ["LANGSMITH_WORKSPACE_ID"]


sync_langsmith_env()


class Settings(BaseSettings):
    """Environment secrets and paths. Behavioral knobs live in agent.yaml."""

    model_config = SettingsConfigDict(
        env_file=str(PROJECT_ROOT / ".env"),
        env_file_encoding="utf-8",
        extra="ignore",
    )

    llama_cpp_base_url: str = "http://localhost:8080"
    llama_cpp_model: str = "qwen3-8b"

    llm_provider: str = "auto"

    deepseek_api_key: str = ""
    deepseek_model: str = "deepseek-chat"
    deepseek_base_url: str = "https://api.deepseek.com"

    anthropic_api_key: str = ""
    anthropic_model: str = "claude-sonnet-4-20250514"

    log_level: str = "INFO"
    default_mode: str = "ask"
    policy_file: str = "config/agent.yaml"

    langsmith_api_key: str = ""
    langsmith_tracing: bool = False
    langsmith_project: str = ""
    langsmith_workspace_id: str = ""

    @property
    def policy_path(self) -> Path:
        path = Path(self.policy_file)
        if not path.is_absolute():
            path = PROJECT_ROOT / path
        return path


def get_settings() -> Settings:
    return Settings()
