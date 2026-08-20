"""LLM factory: llama.cpp (default), DeepSeek, Anthropic — with retries."""

from __future__ import annotations

import logging

from langchain_core.language_models.chat_models import BaseChatModel
from tenacity import retry, stop_after_attempt, wait_exponential

from graph_agent.config import Settings, get_settings
from graph_agent.llm.deepseek import build_deepseek
from graph_agent.llm.llamacpp import build_llamacpp, list_models, server_available
from graph_agent.policy import get_policy

logger = logging.getLogger(__name__)


class LLMFactory:
    """Create a chat model.

    `LLM_PROVIDER=auto` (default): llama.cpp if reachable, else DeepSeek, else Anthropic.
    `llamacpp` / `deepseek` / `anthropic`: force that backend (with fallbacks for llamacpp).
    """

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _create_llamacpp(model: str, base_url: str) -> BaseChatModel:
        return build_llamacpp(model, base_url)

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _create_deepseek(model: str, api_key: str, base_url: str) -> BaseChatModel:
        return build_deepseek(model, api_key, base_url)

    @staticmethod
    @retry(
        stop=stop_after_attempt(3),
        wait=wait_exponential(multiplier=1, min=2, max=10),
        reraise=True,
    )
    def _create_anthropic(model: str, api_key: str) -> BaseChatModel:
        from langchain_anthropic import ChatAnthropic

        if not api_key:
            raise ValueError("ANTHROPIC_API_KEY is not set")
        return ChatAnthropic(model=model, api_key=api_key, temperature=0)

    @staticmethod
    def _try_llamacpp(cfg: Settings, errors: list[str]) -> BaseChatModel | None:
        if not server_available(cfg.llama_cpp_base_url):
            logger.warning("llama.cpp is not reachable at %s", cfg.llama_cpp_base_url)
            errors.append(f"llamacpp: not reachable at {cfg.llama_cpp_base_url}")
            return None
        model = cfg.llama_cpp_model
        installed = list_models(cfg.llama_cpp_base_url)
        if installed and model not in installed:
            if len(installed) == 1:
                logger.warning(
                    "llama.cpp model %r not in /v1/models; using %r",
                    model,
                    installed[0],
                )
                model = installed[0]
            else:
                logger.warning(
                    "llama.cpp model %r not listed (available: %s); trying configured id anyway",
                    model,
                    ", ".join(installed),
                )
        try:
            llm = LLMFactory._create_llamacpp(model, cfg.llama_cpp_base_url)
            logger.info("Using LLM provider: llamacpp (%s)", model)
            return llm
        except Exception as exc:
            logger.warning("llama.cpp failed: %s", exc)
            errors.append(f"llamacpp: {exc}")
            return None

    @staticmethod
    def _try_deepseek(cfg: Settings, errors: list[str]) -> BaseChatModel | None:
        try:
            llm = LLMFactory._create_deepseek(
                cfg.deepseek_model, cfg.deepseek_api_key, cfg.deepseek_base_url
            )
            logger.info("Using LLM provider: deepseek (%s)", cfg.deepseek_model)
            return llm
        except Exception as exc:
            logger.warning("DeepSeek failed: %s", exc)
            errors.append(f"deepseek: {exc}")
            return None

    @staticmethod
    def _try_anthropic(cfg: Settings, errors: list[str]) -> BaseChatModel | None:
        try:
            llm = LLMFactory._create_anthropic(cfg.anthropic_model, cfg.anthropic_api_key)
            logger.info("Using LLM provider: anthropic (%s)", cfg.anthropic_model)
            return llm
        except Exception as exc:
            logger.warning("Anthropic failed: %s", exc)
            errors.append(f"anthropic: {exc}")
            return None

    @staticmethod
    def create(settings: Settings | None = None) -> BaseChatModel:
        cfg = settings or get_settings()
        # Env wins; YAML provider is a soft default when env is auto/empty.
        provider = (cfg.llm_provider or "auto").strip().lower()
        if provider in {"", "auto"}:
            yaml_provider = (get_policy().llm.provider or "auto").strip().lower()
            if yaml_provider and yaml_provider != "auto":
                provider = yaml_provider
        errors: list[str] = []

        prefer_llama = provider in {"", "auto", "llamacpp", "llama.cpp", "llama_cpp"}
        if prefer_llama:
            llm = LLMFactory._try_llamacpp(cfg, errors)
            if llm:
                return llm
            if provider in {"llamacpp", "llama.cpp", "llama_cpp"}:
                logger.warning("llama.cpp unavailable; falling back to DeepSeek / Anthropic")
            llm = LLMFactory._try_deepseek(cfg, errors)
            if llm:
                return llm
            llm = LLMFactory._try_anthropic(cfg, errors)
            if llm:
                return llm
        elif provider == "deepseek":
            llm = LLMFactory._try_deepseek(cfg, errors)
            if llm:
                return llm
        elif provider == "anthropic":
            llm = LLMFactory._try_anthropic(cfg, errors)
            if llm:
                return llm
        else:
            raise RuntimeError(
                f"Unknown LLM_PROVIDER={provider!r}. Use auto, llamacpp, deepseek, or anthropic."
            )

        raise RuntimeError("LLM unavailable, try again later. " + " | ".join(errors))
