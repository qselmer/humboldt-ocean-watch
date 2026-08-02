"""Provider-neutral settings, response model, protocol, and factory."""

from __future__ import annotations

from dataclasses import dataclass, field
import math
import os
from pathlib import Path
from typing import Any, Mapping, Protocol, runtime_checkable

from src.llm_errors import LLMConfigurationError, LLMProviderError


DEFAULT_PROVIDER = "gemini"
DEFAULT_GEMINI_MODEL = "gemini-3.5-flash"
DEFAULT_OLLAMA_MODEL = "qwen3:4b"
DEFAULT_OLLAMA_BASE_URL = "http://localhost:11434"


@dataclass(frozen=True)
class LLMProviderSettings:
    provider_name: str
    model_name: str
    base_url: str | None
    local_generation: bool
    temperature: float = 0.0
    think: bool = False
    num_ctx: int | None = None
    num_predict: int = 3500
    keep_alive: str | None = None
    request_timeout_seconds: float | None = None
    structured_output_mode: str = "json_schema"
    attempt_provider_schema: bool = False
    reasoning_effort: str | None = None
    text_verbosity: str | None = None
    maximum_repair_attempts: int = 1
    approximate_bytes_per_token: float = 3.0
    context_warning_fraction: float = 0.80
    api_key_environment_variable: str | None = None
    use_tools: bool = False
    use_search_grounding: bool = False
    include_thoughts: bool = False
    maximum_transport_attempts: int = 1
    retry_initial_seconds: float = 15.0
    retry_maximum_seconds: float = 60.0

    def approximate_tokens(self, request_bytes: int) -> int:
        return int(math.ceil(request_bytes / self.approximate_bytes_per_token))


@dataclass(frozen=True)
class StructuredGeneration:
    content: str
    provider: str
    model: str | None
    response_id: str | None = None
    status: str = "completed"
    usage: dict[str, int | None] = field(default_factory=dict)
    provider_metadata: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class LLMProvider(Protocol):
    provider_name: str
    model_name: str
    base_url: str | None
    supports_structured_output: bool
    local_generation: bool
    settings: LLMProviderSettings

    def preflight(self, *, output_directory: Path) -> Any: ...
    def generate_structured(self, prompt: Any, schema: Mapping[str, Any]) -> StructuredGeneration: ...
    def normalize_error(self, exc: Exception) -> LLMProviderError: ...
    def extract_generation_metadata(self, response: Any) -> dict[str, Any]: ...


def resolve_provider_name(explicit: str | None, configuration: Mapping[str, Any]) -> str:
    selected = explicit or os.getenv("LLM_PROVIDER") or configuration.get("provider") or DEFAULT_PROVIDER
    name = str(selected).strip().casefold()
    if name not in {"gemini", "ollama", "openai"}:
        raise LLMConfigurationError("unsupported_provider", f"Unsupported LLM provider: {name or '<empty>'}")
    return name


def _positive_int(value: Any, name: str) -> int:
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise LLMConfigurationError("context_configuration_invalid", f"{name} must be a positive integer") from exc
    if parsed <= 0:
        raise LLMConfigurationError("context_configuration_invalid", f"{name} must be a positive integer")
    return parsed


def _positive_float(value: Any, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise LLMConfigurationError(
            "request_timeout_invalid", f"{name} must be a positive number"
        ) from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise LLMConfigurationError(
            "request_timeout_invalid", f"{name} must be a positive number"
        )
    return parsed


def _ollama_structured_output_mode(
    provider_config: Mapping[str, Any],
) -> tuple[str, bool]:
    configured = str(provider_config.get("structured_output_mode", "json")).strip().casefold()
    environment = os.getenv("OLLAMA_STRUCTURED_OUTPUT_MODE")
    selected = str(environment).strip().casefold() if environment is not None else configured
    if selected not in {"json", "json_schema"}:
        raise LLMConfigurationError(
            "structured_output_mode_invalid",
            "OLLAMA_STRUCTURED_OUTPUT_MODE must be 'json' or 'json_schema'",
        )
    attempt_provider_schema = bool(provider_config.get("attempt_provider_schema", False))
    if environment is not None and selected == "json_schema":
        # An explicit environment selection is an intentional schema experiment.
        attempt_provider_schema = True
    if selected == "json_schema" and not attempt_provider_schema:
        raise LLMConfigurationError(
            "provider_schema_attempt_disabled",
            "Ollama JSON Schema mode requires attempt_provider_schema=true or an explicit "
            "OLLAMA_STRUCTURED_OUTPUT_MODE=json_schema override",
        )
    return selected, attempt_provider_schema


def _gemini_structured_output_mode(provider_config: Mapping[str, Any]) -> str:
    configured = str(
        provider_config.get("structured_output_mode", "json")
    ).strip().casefold()
    selected = str(
        os.getenv("GEMINI_STRUCTURED_OUTPUT_MODE") or configured
    ).strip().casefold()
    if selected not in {"json_schema", "json"}:
        raise LLMConfigurationError(
            "structured_output_mode_invalid",
            "GEMINI_STRUCTURED_OUTPUT_MODE must be 'json_schema' or 'json'",
        )
    return selected


def resolve_provider_settings(
    provider_name: str | None,
    model: str | None,
    base_url: str | None,
    configuration: Mapping[str, Any],
) -> LLMProviderSettings:
    name = resolve_provider_name(provider_name, configuration)
    providers = configuration.get("providers", {})
    provider_config = providers.get(name, {}) if isinstance(providers, Mapping) else {}
    if name == "gemini":
        selected_model = (
            model
            or os.getenv("GEMINI_MODEL")
            or provider_config.get("model")
            or DEFAULT_GEMINI_MODEL
        )
        maximum_output_tokens = _positive_int(
            os.getenv("GEMINI_MAXIMUM_OUTPUT_TOKENS")
            or provider_config.get("maximum_output_tokens", 3500),
            "GEMINI_MAXIMUM_OUTPUT_TOKENS",
        )
        request_timeout_seconds = _positive_float(
            os.getenv("GEMINI_REQUEST_TIMEOUT_SECONDS")
            or provider_config.get("request_timeout_seconds", 300),
            "GEMINI_REQUEST_TIMEOUT_SECONDS",
        )
        structured_output_mode = _gemini_structured_output_mode(provider_config)
        maximum_transport_attempts = _positive_int(
            os.getenv("GEMINI_MAXIMUM_TRANSPORT_ATTEMPTS")
            or provider_config.get("maximum_transport_attempts", 3),
            "GEMINI_MAXIMUM_TRANSPORT_ATTEMPTS",
        )
        if maximum_transport_attempts > 3:
            raise LLMConfigurationError(
                "transport_retry_configuration_invalid",
                "GEMINI_MAXIMUM_TRANSPORT_ATTEMPTS must be between 1 and 3",
            )
        retry_initial_seconds = _positive_float(
            os.getenv("GEMINI_RETRY_INITIAL_SECONDS")
            or provider_config.get("retry_initial_seconds", 15),
            "GEMINI_RETRY_INITIAL_SECONDS",
        )
        retry_maximum_seconds = _positive_float(
            os.getenv("GEMINI_RETRY_MAXIMUM_SECONDS")
            or provider_config.get("retry_maximum_seconds", 60),
            "GEMINI_RETRY_MAXIMUM_SECONDS",
        )
        settings = LLMProviderSettings(
            provider_name=name,
            model_name=str(selected_model).strip(),
            base_url=None,
            local_generation=False,
            temperature=float(provider_config.get("temperature", 0)),
            num_predict=maximum_output_tokens,
            request_timeout_seconds=request_timeout_seconds,
            structured_output_mode=structured_output_mode,
            attempt_provider_schema=structured_output_mode == "json_schema",
            maximum_repair_attempts=int(
                provider_config.get("maximum_repair_attempts", 1)
            ),
            approximate_bytes_per_token=float(
                provider_config.get("approximate_bytes_per_token", 3.0)
            ),
            context_warning_fraction=float(
                provider_config.get("context_warning_fraction", 0.80)
            ),
            api_key_environment_variable=str(
                provider_config.get(
                    "api_key_environment_variable", "GEMINI_API_KEY"
                )
            ),
            use_tools=bool(provider_config.get("use_tools", False)),
            use_search_grounding=bool(
                provider_config.get("use_search_grounding", False)
            ),
            include_thoughts=bool(provider_config.get("include_thoughts", False)),
            maximum_transport_attempts=maximum_transport_attempts,
            retry_initial_seconds=retry_initial_seconds,
            retry_maximum_seconds=retry_maximum_seconds,
        )
        if settings.use_tools or settings.use_search_grounding:
            raise LLMConfigurationError(
                "external_tools_not_disabled",
                "Gemini tools and search grounding must be disabled for scientific briefs",
            )
        if settings.include_thoughts:
            raise LLMConfigurationError(
                "thinking_not_disabled",
                "Gemini thought output must be disabled for scientific briefs",
            )
        return settings

    if name == "ollama":
        selected_model = model or os.getenv("OLLAMA_MODEL") or provider_config.get("model") or DEFAULT_OLLAMA_MODEL
        selected_url = base_url or os.getenv("OLLAMA_BASE_URL") or provider_config.get("base_url") or DEFAULT_OLLAMA_BASE_URL
        num_ctx = _positive_int(os.getenv("OLLAMA_NUM_CTX") or provider_config.get("num_ctx", 65536), "OLLAMA_NUM_CTX")
        num_predict = _positive_int(provider_config.get("num_predict", 3500), "num_predict")
        request_timeout_seconds = _positive_float(
            os.getenv("OLLAMA_REQUEST_TIMEOUT_SECONDS")
            or provider_config.get("request_timeout_seconds", 600),
            "OLLAMA_REQUEST_TIMEOUT_SECONDS",
        )
        structured_output_mode, attempt_provider_schema = _ollama_structured_output_mode(
            provider_config
        )
        settings = LLMProviderSettings(
            provider_name=name,
            model_name=str(selected_model).strip(),
            base_url=str(selected_url).rstrip("/"),
            local_generation=True,
            temperature=float(provider_config.get("temperature", 0)),
            think=bool(provider_config.get("think", False)),
            num_ctx=num_ctx,
            num_predict=num_predict,
            keep_alive=str(provider_config.get("keep_alive", "5m")),
            request_timeout_seconds=request_timeout_seconds,
            structured_output_mode=structured_output_mode,
            attempt_provider_schema=attempt_provider_schema,
            maximum_repair_attempts=int(provider_config.get("maximum_repair_attempts", 1)),
            approximate_bytes_per_token=float(provider_config.get("approximate_bytes_per_token", 3.0)),
            context_warning_fraction=float(provider_config.get("context_warning_fraction", 0.80)),
        )
        if settings.think:
            raise LLMConfigurationError("thinking_not_disabled", "Ollama thinking must be disabled for scientific briefs")
        return settings

    selected_model = model or os.getenv("OPENAI_MODEL") or provider_config.get("model") or "gpt-5.6"
    return LLMProviderSettings(
        provider_name=name,
        model_name=str(selected_model).strip(),
        base_url=None,
        local_generation=False,
        num_predict=_positive_int(provider_config.get("maximum_output_tokens", 3500), "maximum_output_tokens"),
        reasoning_effort=str(provider_config.get("reasoning_effort", "medium")),
        text_verbosity=str(provider_config.get("text_verbosity", "medium")),
        maximum_repair_attempts=int(provider_config.get("maximum_repair_attempts", 1)),
    )


def get_llm_provider(
    provider_name: str | None,
    model: str | None,
    configuration: Mapping[str, Any],
    *,
    base_url: str | None = None,
    client: Any | None = None,
) -> LLMProvider:
    settings = resolve_provider_settings(provider_name, model, base_url, configuration)
    if settings.provider_name == "gemini":
        from src.gemini_client import GeminiProvider

        return GeminiProvider(settings, client=client)
    if settings.provider_name == "ollama":
        from src.ollama_client import OllamaProvider

        return OllamaProvider(settings, client=client)
    if settings.provider_name == "openai":
        from src.openai_client import OpenAIProvider

        return OpenAIProvider(settings, client=client)
    raise LLMConfigurationError("unsupported_provider", f"Unsupported LLM provider: {settings.provider_name}")
