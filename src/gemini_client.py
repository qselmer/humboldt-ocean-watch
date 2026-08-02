"""Native Google GenAI provider for validated scientific briefs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import importlib.metadata
import inspect
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping

from src.brief_prompts import PromptPackage
from src.export_utils import dumps_json_safe
from src.gemini_schema_adapter import build_gemini_compatible_schema
from src.llm_errors import LLMConfigurationError, LLMProviderError
from src.llm_provider import LLMProviderSettings, StructuredGeneration
from src.ollama_client import build_json_mode_user_text


LOGGER = logging.getLogger("humboldt_ocean_watch.scientific_brief")


@dataclass(frozen=True)
class GeminiSDK:
    genai: Any
    types: Any
    version: str


class _InjectedConfiguration(dict[str, Any]):
    """Dict-compatible config used only by offline injected test clients."""

    def __getattr__(self, name: str) -> Any:
        try:
            return self[name]
        except KeyError as exc:  # pragma: no cover - ordinary attribute semantics
            raise AttributeError(name) from exc


class InjectedClientTypes:
    """Minimal native-config stand-in when an offline fake client is injected."""

    GenerateContentConfig = _InjectedConfiguration
    ThinkingConfig = _InjectedConfiguration
    AutomaticFunctionCallingConfig = _InjectedConfiguration


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _enum_name(value: Any) -> str:
    name = getattr(value, "name", None)
    return str(name or value or "").split(".")[-1].upper()


def load_gemini_sdk() -> GeminiSDK:
    """Import the maintained native SDK without touching the network."""
    try:
        from google import genai
        from google.genai import types
    except ImportError as exc:
        raise LLMConfigurationError(
            "gemini_package_missing",
            "The google-genai Python package is not installed",
        ) from exc
    try:
        version = importlib.metadata.version("google-genai")
    except importlib.metadata.PackageNotFoundError:
        version = "unknown"
    return GeminiSDK(genai=genai, types=types, version=version)


def gemini_api_key_from_environment(variable: str = "GEMINI_API_KEY") -> str:
    value = os.getenv(variable)
    if value is None or not value.strip():
        raise LLMConfigurationError(
            "missing_api_key",
            f"{variable} is not set",
        )
    return value.strip()


def _declared_fields(value: Any) -> set[str]:
    model_fields = getattr(value, "model_fields", None)
    if isinstance(model_fields, Mapping):
        return set(model_fields)
    annotations = getattr(value, "__annotations__", None)
    if isinstance(annotations, Mapping):
        return set(annotations)
    try:
        return set(inspect.signature(value).parameters)
    except (TypeError, ValueError):
        return set()


def validate_installed_gemini_contract() -> str:
    """Validate native SDK names used by this provider without an API call."""
    sdk = load_gemini_sdk()
    if not callable(getattr(sdk.genai, "Client", None)):
        raise LLMConfigurationError(
            "gemini_package_missing", "The google-genai Client is unavailable"
        )
    config_type = getattr(sdk.types, "GenerateContentConfig", None)
    required_config = {
        "automatic_function_calling",
        "temperature",
        "max_output_tokens",
        "response_mime_type",
        "response_json_schema",
        "system_instruction",
    }
    if config_type is None or not required_config <= _declared_fields(config_type):
        raise LLMConfigurationError(
            "gemini_package_missing",
            "The installed google-genai package lacks required structured-output fields",
        )

    client = sdk.genai.Client(api_key="sdk-contract-check")
    try:
        method = getattr(getattr(client, "models", None), "generate_content", None)
        if not callable(method):
            raise LLMConfigurationError(
                "gemini_package_missing",
                "The installed google-genai package lacks models.generate_content",
            )
        parameters = set(inspect.signature(method).parameters)
        if not {"model", "contents", "config"} <= parameters:
            raise LLMConfigurationError(
                "gemini_package_missing",
                "The installed google-genai Generate Content signature is incompatible",
            )
    finally:
        close = getattr(client, "close", None)
        if callable(close):
            close()
    return sdk.version


def create_gemini_client(
    *,
    api_key: str | None = None,
    request_timeout_seconds: float | None = None,
    sdk: GeminiSDK | None = None,
) -> Any:
    active_sdk = sdk or load_gemini_sdk()
    kwargs: dict[str, Any] = {
        "api_key": api_key or gemini_api_key_from_environment(),
    }
    http_options_type = getattr(active_sdk.types, "HttpOptions", None)
    if callable(http_options_type) and request_timeout_seconds is not None:
        http_kwargs: dict[str, Any] = {
            # Native Google GenAI HttpOptions uses milliseconds.
            "timeout": int(round(float(request_timeout_seconds) * 1000)),
        }
        retry_type = getattr(active_sdk.types, "HttpRetryOptions", None)
        if callable(retry_type):
            http_kwargs["retry_options"] = retry_type(attempts=1)
        kwargs["http_options"] = http_options_type(**http_kwargs)
    return active_sdk.genai.Client(**kwargs)


def normalize_gemini_error(exc: Exception, *, request_mode: str | None = None) -> LLMProviderError:
    """Map native SDK failures to fixed messages that cannot expose secrets."""
    class_name = type(exc).__name__.casefold()
    error_type_names = " ".join(
        candidate.__name__.casefold() for candidate in type(exc).__mro__
    )
    status_code = None
    for candidate in (getattr(exc, "status_code", None), getattr(exc, "code", None)):
        try:
            status_code = int(candidate)
        except (TypeError, ValueError):
            continue
        else:
            break
    provider_status = str(getattr(exc, "status", "") or "").strip().upper()
    if not provider_status:
        code_value = getattr(exc, "code", None)
        if isinstance(code_value, str):
            provider_status = code_value.strip().upper()
    recognized_statuses = {
        "DEADLINE_EXCEEDED",
        "INTERNAL",
        "INVALID_ARGUMENT",
        "NOT_FOUND",
        "PERMISSION_DENIED",
        "RESOURCE_EXHAUSTED",
        "UNAUTHENTICATED",
        "UNAVAILABLE",
    }
    safe_provider_status = provider_status if provider_status in recognized_statuses else None
    internal_text = " ".join(
        str(value).casefold()
        for value in (getattr(exc, "message", None), str(exc))
        if value
    )
    if any(term in internal_text for term in ("capacity", "overloaded", "unavailable")):
        message_category = "provider_capacity"
    elif any(term in internal_text for term in ("quota", "resource_exhausted")):
        message_category = "quota"
    elif any(term in internal_text for term in ("rate limit", "ratelimit")):
        message_category = "rate_limit"
    elif any(term in internal_text for term in ("deadline", "timed out", "timeout")):
        message_category = "timeout"
    elif any(term in internal_text for term in ("schema", "response_json_schema", "structured", "mime type")):
        message_category = "structured_output"
    elif any(term in internal_text for term in ("api key", "api_key", "credential", "authentication")):
        message_category = "authentication"
    elif any(term in internal_text for term in ("model", "not found")):
        message_category = "model_access"
    elif any(term in internal_text for term in ("connection", "network", "transport")):
        message_category = "network"
    else:
        message_category = "unknown"
    diagnostics: dict[str, Any] = {
        "http_status_code": status_code,
        "provider_status": safe_provider_status,
        "provider_message_category": message_category,
        "request_mode": request_mode,
    }

    if status_code == 401 or safe_provider_status == "UNAUTHENTICATED" or "api_key_invalid" in internal_text or "authentication" in class_name:
        code, message, retryable = "invalid_api_key", "Gemini rejected the API credentials", False
    elif status_code in {403, 404} or safe_provider_status in {"PERMISSION_DENIED", "NOT_FOUND"} or "permission" in class_name or "not found" in internal_text:
        code, message, retryable = "inaccessible_model", "The selected Gemini model is not accessible", False
    elif status_code == 429 and (message_category == "quota" or safe_provider_status == "RESOURCE_EXHAUSTED"):
        code, message, retryable = "quota_exceeded", "The Gemini API quota was exceeded", True
    elif status_code == 429 or safe_provider_status == "RESOURCE_EXHAUSTED" or "ratelimit" in class_name or "rate limit" in internal_text:
        code, message, retryable = "rate_limit", "The Gemini rate limit was reached", True
    elif status_code == 504 or safe_provider_status == "DEADLINE_EXCEEDED" or "timeout" in error_type_names or message_category == "timeout":
        code, message, retryable = "request_timeout", "The Gemini request timed out", True
    elif status_code in {502, 503} or safe_provider_status == "UNAVAILABLE":
        code, message, retryable = (
            "service_unavailable",
            "Gemini is temporarily unavailable because provider capacity is constrained",
            True,
        )
    elif status_code == 500 or safe_provider_status == "INTERNAL":
        code, message, retryable = (
            "provider_internal_error",
            "Gemini encountered a temporary internal provider error",
            True,
        )
    elif isinstance(exc, ConnectionError) or any(
        term in error_type_names
        for term in (
            "connect",
            "connection",
            "network",
            "protocolerror",
            "readerror",
            "transport",
            "writeerror",
        )
    ):
        code, message, retryable = "network_failure", "The Gemini API could not be reached", True
    elif "max_tokens" in internal_text or "token limit" in internal_text:
        code, message, retryable = "output_token_limit", "Gemini reached the output-token limit", False
    elif status_code == 400 and any(
        term in internal_text
        for term in ("schema", "response_json_schema", "structured", "mime type")
    ):
        code, message, retryable = (
            "structured_output_failure",
            "Gemini rejected the structured-output schema",
            False,
        )
    elif status_code == 400:
        code, message, retryable = "structured_output_failure", "Gemini rejected the generation request", False
    else:
        code, message, retryable = "unknown_gemini_error", "An unknown Gemini API error occurred", False
    diagnostics["error_code"] = code
    return LLMProviderError(code, message, retryable=retryable, diagnostics=diagnostics)


def _disabled_automatic_function_calling(types_module: Any) -> Any:
    config_type = getattr(types_module, "AutomaticFunctionCallingConfig", None)
    if not callable(config_type):
        raise LLMConfigurationError(
            "gemini_package_missing",
            "The installed google-genai package cannot disable automatic function calling",
        )
    return config_type(disable=True)


def build_gemini_request(
    prompt: PromptPackage,
    provider_schema: Mapping[str, Any] | None,
    settings: LLMProviderSettings,
    *,
    types_module: Any,
    strict_schema: Mapping[str, Any],
) -> dict[str, Any]:
    """Build one native, non-streaming Generate Content request."""
    user_text = prompt.user_text
    if settings.structured_output_mode == "json":
        user_text = build_json_mode_user_text(prompt, strict_schema)
    config_kwargs: dict[str, Any] = {
        "automatic_function_calling": _disabled_automatic_function_calling(types_module),
        "system_instruction": prompt.developer_text,
        "temperature": settings.temperature,
        "max_output_tokens": settings.num_predict,
        "response_mime_type": "application/json",
    }
    if provider_schema is not None:
        config_kwargs["response_json_schema"] = dict(provider_schema)
    thinking_type = getattr(types_module, "ThinkingConfig", None)
    if callable(thinking_type):
        config_kwargs["thinking_config"] = thinking_type(include_thoughts=False)
    config = types_module.GenerateContentConfig(**config_kwargs)
    return {
        "model": settings.model_name,
        "contents": user_text,
        "config": config,
    }


def _response_finish_reason(response: Any) -> str:
    candidates = _get(response, "candidates", []) or []
    if not candidates:
        return ""
    return _enum_name(_get(candidates[0], "finish_reason"))


def extract_gemini_text(response: Any) -> str:
    """Extract visible model text while excluding thought parts."""
    prompt_feedback = _get(response, "prompt_feedback")
    block_reason = _enum_name(_get(prompt_feedback, "block_reason"))
    if block_reason and block_reason not in {"BLOCK_REASON_UNSPECIFIED", "UNSPECIFIED"}:
        raise LLMProviderError("provider_refusal", "Gemini blocked the request")

    candidates = _get(response, "candidates", []) or []
    if not candidates:
        raise LLMProviderError("provider_refusal", "Gemini returned no response candidate")
    finish_reason = _response_finish_reason(response)
    if finish_reason in {"MAX_TOKENS", "MAX_OUTPUT_TOKENS"}:
        raise LLMProviderError("output_token_limit", "Gemini reached the output-token limit")
    if finish_reason in {
        "SAFETY",
        "BLOCKLIST",
        "PROHIBITED_CONTENT",
        "RECITATION",
        "SPII",
    }:
        raise LLMProviderError("provider_refusal", "Gemini declined to provide the requested output")
    if finish_reason and finish_reason != "STOP":
        raise LLMProviderError(
            "structured_output_failure",
            "Gemini did not complete the structured response",
        )

    content = _get(candidates[0], "content")
    parts = _get(content, "parts", []) or []
    visible: list[str] = []
    for part in parts:
        if bool(_get(part, "thought", False)):
            continue
        text = _get(part, "text")
        if isinstance(text, str) and text:
            visible.append(text)
    if not visible:
        try:
            fallback = _get(response, "text")
        except Exception as exc:
            raise LLMProviderError(
                "malformed_structured_output",
                "Gemini returned no visible structured output",
            ) from exc
        if isinstance(fallback, str) and fallback.strip():
            visible.append(fallback)
    result = "".join(visible).strip()
    if not result:
        raise LLMProviderError(
            "malformed_structured_output",
            "Gemini returned no visible structured output",
        )
    return result


def gemini_usage(response: Any) -> dict[str, int | None]:
    usage = _get(response, "usage_metadata")
    return {
        "input_tokens": _get(usage, "prompt_token_count"),
        "cached_input_tokens": _get(usage, "cached_content_token_count"),
        "output_tokens": _get(usage, "candidates_token_count"),
        "reasoning_tokens": None,
        "total_tokens": _get(usage, "total_token_count"),
    }


def _response_identifier(response: Any) -> str | None:
    value = _get(response, "response_id") or _get(response, "id")
    return str(value) if value else None


class GeminiProvider:
    """Provider-neutral adapter over native ``google.genai.Client``."""

    provider_name = "gemini"
    supports_structured_output = True
    local_generation = False

    def __init__(
        self,
        settings: LLMProviderSettings,
        *,
        client: Any | None = None,
        types_module: Any | None = None,
        sleep: Any = time.sleep,
    ) -> None:
        self.settings = settings
        self.model_name = settings.model_name
        self.base_url = None
        self._client = client
        self._types_module = types_module or getattr(client, "types", None)
        self._owns_client = client is None
        self._sleep = sleep

    def _active_types(self) -> Any:
        if self._types_module is None:
            try:
                self._types_module = load_gemini_sdk().types
            except LLMConfigurationError:
                if self._client is None:
                    raise
                self._types_module = InjectedClientTypes
        return self._types_module

    def _active_client(self) -> Any:
        if self._client is None:
            variable = self.settings.api_key_environment_variable or "GEMINI_API_KEY"
            self._client = create_gemini_client(
                api_key=gemini_api_key_from_environment(variable),
                request_timeout_seconds=self.settings.request_timeout_seconds,
            )
        return self._client

    def _release_owned_client(self) -> None:
        if not self._owns_client or self._client is None:
            return
        client, self._client = self._client, None
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                LOGGER.warning("Gemini client cleanup failed")

    def preflight(self, *, output_directory: Any) -> Any:
        from src.gemini_preflight import run_gemini_preflight

        return run_gemini_preflight(
            settings=self.settings,
            output_directory=output_directory,
            client=self._client,
            types_module=self._types_module,
        )

    def validate_local_environment(self, *, output_directory: Path) -> None:
        """Run local generation readiness checks without a provider request."""
        validate_installed_gemini_contract()
        variable = self.settings.api_key_environment_variable or "GEMINI_API_KEY"
        gemini_api_key_from_environment(variable)
        output_directory.mkdir(parents=True, exist_ok=True)
        descriptor, name = tempfile.mkstemp(
            prefix=".gemini-generation-check-", dir=output_directory
        )
        os.close(descriptor)
        Path(name).unlink(missing_ok=True)

    def generate_structured(
        self,
        prompt: PromptPackage,
        schema: Mapping[str, Any],
    ) -> StructuredGeneration:
        mode = self.settings.structured_output_mode
        strict_schema_bytes = len(dumps_json_safe(schema, indent=None).encode("utf-8"))
        if mode == "json_schema":
            adaptation = build_gemini_compatible_schema(schema)
            schema_diagnostics = adaptation.diagnostics.to_dict()
            if adaptation.diagnostics.provider_schema_validation_result != "valid":
                raise LLMProviderError(
                    "structured_output_failure",
                    "The Gemini provider schema failed local compatibility validation",
                    diagnostics=schema_diagnostics,
                )
            provider_schema: Mapping[str, Any] | None = adaptation.schema
            provider_schema_supplied = True
        elif mode == "json":
            provider_schema = None
            provider_schema_supplied = False
            schema_diagnostics = {
                "strict_schema_bytes": strict_schema_bytes,
                "provider_schema_bytes": 0,
                "removed_keyword_counts": {},
                "converted_const_count": 0,
                "resolved_reference_count": 0,
                "unresolved_reference_count": 0,
                "provider_schema_validation_result": "not_supplied",
                "adaptation_status": "not_required",
            }
        else:
            raise LLMConfigurationError(
                "structured_output_mode_invalid",
                f"Unsupported Gemini structured-output mode: {mode}",
            )

        LOGGER.info("Structured output mode: %s", mode)
        request = build_gemini_request(
            prompt,
            provider_schema,
            self.settings,
            types_module=self._active_types(),
            strict_schema=schema,
        )
        client = self._active_client()
        started = time.perf_counter()
        transport_attempt_count = 0
        retry_delays: list[float] = []
        try:
            while transport_attempt_count < self.settings.maximum_transport_attempts:
                transport_attempt_count += 1
                try:
                    response = client.models.generate_content(**request)
                    content = extract_gemini_text(response)
                    break
                except LLMConfigurationError:
                    raise
                except Exception as exc:
                    safe = (
                        exc
                        if isinstance(exc, LLMProviderError)
                        else normalize_gemini_error(exc, request_mode=mode)
                    )
                    safe.diagnostics.update(
                        {
                            "transport_attempt_count": transport_attempt_count,
                            "transient_retry_count": len(retry_delays),
                            "transport_retry_delays_seconds": list(retry_delays),
                            "final_provider_status": safe.code,
                            "tools_enabled": False,
                            "automatic_function_calling_enabled": False,
                        }
                    )
                    LOGGER.warning(
                        "Gemini request failed: code=%s http_status=%s provider_status=%s "
                        "message_category=%s attempt=%d/%d retryable=%s",
                        safe.code,
                        safe.diagnostics.get("http_status_code"),
                        safe.diagnostics.get("provider_status"),
                        safe.diagnostics.get("provider_message_category"),
                        transport_attempt_count,
                        self.settings.maximum_transport_attempts,
                        safe.retryable,
                    )
                    if (
                        not safe.retryable
                        or transport_attempt_count >= self.settings.maximum_transport_attempts
                    ):
                        safe.diagnostics["request_duration_seconds"] = round(
                            time.perf_counter() - started, 3
                        )
                        raise safe from exc
                    delay = min(
                        self.settings.retry_maximum_seconds,
                        self.settings.retry_initial_seconds
                        * (2 ** (transport_attempt_count - 1)),
                    )
                    retry_delays.append(float(delay))
                    LOGGER.warning(
                        "Retrying Gemini transport after %.1f seconds (retry %d/%d)",
                        delay,
                        len(retry_delays),
                        self.settings.maximum_transport_attempts - 1,
                    )
                    self._sleep(delay)
        finally:
            self._release_owned_client()
        duration = time.perf_counter() - started
        response_hash = hashlib.sha256(content.encode("utf-8")).hexdigest()
        return StructuredGeneration(
            content=content,
            provider=self.provider_name,
            model=str(_get(response, "model_version", self.model_name)),
            response_id=_response_identifier(response),
            status="completed",
            usage=gemini_usage(response),
            provider_metadata={
                **schema_diagnostics,
                "structured_output_mode": mode,
                "provider_schema_supplied": provider_schema_supplied,
                "strict_local_schema_validation": True,
                "schema_fallback_used": False,
                "tools_enabled": False,
                "automatic_function_calling_enabled": False,
                "search_grounding_enabled": False,
                "include_thoughts": False,
                "request_duration_seconds": round(duration, 3),
                "response_sha256": response_hash,
                "transport_attempt_count": transport_attempt_count,
                "transient_retry_count": len(retry_delays),
                "transport_retry_delays_seconds": retry_delays,
                "final_provider_status": "completed",
            },
        )

    def normalize_error(self, exc: Exception) -> LLMProviderError:
        return normalize_gemini_error(
            exc,
            request_mode=self.settings.structured_output_mode,
        )

    def extract_generation_metadata(self, response: Any) -> dict[str, Any]:
        return dict(getattr(response, "provider_metadata", {}) or {})
