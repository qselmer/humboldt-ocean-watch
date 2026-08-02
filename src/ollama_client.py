"""Official native Ollama client adapter for structured local generation."""

from __future__ import annotations

import inspect
import json
import logging
import re
from typing import Any, Mapping

from src.llm_errors import LLMConfigurationError, LLMProviderError
from src.llm_provider import LLMProviderSettings, StructuredGeneration
from src.ollama_schema_adapter import adapt_ollama_schema


REQUIRED_CHAT_PARAMETERS = {
    "model", "messages", "format", "stream", "think", "options", "keep_alive",
}

LOGGER = logging.getLogger(__name__)


def validate_installed_ollama_contract() -> str:
    """Inspect the installed official client before any local-server call."""
    try:
        import ollama
        from ollama import Client
    except ImportError as exc:
        raise LLMConfigurationError(
            "ollama_package_missing", "The official ollama Python package is not installed"
        ) from exc
    parameters = set(inspect.signature(Client.chat).parameters)
    missing = sorted(REQUIRED_CHAT_PARAMETERS - parameters)
    if missing:
        raise LLMConfigurationError(
            "ollama_package_incompatible",
            "The installed ollama Python package lacks required chat parameters: " + ", ".join(missing),
        )
    return str(getattr(ollama, "__version__", "installed"))


def create_ollama_client(
    base_url: str, *, request_timeout_seconds: float | None = None,
) -> Any:
    """Create the synchronous native client with a bounded HTTP timeout."""
    validate_installed_ollama_contract()
    from ollama import Client

    kwargs = {"timeout": request_timeout_seconds} if request_timeout_seconds else {}
    return Client(host=base_url, **kwargs)


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, Mapping):
        return value.get(name, default)
    return getattr(value, name, default)


def _sanitize_ollama_error(value: Any) -> str:
    text = " ".join(str(value or "").split())
    if not text:
        return "Ollama returned no error detail"
    if "SCIENTIFIC_CONTEXT_JSON" in text or len(text) > 1200:
        return "Ollama returned an error containing redacted request data"
    text = re.sub(
        r"(?i)\b[A-Z0-9_]*(?:API[_-]?KEY|AUTHORIZATION|BEARER|PASSWORD|TOKEN)[A-Z0-9_]*\b\s*[:=]?\s*\S+",
        "credential=[redacted]",
        text,
    )
    text = re.sub(
        r"\b[A-Z][A-Z0-9_]{2,}\s*=\s*\S+",
        "environment=[redacted]",
        text,
    )
    text = re.sub(r"(?i)[A-Z]:\\[^\s]+", "[redacted-local-path]", text)
    return text[:600]


_SCHEMA_REJECTION_MARKERS = (
    "schema",
    "format",
    "grammar",
    "structured output",
    "structured-output",
    "structured_output",
    "json_schema",
    "json-schema",
)
_NON_SCHEMA_REQUEST_MARKERS = (
    "num_ctx",
    "num_predict",
    "temperature",
    "keep_alive",
    "invalid option",
    "invalid messages",
    "missing model",
)
_OPAQUE_BAD_REQUESTS = {
    "bad request",
    "400 bad request",
    "http 400 bad request",
    "ollama returned no error detail",
}


def _schema_rejection_basis(
    *, status: Any, safe_error: str, json_schema_supplied: bool,
) -> str | None:
    """Identify schema compatibility failures without broad 400 fallback.

    Some Ollama releases return only ``Bad Request`` when the ``format``
    object cannot be compiled. That opaque response is eligible only when
    this exact request supplied a JSON Schema. Explicit non-schema request
    errors remain ineligible.
    """
    if status != 400 or not json_schema_supplied:
        return None
    text = safe_error.casefold().strip()
    if any(marker in text for marker in _NON_SCHEMA_REQUEST_MARKERS):
        return None
    if any(marker in text for marker in _SCHEMA_REJECTION_MARKERS):
        return "explicit_schema_or_format_message"
    if text in _OPAQUE_BAD_REQUESTS:
        return "opaque_http_400_with_json_schema"
    return None


def normalize_ollama_error(
    exc: Exception,
    *,
    model: str | None = None,
    request_mode: str = "unknown",
    json_schema_supplied: bool = False,
) -> LLMProviderError:
    """Map native-client failures to useful bounded diagnostics."""
    name = type(exc).__name__.casefold()
    status = getattr(exc, "status_code", None)
    error_field = getattr(exc, "error", None)
    raw_error = str(error_field or str(exc))
    safe_error = _sanitize_ollama_error(raw_error)
    text = safe_error.casefold()
    diagnostics = {
        "exception_type": type(exc).__name__,
        "http_status_code": status,
        "ollama_error": safe_error,
        "provider": "ollama",
        "model": model,
        "request_mode": request_mode,
        "json_schema_supplied": bool(json_schema_supplied),
    }
    schema_rejection_basis = _schema_rejection_basis(
        status=status,
        safe_error=raw_error,
        json_schema_supplied=json_schema_supplied,
    )
    if "timeout" in name or "timed out" in text:
        code, message, retryable = "local_timeout", "The local Ollama request timed out", False
    elif "connection" in name or status in {502, 503, 504}:
        code, message, retryable = "ollama_server_unavailable", "The local Ollama server is unavailable", True
    elif status == 404 or ("model" in text and "not found" in text):
        code, message, retryable = "model_not_installed", "The requested Ollama model is not installed", False
    elif "memory" in text or "allocate" in text:
        code, message, retryable = "model_load_failure", "Ollama could not load the model with available local memory", False
    elif "context" in text and ("length" in text or "window" in text):
        code, message, retryable = "context_length_exceeded", "The request exceeded the configured Ollama context length", False
    elif schema_rejection_basis is not None:
        code = "structured_output_failure"
        message = f"Ollama rejected the JSON Schema grammar: {safe_error}"
        retryable = False
        diagnostics["schema_or_format_error"] = True
        diagnostics["schema_rejection_basis"] = schema_rejection_basis
    elif status == 400:
        code, message, retryable = "invalid_local_request", f"Ollama rejected the local request: {safe_error}", False
        diagnostics["schema_or_format_error"] = False
    else:
        code, message, retryable = "unknown_ollama_error", f"Ollama request failed: {safe_error}", False
    diagnostics["error_code"] = code
    return LLMProviderError(code, message, retryable=retryable, diagnostics=diagnostics)


def build_json_mode_user_text(prompt: Any, strict_schema: Mapping[str, Any]) -> str:
    required_fields = [
        str(item)
        for item in strict_schema.get("required", [])
        if str(item) != "disclaimer"
    ]
    required = ", ".join(required_fields)
    structural_template = json.dumps(
        {field: None for field in required_fields},
        ensure_ascii=False,
        separators=(",", ":"),
    )
    return "\n".join([
        prompt.user_text,
        "<JSON_MODE_OUTPUT_CONTRACT>",
        "Return exactly one JSON object: no Markdown fences and no prose before or after it.",
        f"Required top-level fields: {required}.",
        f"Compact top-level structural template: {structural_template}",
        "Use the exact requested language and analysis date already supplied above.",
        "Use only permitted fact IDs. Do not generate the application-owned disclaimer field.",
        "Use null for unavailable values and perform no additional calculations.",
        "The complete response will be rejected unless it passes the authoritative local strict schema.",
        "</JSON_MODE_OUTPUT_CONTRACT>",
    ])


def build_ollama_request(
    prompt: Any,
    schema: Mapping[str, Any] | str,
    settings: LLMProviderSettings,
    *,
    request_mode: str = "json_schema",
    strict_schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a native ``/api/chat`` request; context stays only in the user role."""
    user_text = prompt.user_text
    if request_mode == "json":
        if strict_schema is None:
            raise ValueError("strict_schema is required for JSON-mode fallback")
        user_text = build_json_mode_user_text(prompt, strict_schema)
    return {
        "model": settings.model_name,
        "messages": [
            {"role": "system", "content": prompt.developer_text},
            {"role": "user", "content": user_text},
        ],
        "format": schema if isinstance(schema, str) else dict(schema),
        "stream": False,
        "think": False,
        "options": {
            "temperature": settings.temperature,
            "num_ctx": settings.num_ctx,
            "num_predict": settings.num_predict,
        },
        "keep_alive": settings.keep_alive,
    }


class OllamaProvider:
    provider_name = "ollama"
    supports_structured_output = True
    local_generation = True

    def __init__(self, settings: LLMProviderSettings, *, client: Any | None = None) -> None:
        self.settings = settings
        self.model_name = settings.model_name
        self.base_url = settings.base_url
        self._client = client
        self._owns_client = client is None

    def _active_client(self) -> Any:
        if self._client is None:
            assert self.base_url is not None
            self._client = create_ollama_client(
                self.base_url,
                request_timeout_seconds=self.settings.request_timeout_seconds,
            )
        return self._client

    def _release_owned_client(self) -> None:
        """Close only clients constructed by this provider instance."""
        if not self._owns_client or self._client is None:
            return
        client, self._client = self._client, None
        close = getattr(client, "close", None)
        if callable(close):
            try:
                close()
            except Exception as exc:  # pragma: no cover - defensive cleanup
                LOGGER.warning(
                    "Ollama client cleanup failed: %s",
                    _sanitize_ollama_error(exc),
                )

    def preflight(self, *, output_directory: Any) -> Any:
        from src.ollama_preflight import run_ollama_preflight

        client = self._active_client()
        try:
            return run_ollama_preflight(
                settings=self.settings,
                output_directory=output_directory,
                client=client,
            )
        finally:
            self._release_owned_client()

    def generate_structured(self, prompt: Any, schema: Mapping[str, Any]) -> StructuredGeneration:
        client = self._active_client()
        try:
            return self._generate_structured_with_client(client, prompt, schema)
        finally:
            self._release_owned_client()

    def _generate_structured_with_client(
        self, client: Any, prompt: Any, schema: Mapping[str, Any],
    ) -> StructuredGeneration:
        mode = self.settings.structured_output_mode
        strict_schema_bytes = len(
            json.dumps(schema, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        )
        if mode == "json":
            provider_format: Mapping[str, Any] | str = "json"
            schema_diagnostics = {
                "strict_schema_bytes": strict_schema_bytes,
                "provider_schema_bytes": 0,
                "removed_keyword_counts": {},
                "converted_const_count": 0,
                "resolved_reference_count": 0,
                "unresolved_reference_count": 0,
                "provider_schema_validation_result": "not_supplied",
            }
            json_schema_supplied = False
        elif mode == "json_schema":
            if not self.settings.attempt_provider_schema:
                raise LLMConfigurationError(
                    "provider_schema_attempt_disabled",
                    "Ollama provider-schema attempts are disabled",
                )
            adapted = adapt_ollama_schema(schema)
            schema_diagnostics = adapted.diagnostics.to_dict()
            if adapted.diagnostics.provider_schema_validation_result != "valid":
                raise LLMProviderError(
                    "provider_schema_invalid",
                    "The Ollama provider schema failed local compatibility validation",
                    diagnostics=schema_diagnostics,
                )
            provider_format = adapted.schema
            json_schema_supplied = True
        else:  # pragma: no cover - settings validation is authoritative
            raise LLMConfigurationError(
                "structured_output_mode_invalid",
                f"Unsupported Ollama structured-output mode: {mode}",
            )

        attempted_modes = [mode]
        LOGGER.info("Structured output mode: %s", mode)
        request = build_ollama_request(
            prompt,
            provider_format,
            self.settings,
            request_mode=mode,
            strict_schema=schema if mode == "json" else None,
        )
        try:
            response = client.chat(**request)
        except Exception as exc:
            if isinstance(exc, (LLMProviderError, LLMConfigurationError)):
                raise
            provider_error = normalize_ollama_error(
                exc,
                model=self.model_name,
                request_mode=mode,
                json_schema_supplied=json_schema_supplied,
            )
            LOGGER.warning(
                "Ollama request failed: mode=%s status=%s code=%s error=%s",
                provider_error.diagnostics.get("request_mode"),
                provider_error.diagnostics.get("http_status_code"),
                provider_error.code,
                provider_error.diagnostics.get("ollama_error"),
            )
            provider_error.diagnostics["attempted_modes"] = attempted_modes
            raise provider_error from exc
        message = _get(response, "message", {})
        content = _get(message, "content")
        if not isinstance(content, str) or not content.strip():
            raise LLMProviderError("malformed_local_response", "Ollama returned no final message content")
        # Deliberately never inspect or retain message.thinking.
        return StructuredGeneration(
            content=content,
            provider=self.provider_name,
            model=_get(response, "model", self.model_name),
            status="completed" if _get(response, "done", True) else "incomplete",
            usage={},
            provider_metadata={
                **self.extract_generation_metadata(response),
                **schema_diagnostics,
                "structured_output_mode": mode,
                "provider_schema_supplied": json_schema_supplied,
                "strict_local_schema_validation": True,
                "schema_fallback_used": False,
                "attempted_modes": attempted_modes,
            },
        )

    def normalize_error(self, exc: Exception) -> LLMProviderError:
        return normalize_ollama_error(exc, model=self.model_name)

    def extract_generation_metadata(self, response: Any) -> dict[str, Any]:
        fields = (
            "created_at", "done_reason", "load_duration", "total_duration",
            "prompt_eval_count", "prompt_eval_duration", "eval_count", "eval_duration",
        )
        return {field: _get(response, field) for field in fields}


def parse_minimal_json_content(response: Any) -> dict[str, Any]:
    message = _get(response, "message", {})
    content = _get(message, "content")
    if not isinstance(content, str) or not content.strip():
        raise LLMProviderError("malformed_local_response", "Ollama returned no final message content")
    try:
        parsed = json.loads(content)
    except json.JSONDecodeError as exc:
        raise LLMProviderError("malformed_local_response", "Ollama returned malformed structured JSON") from exc
    if not isinstance(parsed, dict):
        raise LLMProviderError("malformed_local_response", "Ollama structured output was not an object")
    return parsed
