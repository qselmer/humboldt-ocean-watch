"""Provider-neutral, sanitized LLM errors."""

from __future__ import annotations

from typing import Any, Mapping


GEMINI_ERROR_CODES = frozenset(
    {
        "gemini_package_missing",
        "missing_api_key",
        "invalid_api_key",
        "inaccessible_model",
        "quota_exceeded",
        "rate_limit",
        "provider_internal_error",
        "service_unavailable",
        "request_timeout",
        "network_failure",
        "structured_output_failure",
        "malformed_structured_output",
        "provider_refusal",
        "output_token_limit",
        "unknown_gemini_error",
    }
)


class LLMConfigurationError(RuntimeError):
    """A local configuration or installed-client incompatibility."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = False


class LLMProviderError(RuntimeError):
    """A sanitized provider request failure."""

    def __init__(
        self,
        code: str,
        message: str,
        *,
        retryable: bool = False,
        diagnostics: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.retryable = retryable
        self.diagnostics = dict(diagnostics or {})
