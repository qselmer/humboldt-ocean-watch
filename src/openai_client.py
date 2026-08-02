"""Official OpenAI Responses API client with sanitized bounded failures."""

from __future__ import annotations

from dataclasses import dataclass
import inspect
import os
from typing import Any, Mapping

from src.brief_output_schema import responses_text_format
from src.brief_prompts import PromptPackage
from src.llm_provider import LLMProviderSettings, StructuredGeneration
from src.llm_errors import LLMProviderError


DEFAULT_OPENAI_MODEL = "gpt-5.6"


class OpenAIConfigurationError(RuntimeError):
    pass


class OpenAIRequestError(LLMProviderError):
    def __init__(self, code: str, message: str, *, retryable: bool = False) -> None:
        super().__init__(code, message, retryable=retryable)


@dataclass(frozen=True)
class OpenAIRequestSettings:
    model: str
    reasoning_effort: str
    text_verbosity: str
    maximum_output_tokens: int


def api_key_from_environment() -> str:
    """Read the only permitted credential source without logging its value."""
    value = os.getenv("OPENAI_API_KEY")
    if value is None:
        raise OpenAIConfigurationError("OPENAI_API_KEY is not set")
    if not value.strip():
        raise OpenAIConfigurationError("OPENAI_API_KEY is empty")
    return value.strip()


def resolve_model(configured: str | None, explicit: str | None = None) -> str:
    selected = explicit or os.getenv("OPENAI_MODEL") or configured or DEFAULT_OPENAI_MODEL
    if not str(selected).strip():
        raise OpenAIConfigurationError("The selected OpenAI model name is empty")
    return str(selected).strip()


def validate_installed_sdk_contract() -> str:
    """Validate the installed SDK surface before any network call."""
    try:
        import openai
        from openai import OpenAI
    except ImportError as exc:
        raise OpenAIConfigurationError("The openai Python package is not installed") from exc
    client = OpenAI(api_key="sdk-contract-check", max_retries=0)
    required = {"model", "input", "reasoning", "text", "store", "tools", "max_output_tokens"}
    parameters = set(inspect.signature(client.responses.create).parameters)
    missing = sorted(required - parameters)
    if missing:
        raise OpenAIConfigurationError(
            "The installed OpenAI SDK is incompatible with the required Responses API fields: "
            + ", ".join(missing)
        )
    if not callable(getattr(client.models, "retrieve", None)):
        raise OpenAIConfigurationError("The installed OpenAI SDK does not support model retrieval")
    return str(openai.__version__)


def create_openai_client(*, api_key: str | None = None) -> Any:
    validate_installed_sdk_contract()
    from openai import OpenAI

    return OpenAI(api_key=api_key or api_key_from_environment(), max_retries=0)


def build_responses_request(
    prompt: PromptPackage,
    request_settings: OpenAIRequestSettings,
    schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build the exact openai 2.46.0 request shape used by this project."""
    return {
        "model": request_settings.model,
        "input": prompt.as_responses_input(),
        "reasoning": {"effort": request_settings.reasoning_effort},
        "text": {
            "verbosity": request_settings.text_verbosity,
            "format": responses_text_format(schema),
        },
        "max_output_tokens": int(request_settings.maximum_output_tokens),
        "store": False,
        "tools": [],
    }


def classify_openai_error(exc: Exception) -> OpenAIRequestError:
    """Map SDK/network failures to fixed messages that cannot leak secrets."""
    class_name = type(exc).__name__.casefold()
    status = getattr(exc, "status_code", None)
    code = str(getattr(exc, "code", "") or "").casefold()
    body = getattr(exc, "body", None)
    if isinstance(body, Mapping):
        nested = body.get("error", body)
        if isinstance(nested, Mapping):
            code = str(nested.get("code", code) or code).casefold()

    if "authentication" in class_name or status == 401:
        return OpenAIRequestError("invalid_api_key", "OpenAI rejected the API credentials")
    if "permission" in class_name or status in {403, 404}:
        return OpenAIRequestError("inaccessible_model", "The selected model is not accessible to this API project")
    if "insufficient_quota" in code or "quota" in code:
        return OpenAIRequestError("insufficient_quota", "The OpenAI project has insufficient quota")
    if "ratelimit" in class_name or status == 429:
        return OpenAIRequestError("rate_limit", "The OpenAI rate limit was reached", retryable=True)
    if "timeout" in class_name:
        return OpenAIRequestError("request_timeout", "The OpenAI request timed out", retryable=True)
    if "connection" in class_name or "network" in class_name:
        return OpenAIRequestError("network_failure", "The OpenAI API could not be reached", retryable=True)
    if "context_length" in code:
        return OpenAIRequestError("context_length", "The request exceeded the model context limit")
    if status == 400 or "badrequest" in class_name:
        return OpenAIRequestError("invalid_request", "OpenAI rejected the request as invalid")
    return OpenAIRequestError("unknown_api_error", "An unknown OpenAI API error occurred")


def create_response(client: Any, request: dict[str, Any]) -> Any:
    try:
        return client.responses.create(**request)
    except Exception as exc:
        raise classify_openai_error(exc) from exc


class OpenAIProvider:
    """Provider-neutral adapter over the existing Responses API client."""

    provider_name = "openai"
    supports_structured_output = True
    local_generation = False

    def __init__(self, settings: LLMProviderSettings, *, client: Any | None = None) -> None:
        self.settings = settings
        self.model_name = settings.model_name
        self.base_url = None
        self._client = client

    def _request_settings(self) -> OpenAIRequestSettings:
        return OpenAIRequestSettings(
            model=self.model_name,
            reasoning_effort=self.settings.reasoning_effort or "medium",
            text_verbosity=self.settings.text_verbosity or "medium",
            maximum_output_tokens=self.settings.num_predict,
        )

    def preflight(self, *, output_directory: Any) -> Any:
        from src.openai_preflight import run_openai_preflight

        return run_openai_preflight(
            configured_model=self.model_name,
            model_override=self.model_name,
            output_directory=output_directory,
            client=self._client,
        )

    def generate_structured(self, prompt: PromptPackage, schema: Mapping[str, Any]) -> StructuredGeneration:
        # The provider-facing schema excludes application-owned metadata.  The
        # final assembled payload is validated against the complete local schema.
        response = create_response(
            self._client or create_openai_client(),
            build_responses_request(prompt, self._request_settings(), schema),
        )
        # Preserve refusal/incomplete handling from the established OpenAI parser.
        from src.brief_response_parser import parse_brief_response
        from src.export_utils import dumps_json_safe

        parsed = parse_brief_response(response)
        return StructuredGeneration(
            content=dumps_json_safe(parsed.payload, indent=None),
            provider=self.provider_name,
            model=parsed.model,
            response_id=parsed.response_id,
            status=parsed.status,
            usage=parsed.usage,
            provider_metadata={
                "structured_output_mode": "json_schema",
                "provider_schema_supplied": True,
                "strict_local_schema_validation": True,
                "schema_fallback_used": False,
            },
        )

    def normalize_error(self, exc: Exception) -> OpenAIRequestError:
        return classify_openai_error(exc)

    def extract_generation_metadata(self, response: Any) -> dict[str, Any]:
        return dict(getattr(response, "provider_metadata", {}) or {})
