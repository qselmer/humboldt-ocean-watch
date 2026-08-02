"""Independent minimal preflight for the native Google GenAI provider."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

from src.export_utils import to_json_compatible
from src.gemini_client import (
    InjectedClientTypes,
    create_gemini_client,
    extract_gemini_text,
    gemini_api_key_from_environment,
    load_gemini_sdk,
    normalize_gemini_error,
    validate_installed_gemini_contract,
)
from src.llm_errors import LLMConfigurationError
from src.llm_provider import LLMProviderSettings


PREFLIGHT_SCHEMA = {
    "type": "object",
    "properties": {"status": {"type": "string", "enum": ["OK"]}},
    "required": ["status"],
    "additionalProperties": False,
}


@dataclass
class GeminiPreflightReport:
    provider: str
    model: str
    package_available: bool = False
    package_version: str | None = None
    api_key_available: bool = False
    api_reachable: bool = False
    model_accessible: bool = False
    basic_generation_available: bool = False
    structured_generation_available: bool = False
    output_directory_writable: bool = False
    status: str = "failed"
    errors: list[dict[str, str]] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    def add_error(self, code: str, message: str) -> None:
        if code not in {item["code"] for item in self.errors}:
            self.errors.append({"code": code, "message": message})

    def to_dict(self) -> dict[str, Any]:
        converted = to_json_compatible(asdict(self))
        assert isinstance(converted, dict)
        return converted


def _check_writable(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".gemini-preflight-", dir=directory)
    os.close(descriptor)
    Path(name).unlink(missing_ok=True)


def _minimal_config(
    types_module: Any,
    *,
    maximum_output_tokens: int,
    schema: dict[str, Any] | None = None,
) -> Any:
    kwargs: dict[str, Any] = {
        "automatic_function_calling": types_module.AutomaticFunctionCallingConfig(
            disable=True
        ),
        "temperature": 0,
        "max_output_tokens": maximum_output_tokens,
        "response_mime_type": "application/json" if schema is not None else "text/plain",
    }
    if schema is not None:
        kwargs["response_json_schema"] = schema
    thinking_type = getattr(types_module, "ThinkingConfig", None)
    if callable(thinking_type):
        kwargs["thinking_config"] = thinking_type(include_thoughts=False)
    return types_module.GenerateContentConfig(**kwargs)


def run_gemini_preflight(
    *,
    settings: LLMProviderSettings,
    output_directory: Path,
    client: Any | None = None,
    types_module: Any | None = None,
    package_validator: Callable[[], str] = validate_installed_gemini_contract,
) -> GeminiPreflightReport:
    """Run small independent checks without sending scientific context."""
    report = GeminiPreflightReport(provider="gemini", model=settings.model_name)
    try:
        report.package_version = package_validator()
        report.package_available = True
    except Exception:
        report.add_error(
            "gemini_package_missing",
            "The google-genai Python package is unavailable or incompatible",
        )

    variable = settings.api_key_environment_variable or "GEMINI_API_KEY"
    key: str | None = None
    try:
        key = gemini_api_key_from_environment(variable)
        report.api_key_available = True
    except LLMConfigurationError as exc:
        report.add_error(exc.code, str(exc))

    try:
        _check_writable(output_directory)
        report.output_directory_writable = True
    except OSError:
        report.add_error(
            "output_directory_unwritable",
            "The configured brief output directory is not writable",
        )

    active = client
    owns_active = False
    active_types = types_module
    if report.package_available and active_types is None:
        try:
            active_types = load_gemini_sdk().types
        except LLMConfigurationError as exc:
            if active is not None:
                active_types = InjectedClientTypes
            else:
                report.add_error(exc.code, str(exc))
    if active is None and report.package_available and key:
        try:
            active = create_gemini_client(
                api_key=key,
                request_timeout_seconds=settings.request_timeout_seconds,
            )
            owns_active = True
        except Exception as exc:
            safe = (
                exc
                if isinstance(exc, LLMConfigurationError)
                else normalize_gemini_error(exc)
            )
            report.add_error(getattr(safe, "code", "network_failure"), str(safe))

    if active is not None and active_types is not None and report.api_key_available:
        try:
            active.models.get(model=settings.model_name)
            report.api_reachable = True
            report.model_accessible = True
        except Exception as exc:
            safe = normalize_gemini_error(exc)
            report.add_error(safe.code, str(safe))

        try:
            basic = active.models.generate_content(
                model=settings.model_name,
                contents="Reply only with OK.",
                config=_minimal_config(active_types, maximum_output_tokens=8),
            )
            report.api_reachable = True
            report.basic_generation_available = extract_gemini_text(basic).strip() == "OK"
            if not report.basic_generation_available:
                report.add_error(
                    "malformed_structured_output",
                    "Gemini basic generation did not return the expected minimal response",
                )
        except Exception as exc:
            safe = exc if hasattr(exc, "code") else normalize_gemini_error(exc)
            report.add_error(getattr(safe, "code", "unknown_gemini_error"), str(safe))

        try:
            structured = active.models.generate_content(
                model=settings.model_name,
                contents="Return exactly one JSON object with status OK.",
                config=_minimal_config(
                    active_types,
                    maximum_output_tokens=32,
                    schema=PREFLIGHT_SCHEMA,
                ),
            )
            parsed = json.loads(extract_gemini_text(structured))
            report.api_reachable = True
            report.structured_generation_available = parsed == {"status": "OK"}
            if not report.structured_generation_available:
                report.add_error(
                    "structured_output_failure",
                    "Gemini did not satisfy the minimal structured-output schema",
                )
        except Exception as exc:
            safe = exc if hasattr(exc, "code") else normalize_gemini_error(exc)
            report.add_error(getattr(safe, "code", "structured_output_failure"), str(safe))

    if owns_active and active is not None:
        close = getattr(active, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                report.add_error(
                    "gemini_client_cleanup_failed",
                    "The internally created Gemini client could not be closed",
                )

    report.status = "passed" if all(
        (
            report.package_available,
            report.api_key_available,
            report.api_reachable,
            report.model_accessible,
            report.basic_generation_available,
            report.structured_generation_available,
            report.output_directory_writable,
        )
    ) else "failed"
    if report.status == "passed":
        report.messages.append(
            "Google GenAI package, API, model, structured generation, and output path are available"
        )
    return report
