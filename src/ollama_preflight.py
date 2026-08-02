"""Independent offline-local preflight checks for Ollama."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import tempfile
from typing import Any, Callable

from src.export_utils import to_json_compatible
from src.llm_provider import LLMProviderSettings
from src.ollama_client import (
    create_ollama_client,
    normalize_ollama_error,
    parse_minimal_json_content,
    validate_installed_ollama_contract,
)


PREFLIGHT_SCHEMA = {
    "type": "object",
    "properties": {"status": {"type": "string", "const": "OK"}},
    "required": ["status"],
    "additionalProperties": False,
}


@dataclass
class OllamaPreflightReport:
    provider: str
    model: str
    base_url: str | None
    package_available: bool = False
    package_version: str | None = None
    server_reachable: bool = False
    model_list_retrievable: bool = False
    model_installed: bool = False
    basic_generation_available: bool = False
    structured_generation_available: bool = False
    thinking_excluded: bool = False
    output_directory_writable: bool = False
    configured_context_length: int | None = None
    status: str = "failed"
    errors: list[dict[str, str]] = field(default_factory=list)
    messages: list[str] = field(default_factory=list)

    def add_error(self, code: str, message: str) -> None:
        self.errors.append({"code": code, "message": message})

    def to_dict(self) -> dict[str, Any]:
        converted = to_json_compatible(asdict(self))
        assert isinstance(converted, dict)
        return converted


def _check_writable(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".ollama-preflight-", dir=directory)
    os.close(descriptor)
    Path(name).unlink(missing_ok=True)


def _model_names(response: Any) -> set[str]:
    models = response.get("models", []) if isinstance(response, dict) else getattr(response, "models", [])
    names: set[str] = set()
    for model in models:
        value = model.get("model") or model.get("name") if isinstance(model, dict) else getattr(model, "model", None) or getattr(model, "name", None)
        if value:
            names.add(str(value))
    return names


def run_ollama_preflight(
    *,
    settings: LLMProviderSettings,
    output_directory: Path,
    client: Any | None = None,
    package_validator: Callable[[], str] = validate_installed_ollama_contract,
) -> OllamaPreflightReport:
    """Run every safe check independently; never pull a model automatically."""
    report = OllamaPreflightReport(
        provider="ollama", model=settings.model_name, base_url=settings.base_url,
        configured_context_length=settings.num_ctx,
    )
    try:
        report.package_version = package_validator()
        report.package_available = True
    except Exception:
        report.add_error("ollama_package_missing", "The official ollama Python package is unavailable or incompatible")
    if not settings.base_url:
        report.add_error("ollama_server_unavailable", "The Ollama base URL is not configured")
    if not settings.num_ctx or settings.num_ctx <= 0:
        report.add_error("context_configuration_invalid", "The configured Ollama context length is invalid")
    try:
        _check_writable(output_directory)
        report.output_directory_writable = True
    except OSError:
        report.add_error("output_directory_unwritable", "The configured brief output directory is not writable")

    active = client
    owns_active = False
    if active is None and report.package_available and settings.base_url:
        try:
            active = create_ollama_client(
                settings.base_url,
                request_timeout_seconds=settings.request_timeout_seconds,
            )
            owns_active = True
        except Exception:
            report.add_error("ollama_server_unavailable", "The Ollama client could not be initialized")
    if active is not None:
        try:
            listed = active.list()
            report.server_reachable = True
            report.model_list_retrievable = True
            report.model_installed = settings.model_name in _model_names(listed)
            if not report.model_installed:
                report.add_error("model_not_installed", f"Requested model is not installed. Run: ollama pull {settings.model_name}")
        except Exception as exc:
            safe = normalize_ollama_error(exc)
            report.add_error(safe.code, str(safe))

        basic_request = {
            "model": settings.model_name,
            "messages": [{"role": "user", "content": "Reply only with OK."}],
            "stream": False,
            "think": False,
            "options": {"temperature": 0, "num_ctx": min(settings.num_ctx or 2048, 2048), "num_predict": 8},
            "keep_alive": settings.keep_alive,
        }
        try:
            basic = active.chat(**basic_request)
            message = basic.get("message", {}) if isinstance(basic, dict) else getattr(basic, "message", {})
            content = message.get("content") if isinstance(message, dict) else getattr(message, "content", None)
            report.basic_generation_available = isinstance(content, str) and bool(content.strip())
            report.thinking_excluded = not bool(message.get("thinking") if isinstance(message, dict) else getattr(message, "thinking", None))
            if not report.basic_generation_available:
                report.add_error("malformed_local_response", "Ollama basic generation returned no content")
        except Exception as exc:
            safe = normalize_ollama_error(exc)
            report.add_error(safe.code, str(safe))
        try:
            preflight_format: str | dict[str, Any] = (
                "json"
                if settings.structured_output_mode == "json"
                else PREFLIGHT_SCHEMA
            )
            structured = active.chat(**{
                **basic_request,
                "messages": [{"role": "user", "content": "Return exactly one JSON object with status OK."}],
                "format": preflight_format,
                "options": {**basic_request["options"], "num_predict": 32},
            })
            parsed = parse_minimal_json_content(structured)
            report.structured_generation_available = parsed == {"status": "OK"}
            if not report.structured_generation_available:
                report.add_error("structured_output_failure", "Ollama did not satisfy the minimal structured-output schema")
        except Exception as exc:
            safe = exc if hasattr(exc, "code") else normalize_ollama_error(exc)
            report.add_error(getattr(safe, "code", "structured_output_failure"), str(safe))

    if owns_active and active is not None:
        close = getattr(active, "close", None)
        if callable(close):
            try:
                close()
            except Exception:
                report.add_error(
                    "ollama_client_cleanup_failed",
                    "The internally created Ollama preflight client could not be closed",
                )

    report.status = "passed" if all((
        report.package_available, report.server_reachable, report.model_list_retrievable,
        report.model_installed, report.basic_generation_available,
        report.structured_generation_available, report.thinking_excluded,
        report.output_directory_writable, bool(settings.num_ctx and settings.num_ctx > 0),
    )) else "failed"
    if report.status == "passed":
        report.messages.append("Ollama package, local server, model, structured generation, and output path are available")
    return report
