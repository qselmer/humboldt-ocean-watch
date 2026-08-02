"""Minimal OpenAI access preflight without scientific generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
import os
from pathlib import Path
import tempfile
from typing import Any

from src.export_utils import to_json_compatible
from src.openai_client import (
    OpenAIConfigurationError,
    api_key_from_environment,
    classify_openai_error,
    create_openai_client,
    resolve_model,
    validate_installed_sdk_contract,
)


@dataclass
class OpenAIPreflightReport:
    status: str
    sdk_version: str | None
    api_key_status: str
    selected_model: str | None
    model_accessible: bool
    responses_api_available: bool
    output_directory_writable: bool
    messages: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        converted = to_json_compatible(asdict(self))
        assert isinstance(converted, dict)
        return converted


def _check_writable(directory: Path) -> None:
    directory.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=".openai-preflight-", dir=directory)
    os.close(descriptor)
    Path(name).unlink(missing_ok=True)


def run_openai_preflight(
    *,
    configured_model: str,
    output_directory: Path,
    model_override: str | None = None,
    client: Any | None = None,
) -> OpenAIPreflightReport:
    report = OpenAIPreflightReport(
        status="failed",
        sdk_version=None,
        api_key_status="missing",
        selected_model=None,
        model_accessible=False,
        responses_api_available=False,
        output_directory_writable=False,
    )
    try:
        report.sdk_version = validate_installed_sdk_contract()
        key = api_key_from_environment()
        report.api_key_status = "present"
        report.selected_model = resolve_model(configured_model, model_override)
        active_client = client or create_openai_client(api_key=key)
        report.responses_api_available = callable(
            getattr(getattr(active_client, "responses", None), "create", None)
        )
        if not report.responses_api_available:
            raise OpenAIConfigurationError("The Responses API cannot be instantiated")
        active_client.models.retrieve(report.selected_model)
        report.model_accessible = True
        _check_writable(output_directory)
        report.output_directory_writable = True
        report.status = "passed"
        report.messages.append("OpenAI SDK, model access, Responses API, and output path are available")
    except OpenAIConfigurationError as exc:
        report.messages.append(str(exc))
    except OSError:
        report.messages.append("The configured brief output directory is not writable")
    except Exception as exc:
        safe = classify_openai_error(exc)
        report.messages.append(str(safe))
    return report
