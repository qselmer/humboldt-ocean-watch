"""Offline-safe, provider-neutral orchestration of scientific briefs."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import logging
import os
from pathlib import Path
import tempfile
import time
from typing import Any, Callable, Mapping

from src.brief_claim_validation import (
    ScientificBriefValidationReport,
    context_sha256,
    validate_cross_language,
    validate_scientific_brief,
)
from src.brief_generation_metadata import BriefGenerationMetadata, new_generation_metadata
from src.brief_disclaimer import (
    DISCLAIMER_SOURCE,
    DisclaimerInsertionAudit,
    MandatoryDisclaimerConfigurationError,
    get_mandatory_disclaimer,
    insert_mandatory_disclaimer,
)
from src.brief_fact_coverage import (
    FACTS_USED_SOURCE,
    FINAL_COVERAGE_CHECKLIST,
    derive_facts_used_from_claim_support,
    derive_mandatory_fact_coverage_plan,
)
from src.gemini_schema_adapter import build_gemini_compatible_schema
from src.brief_output_schema import (
    model_generated_brief_json_schema,
    scientific_brief_json_schema,
)
from src.brief_prompt_contract import (
    build_compact_response_contract,
    build_repair_contract,
)
from src.brief_prompts import (
    PromptPackage,
    build_prompt_package,
    build_repair_prompt_package,
)
from src.brief_renderer import render_scientific_brief_markdown
from src.brief_response_parser import BriefResponseError, ParsedBriefResponse, parse_structured_generation
from src.brief_validation import validate_brief_context
from src.export_utils import dumps_json_safe
from src.llm_errors import LLMConfigurationError, LLMProviderError
from src.llm_provider import (
    LLMProvider,
    LLMProviderSettings,
    get_llm_provider,
    resolve_provider_settings,
)
from src.ollama_schema_adapter import adapt_ollama_schema
from src.ollama_client import build_json_mode_user_text
from src.brief_provenance import repository_relative


LOGGER = logging.getLogger("humboldt_ocean_watch.scientific_brief")


class ScientificBriefGenerationError(RuntimeError):
    def __init__(
        self, code: str, message: str, *, details: Mapping[str, Any] | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.details = dict(details or {})


@dataclass(frozen=True)
class ValidatedContext:
    payload: dict[str, Any]
    sha256: str
    analysis_date: str
    source_path: Path
    validation_result: str


@dataclass(frozen=True)
class LanguageGeneration:
    language: str
    payload: dict[str, Any]
    markdown: str
    validation: ScientificBriefValidationReport
    metadata: BriefGenerationMetadata


@dataclass(frozen=True)
class ScientificBriefArtifacts:
    analysis_date: str
    languages: tuple[str, ...]
    paths: dict[str, Path]
    cross_language_report: dict[str, Any] | None
    generations: dict[str, LanguageGeneration]


def language_codes(language: str) -> tuple[str, ...]:
    if language == "both":
        return ("es", "en")
    if language in {"es", "en"}:
        return (language,)
    raise ValueError("Language must be 'es', 'en', or 'both'")


def load_validated_context(
    path: Path,
    *,
    settings: Mapping[str, Any],
    expected_analysis_date: str | None = None,
) -> ValidatedContext:
    if not path.exists():
        raise FileNotFoundError(f"Brief context not found: {path}")
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ScientificBriefGenerationError("invalid_context_json", "Brief context is not readable strict JSON") from exc
    if not isinstance(payload, dict):
        raise ScientificBriefGenerationError("invalid_context_root", "Brief context must be a JSON object")
    report = validate_brief_context(
        payload,
        maximum_payload_kb=int(settings.get("maximum_context_payload_kb", 150)),
        strict=True,
    )
    if report.result == "failed":
        messages = "; ".join(item["message"] for item in report.errors)
        raise ScientificBriefGenerationError("context_validation_failed", f"Brief context validation failed: {messages}")
    if payload.get("schema_version") != str(settings.get("context_schema_version", "1.0.0")):
        raise ScientificBriefGenerationError("context_schema_mismatch", "Brief context schema version is unsupported")
    analysis_date = str(payload.get("analysis", {}).get("resolved_analysis_date", ""))
    if not analysis_date:
        raise ScientificBriefGenerationError("missing_analysis_date", "Brief context has no resolved analysis date")
    if expected_analysis_date and expected_analysis_date != analysis_date:
        raise ScientificBriefGenerationError(
            "analysis_date_mismatch",
            f"Requested analysis date {expected_analysis_date} does not match context date {analysis_date}",
        )
    if not payload.get("fact_registry") or not payload.get("generation_constraints"):
        raise ScientificBriefGenerationError("incomplete_context", "Fact registry or generation constraints are missing")
    return ValidatedContext(
        payload=payload,
        sha256=context_sha256(payload),
        analysis_date=analysis_date,
        source_path=path,
        validation_result=report.result,
    )


def _request_settings(
    settings: Mapping[str, Any],
    *,
    provider: str | None,
    model: str | None,
    base_url: str | None,
) -> LLMProviderSettings:
    return resolve_provider_settings(provider, model, base_url, settings)


def _preview_path(output_directory: Path, settings: Mapping[str, Any]) -> Path:
    configured = Path(str(settings.get("debug_directory", output_directory / "debug")))
    debug = configured if configured.is_absolute() else output_directory.parents[1] / configured
    return debug / "brief_request_preview.json"


def _temporary(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.stem}-", suffix=destination.suffix, dir=destination.parent,
    )
    os.close(descriptor)
    return Path(name)


def _atomic_write_group(contents: Mapping[Path, str], *, overwrite: bool) -> None:
    destinations = list(contents)
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Output exists; use --overwrite: " + ", ".join(map(str, existing)))
    temporary = {path: _temporary(path) for path in destinations}
    backups: dict[Path, Path] = {}
    replaced: list[Path] = []
    try:
        for destination, text in contents.items():
            temporary[destination].write_text(text, encoding="utf-8")
            if destination.suffix == ".json":
                json.loads(temporary[destination].read_text(encoding="utf-8"))
        for destination in destinations:
            if destination.exists():
                backup = _temporary(destination.with_name(f".{destination.name}.backup"))
                backup.unlink(missing_ok=True)
                destination.replace(backup)
                backups[destination] = backup
            temporary[destination].replace(destination)
            replaced.append(destination)
    except BaseException:
        for destination in reversed(replaced):
            destination.unlink(missing_ok=True)
        for destination, backup in backups.items():
            if backup.exists():
                backup.replace(destination)
        raise
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)
        for path in backups.values():
            path.unlink(missing_ok=True)


def build_dry_run_preview(
    validated: ValidatedContext,
    *,
    languages: tuple[str, ...],
    settings: Mapping[str, Any],
    request_settings: LLMProviderSettings,
    root: Path,
) -> dict[str, Any]:
    prompts = {
        language: build_prompt_package(validated.payload, language, settings)
        for language in languages
    }
    strict_output_schema = scientific_brief_json_schema()
    format_object = model_generated_brief_json_schema()
    coverage_plan = derive_mandatory_fact_coverage_plan(validated.payload)
    response_contract = build_compact_response_contract(format_object)
    response_contract_bytes = len(
        dumps_json_safe(response_contract, indent=None).encode("utf-8")
    )
    if request_settings.provider_name == "gemini":
        mode = request_settings.structured_output_mode
        strict_bytes = len(dumps_json_safe(format_object, indent=None).encode("utf-8"))
        if mode == "json_schema":
            adapted = build_gemini_compatible_schema(format_object)
            provider_format = adapted.schema
            schema_diagnostics = adapted.diagnostics.to_dict()
        else:
            provider_format = "json"
            schema_diagnostics = {
                "strict_schema_bytes": strict_bytes,
                "provider_schema_bytes": 0,
                "removed_keyword_counts": {},
                "converted_const_count": 0,
                "resolved_reference_count": 0,
                "unresolved_reference_count": 0,
                "provider_schema_validation_result": "not_supplied",
                "adaptation_status": "not_required",
            }
    elif request_settings.provider_name == "ollama":
        mode = request_settings.structured_output_mode
        if mode == "json":
            provider_format: Mapping[str, Any] | str = "json"
            strict_bytes = len(dumps_json_safe(format_object, indent=None).encode("utf-8"))
            schema_diagnostics = {
                "strict_schema_bytes": strict_bytes,
                "provider_schema_bytes": 0,
                "removed_keyword_counts": {},
                "converted_const_count": 0,
                "resolved_reference_count": 0,
                "unresolved_reference_count": 0,
                "provider_schema_validation_result": "not_supplied",
            }
        else:
            adapted = adapt_ollama_schema(format_object)
            provider_format = adapted.schema
            schema_diagnostics = adapted.diagnostics.to_dict()
    else:
        mode = "json_schema"
        provider_format = format_object
        strict_bytes = len(dumps_json_safe(format_object, indent=None).encode("utf-8"))
        schema_diagnostics = {
            "strict_schema_bytes": strict_bytes,
            "provider_schema_bytes": strict_bytes,
            "removed_keyword_counts": {},
            "converted_const_count": 0,
            "resolved_reference_count": 0,
            "unresolved_reference_count": 0,
            "provider_schema_validation_result": "valid",
        }
    provider_schema_supplied = isinstance(provider_format, Mapping)
    format_bytes = (
        len(dumps_json_safe(provider_format, indent=None).encode("utf-8"))
        if provider_schema_supplied else len(str(provider_format).encode("utf-8"))
    )
    effective_prompt_bytes: dict[str, int] = {}
    request_previews: dict[str, dict[str, Any]] = {}
    for language, prompt in prompts.items():
        preview = prompt.to_preview()
        if request_settings.provider_name in {"gemini", "ollama"} and mode == "json":
            json_user_text = build_json_mode_user_text(prompt, format_object)
            effective_prompt_bytes[language] = (
                len(prompt.developer_text.encode("utf-8"))
                + len(json_user_text.encode("utf-8"))
            )
            preview["input"][1]["content"] = json_user_text
            preview["estimated_prompt_bytes"] = effective_prompt_bytes[language]
            preview["json_mode_contract_added"] = True
        else:
            effective_prompt_bytes[language] = prompt.byte_size
        request_previews[language] = preview
    request_bytes = {
        language: effective_prompt_bytes[language] + format_bytes
        for language in prompts
    }
    approximate_tokens = {
        language: request_settings.approximate_tokens(size)
        for language, size in request_bytes.items()
    }
    context_warning = bool(
        request_settings.num_ctx
        and max(approximate_tokens.values(), default=0)
        >= request_settings.num_ctx * request_settings.context_warning_fraction
    )
    return {
        "dry_run": True,
        "context_path": repository_relative(validated.source_path, root),
        "context_schema_version": validated.payload["schema_version"],
        "context_sha256": validated.sha256,
        "analysis_date": validated.analysis_date,
        "context_validation_result": validated.validation_result,
        "provider": request_settings.provider_name,
        "model": request_settings.model_name,
        "base_url": request_settings.base_url,
        "local_generation": request_settings.local_generation,
        "external_network_used": False,
        "reasoning_effort": request_settings.reasoning_effort,
        "text_verbosity": request_settings.text_verbosity,
        "think": request_settings.think,
        "temperature": request_settings.temperature,
        "num_ctx": request_settings.num_ctx,
        "configured_num_ctx": request_settings.num_ctx,
        "num_predict": request_settings.num_predict,
        "request_timeout_seconds": request_settings.request_timeout_seconds,
        "maximum_transport_attempts": request_settings.maximum_transport_attempts,
        "retry_initial_seconds": request_settings.retry_initial_seconds,
        "retry_maximum_seconds": request_settings.retry_maximum_seconds,
        "attempt_provider_schema": request_settings.attempt_provider_schema,
        "store": False,
        "tools": [],
        "tools_enabled": False,
        "automatic_function_calling_enabled": False,
        "maximum_output_tokens": request_settings.num_predict,
        "structured_output": {
            "type": mode,
            "strict": True,
            "schema": format_object,
            "provider_schema_supplied": provider_schema_supplied,
            "strict_local_schema_validation": True,
            "strict_local_schema": strict_output_schema,
        },
        "structured_output_status": (
            "json_mode_with_strict_local_validation"
            if mode == "json" else "provider_json_schema_with_strict_local_validation"
        ),
        "structured_output_mode": mode,
        "provider_schema_supplied": provider_schema_supplied,
        "strict_local_schema_validation": True,
        "compact_response_contract_included": True,
        "compact_response_contract": response_contract,
        "compact_response_contract_bytes": response_contract_bytes,
        "final_output_schema_bytes": len(
            dumps_json_safe(strict_output_schema, indent=None).encode("utf-8")
        ),
        "disclaimer_source": DISCLAIMER_SOURCE,
        "disclaimer_inserted_before_strict_validation": True,
        "disclaimer_language": languages[0] if len(languages) == 1 else "per_language",
        "model_generates_disclaimer": False,
        "facts_used_source": FACTS_USED_SOURCE,
        "mandatory_operational_coverage_plan_included": True,
        "mandatory_coverage_matrix_included": True,
        "mandatory_operational_fact_count": len(
            coverage_plan.required_facts
        ),
        "mandatory_operational_fact_ids": list(
            coverage_plan.required_fact_ids
        ),
        "mandatory_operational_coverage_plan": coverage_plan.to_dict(),
        "final_mandatory_coverage_checklist_included": True,
        "final_mandatory_coverage_checklist": list(FINAL_COVERAGE_CHECKLIST),
        "schema_fallback_used": False,
        "schema_fallback_policy": "disabled; configured mode is used directly",
        **schema_diagnostics,
        "estimated_context_bytes": len(dumps_json_safe(validated.payload, indent=None).encode("utf-8")),
        "estimated_schema_bytes": format_bytes,
        "estimated_request_bytes": request_bytes,
        "approximate_request_tokens": approximate_tokens,
        "token_estimate_is_approximate": True,
        "context_capacity_warning": context_warning,
        "requests": request_previews,
    }


def write_dry_run_preview(
    preview: dict[str, Any], path: Path, *, overwrite: bool,
) -> Path:
    _atomic_write_group({path: dumps_json_safe(preview) + "\n"}, overwrite=overwrite)
    return path


def _sum_usage(total: dict[str, int | None], usage: Mapping[str, int | None]) -> None:
    for key in total:
        value = usage.get(key)
        if value is not None:
            total[key] = int(total.get(key) or 0) + int(value)


def _estimated_generation_request(
    prompt: PromptPackage,
    schema: Mapping[str, Any],
    request_settings: LLMProviderSettings,
) -> tuple[int, int, int, int]:
    """Return prompt, provider-format, total bytes, and approximate tokens."""
    if request_settings.provider_name == "gemini":
        if request_settings.structured_output_mode == "json":
            user_text = build_json_mode_user_text(prompt, schema)
            prompt_bytes = (
                len(prompt.developer_text.encode("utf-8"))
                + len(user_text.encode("utf-8"))
            )
            provider_format_bytes = len("json".encode("utf-8"))
        else:
            prompt_bytes = prompt.byte_size
            provider_format_bytes = build_gemini_compatible_schema(
                schema
            ).diagnostics.provider_schema_bytes
    elif request_settings.provider_name == "ollama":
        if request_settings.structured_output_mode == "json":
            user_text = build_json_mode_user_text(prompt, schema)
            prompt_bytes = (
                len(prompt.developer_text.encode("utf-8"))
                + len(user_text.encode("utf-8"))
            )
            provider_format_bytes = len("json".encode("utf-8"))
        else:
            prompt_bytes = prompt.byte_size
            provider_format_bytes = adapt_ollama_schema(schema).diagnostics.provider_schema_bytes
    else:
        prompt_bytes = prompt.byte_size
        provider_format_bytes = len(dumps_json_safe(schema, indent=None).encode("utf-8"))
    request_bytes = prompt_bytes + provider_format_bytes
    return (
        prompt_bytes,
        provider_format_bytes,
        request_bytes,
        request_settings.approximate_tokens(request_bytes),
    )


def _failure_report(
    *,
    code: str,
    message: str,
    language: str,
    validated: ValidatedContext,
    model: str,
    provider: str,
    calls_made: int,
    transport_attempt_count: int = 0,
    transient_retry_count: int = 0,
    transport_retry_delays_seconds: list[float] | None = None,
    scientific_repair_count: int = 0,
    final_provider_status: str | None = None,
    validation_status: str = "not_completed",
) -> dict[str, Any]:
    return {
        "final_status": "failed",
        "error_code": code,
        "message": message,
        "language": language,
        "analysis_date": validated.analysis_date,
        "context_sha256": validated.sha256,
        "model_requested": model,
        "provider": provider,
        "calls_made": calls_made,
        "transport_attempt_count": transport_attempt_count,
        "transient_retry_count": transient_retry_count,
        "transport_retry_delays_seconds": list(
            transport_retry_delays_seconds or []
        ),
        "scientific_repair_count": scientific_repair_count,
        "final_provider_status": final_provider_status or code,
        "validation_status": validation_status,
        "facts_used_source": FACTS_USED_SOURCE,
        "store": False,
        "tools": [],
        "tools_enabled": False,
        "automatic_function_calling_enabled": False,
    }


def _write_failure(
    report: dict[str, Any], output_directory: Path, *, language: str,
) -> Path:
    debug = output_directory / "debug"
    path = debug / f"scientific_brief_{report['analysis_date']}_{language}_failure.json"
    _atomic_write_group({path: dumps_json_safe(report) + "\n"}, overwrite=True)
    return path


def _response_hash(value: Any) -> str:
    serialized = value if isinstance(value, str) else dumps_json_safe(value, indent=None)
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()


def _safe_structure_path(path: str) -> str:
    return "".join(character if character.isalnum() or character in "_.$[]-" else "_" for character in path)[:320]


def _response_structure_summary(
    value: Any,
    *,
    maximum_nodes: int = 200,
    maximum_depth: int = 6,
) -> dict[str, Any]:
    """Summarize container types and sizes without retaining response values."""
    entries: list[dict[str, Any]] = []
    truncated = False

    def visit(item: Any, path: str, depth: int) -> None:
        nonlocal truncated
        if len(entries) >= maximum_nodes:
            truncated = True
            return
        if isinstance(item, Mapping):
            entry = {"path": _safe_structure_path(path), "type": "object", "size": len(item)}
        elif isinstance(item, list):
            entry = {"path": _safe_structure_path(path), "type": "array", "size": len(item)}
        elif item is None:
            entry = {"path": _safe_structure_path(path), "type": "null"}
        elif isinstance(item, bool):
            entry = {"path": _safe_structure_path(path), "type": "boolean"}
        elif isinstance(item, str):
            entry = {"path": _safe_structure_path(path), "type": "string"}
        elif isinstance(item, (int, float)):
            entry = {"path": _safe_structure_path(path), "type": "number"}
        else:
            entry = {"path": _safe_structure_path(path), "type": type(item).__name__}
        entries.append(entry)
        if depth >= maximum_depth:
            if isinstance(item, (Mapping, list)) and item:
                truncated = True
            return
        if isinstance(item, Mapping):
            for key in sorted(item, key=lambda candidate: str(candidate)):
                visit(item[key], f"{path}.{key}", depth + 1)
                if len(entries) >= maximum_nodes:
                    break
        elif isinstance(item, list):
            for index, child in enumerate(item):
                visit(child, f"{path}[{index}]", depth + 1)
                if len(entries) >= maximum_nodes:
                    break

    visit(value, "root", 0)
    return {"entries": entries, "truncated": truncated}


def _write_validation_diagnostic(
    *,
    output_directory: Path,
    provider: str,
    model: str,
    language: str,
    analysis_date: str,
    validation_errors: list[dict[str, Any]],
    response_value: Any,
    attempt_number: int,
    repair_attempted: bool,
) -> Path | None:
    """Write a bounded diagnostic that never includes the complete response."""
    issues = build_repair_contract(validation_errors)[:100]
    for issue in issues:
        if issue.get("code") == "required_operational_fact_not_represented":
            issue["repair_instruction_generated"] = bool(repair_attempted)
    issue_codes = list(dict.fromkeys(str(item["code"]) for item in issues))
    issue_paths = list(dict.fromkeys(str(item["path"]) for item in issues))
    report = {
        "final_status": "failed_validation",
        "provider": provider,
        "model": model,
        "language": language,
        "analysis_date": analysis_date,
        "attempt_number": attempt_number,
        "validation_issue_codes": issue_codes,
        "json_paths": issue_paths,
        "validation_issues": issues,
        "facts_used_source": FACTS_USED_SOURCE,
        "response_structure_summary": _response_structure_summary(response_value),
        "response_sha256": _response_hash(response_value),
        "repair_attempted": repair_attempted,
    }
    path = (
        output_directory
        / "debug"
        / f"scientific_brief_{analysis_date}_{language}_validation_attempt_{attempt_number}.json"
    )
    try:
        _atomic_write_group({path: dumps_json_safe(report) + "\n"}, overwrite=True)
    except Exception as exc:  # diagnostics must never mask the validation result
        LOGGER.warning(
            "Sanitized validation diagnostic could not be written (%s)",
            type(exc).__name__,
        )
        return None
    return path


def _generate_language(
    *,
    provider: LLMProvider,
    validated: ValidatedContext,
    language: str,
    settings: Mapping[str, Any],
    request_settings: LLMProviderSettings,
    output_directory: Path,
    allow_repair: bool,
    sleep: Callable[[float], None],
) -> LanguageGeneration:
    maximum_calls = int(settings["maximum_generation_attempts"])
    maximum_repairs = min(1, int(request_settings.maximum_repair_attempts)) if allow_repair else 0
    calls = retries = repairs = 0
    transport_attempt_count = 0
    transient_retry_count = 0
    transport_retry_delays_seconds: list[float] = []
    final_provider_status: str | None = None
    started = time.perf_counter()
    prompt = build_prompt_package(validated.payload, language, settings)
    schema = model_generated_brief_json_schema()
    context_bytes = len(dumps_json_safe(validated.payload, indent=None).encode("utf-8"))
    current_prompt = prompt
    invalid_payload: Any = None
    validation_errors: list[dict[str, Any]] = []
    usage_total: dict[str, int | None] = {
        "input_tokens": None,
        "cached_input_tokens": None,
        "output_tokens": None,
        "reasoning_tokens": None,
        "total_tokens": None,
    }
    final_parsed: ParsedBriefResponse | None = None
    final_report: ScientificBriefValidationReport | None = None
    disclaimer_audit: DisclaimerInsertionAudit | None = None

    while calls < maximum_calls:
        calls += 1
        prompt_bytes, format_bytes, request_bytes, approximate_tokens = (
            _estimated_generation_request(current_prompt, schema, request_settings)
        )
        if request_settings.num_ctx and approximate_tokens > request_settings.num_ctx:
            raise ScientificBriefGenerationError(
                "context_length_exceeded",
                f"Approximate request size ({approximate_tokens} tokens) exceeds configured context ({request_settings.num_ctx})",
                details={"calls_made": calls - 1},
            )
        if (
            request_settings.num_ctx
            and approximate_tokens
            >= request_settings.num_ctx * request_settings.context_warning_fraction
        ):
            LOGGER.warning(
                "Approximate %s request size (%d tokens) is approaching configured context (%d); context is not truncated",
                language,
                approximate_tokens,
                request_settings.num_ctx,
            )
        LOGGER.info(
            "Prepared brief request: language=%s call=%d provider=%s model=%s mode=%s "
            "context_bytes=%d prompt_bytes=%d provider_format_bytes=%d request_bytes=%d "
            "approximate_tokens=%d configured_context=%s",
            language,
            calls,
            request_settings.provider_name,
            request_settings.model_name,
            request_settings.structured_output_mode,
            context_bytes,
            prompt_bytes,
            format_bytes,
            request_bytes,
            approximate_tokens,
            request_settings.num_ctx,
        )
        try:
            response = provider.generate_structured(current_prompt, schema)
            provider_metadata = response.provider_metadata or {}
            transport_attempt_count += int(
                provider_metadata.get("transport_attempt_count", 1)
            )
            transient_retry_count += int(
                provider_metadata.get("transient_retry_count", 0)
            )
            transport_retry_delays_seconds.extend(
                float(value)
                for value in provider_metadata.get(
                    "transport_retry_delays_seconds", []
                )
            )
            final_provider_status = str(
                provider_metadata.get("final_provider_status", response.status)
            )
            parsed = parse_structured_generation(response)
            _sum_usage(usage_total, parsed.usage)
            derive_facts_used_from_claim_support(parsed.payload)
            disclaimer_audit = insert_mandatory_disclaimer(
                parsed.payload, language
            )
            report = validate_scientific_brief(
                parsed.payload,
                validated.payload,
                language=language,
                settings=settings,
            )
            if report.final_status != "failed":
                final_parsed, final_report = parsed, report
                break
            invalid_payload = parsed.payload
            validation_errors = report.errors
            can_repair = repairs < maximum_repairs and calls < maximum_calls
            _write_validation_diagnostic(
                output_directory=output_directory,
                provider=request_settings.provider_name,
                model=request_settings.model_name,
                language=language,
                analysis_date=validated.analysis_date,
                validation_errors=validation_errors,
                response_value=invalid_payload,
                attempt_number=calls,
                repair_attempted=repairs > 0 or can_repair,
            )
            if not can_repair:
                break
            repairs += 1
            current_prompt = build_repair_prompt_package(
                validated.payload,
                language,
                settings,
                invalid_payload=invalid_payload,
                validation_errors=validation_errors,
            )
        except (LLMProviderError, LLMConfigurationError) as exc:
            if (
                request_settings.provider_name != "gemini"
                and exc.retryable
                and calls < maximum_calls
            ):
                retries += 1
                delay = min(
                    float(settings["retry_maximum_seconds"]),
                    float(settings["retry_initial_seconds"]) * (2 ** (retries - 1)),
                )
                sleep(delay)
                continue
            LOGGER.warning(
                "Scientific brief generation completed: provider=%s model=%s language=%s "
                "duration_seconds=%.3f validation=not_completed repair_count=%d",
                request_settings.provider_name,
                request_settings.model_name,
                language,
                time.perf_counter() - started,
                repairs,
            )
            diagnostics = dict(getattr(exc, "diagnostics", {}))
            failed_transport_attempts = transport_attempt_count + int(
                diagnostics.get("transport_attempt_count", 0)
            )
            failed_transient_retries = transient_retry_count + int(
                diagnostics.get("transient_retry_count", 0)
            )
            failed_delays = transport_retry_delays_seconds + [
                float(value)
                for value in diagnostics.get(
                    "transport_retry_delays_seconds", []
                )
            ]
            raise ScientificBriefGenerationError(
                exc.code,
                str(exc),
                details={
                    **diagnostics,
                    "calls_made": calls,
                    "transport_attempt_count": failed_transport_attempts,
                    "transient_retry_count": failed_transient_retries,
                    "transport_retry_delays_seconds": failed_delays,
                    "scientific_repair_count": repairs,
                    "final_provider_status": diagnostics.get(
                        "final_provider_status", exc.code
                    ),
                    "validation_status": "not_completed",
                },
            ) from exc
        except BriefResponseError as exc:
            parse_errors = [{"code": exc.code, "path": "root", "message": str(exc)}]
            raw = getattr(response, "content", None)
            can_repair = (
                exc.code != "refusal"
                and repairs < maximum_repairs
                and calls < maximum_calls
            )
            _write_validation_diagnostic(
                output_directory=output_directory,
                provider=request_settings.provider_name,
                model=request_settings.model_name,
                language=language,
                analysis_date=validated.analysis_date,
                validation_errors=parse_errors,
                response_value=raw,
                attempt_number=calls,
                repair_attempted=repairs > 0 or can_repair,
            )
            if exc.code == "refusal" or repairs >= maximum_repairs or calls >= maximum_calls:
                LOGGER.warning(
                    "Scientific brief generation completed: provider=%s model=%s language=%s "
                    "duration_seconds=%.3f validation=failed repair_count=%d",
                    request_settings.provider_name,
                    request_settings.model_name,
                    language,
                    time.perf_counter() - started,
                    repairs,
                )
                raise ScientificBriefGenerationError(
                    exc.code,
                    str(exc),
                    details={
                        "calls_made": calls,
                        "transport_attempt_count": transport_attempt_count,
                        "transient_retry_count": transient_retry_count,
                        "transport_retry_delays_seconds": transport_retry_delays_seconds,
                        "scientific_repair_count": repairs,
                        "final_provider_status": final_provider_status or "completed",
                        "validation_status": "failed",
                    },
                ) from exc
            repairs += 1
            current_prompt = build_repair_prompt_package(
                validated.payload,
                language,
                settings,
                invalid_payload={"raw_output": raw, "parse_error": exc.code},
                validation_errors=parse_errors,
            )

    if final_parsed is None or final_report is None:
        LOGGER.warning(
            "Scientific brief generation completed: provider=%s model=%s language=%s "
            "duration_seconds=%.3f validation=failed repair_count=%d",
            request_settings.provider_name,
            request_settings.model_name,
            language,
            time.perf_counter() - started,
            repairs,
        )
        message = validation_errors[0]["message"] if validation_errors else "The model output failed validation"
        raise ScientificBriefGenerationError(
            "output_validation_failed",
            message,
            details={
                "validation_errors": validation_errors,
                "calls_made": calls,
                "retry_count": retries,
                "repair_count": repairs,
                "transport_attempt_count": transport_attempt_count,
                "transient_retry_count": transient_retry_count,
                "transport_retry_delays_seconds": transport_retry_delays_seconds,
                "scientific_repair_count": repairs,
                "final_provider_status": final_provider_status or "completed",
                "validation_status": "failed",
                **(
                    disclaimer_audit.to_dict()
                    if disclaimer_audit is not None
                    else {
                        "disclaimer_source": DISCLAIMER_SOURCE,
                        "disclaimer_inserted": False,
                        "disclaimer_language": language,
                        "raw_model_disclaimer_present": False,
                        "raw_model_disclaimer_matched": False,
                    }
                ),
            },
        )
    markdown = render_scientific_brief_markdown(final_parsed.payload)
    validation_messages = [item["message"] for item in final_report.warnings]
    assert disclaimer_audit is not None
    metadata = new_generation_metadata(
        context_schema_version=str(validated.payload["schema_version"]),
        context_sha256=validated.sha256,
        prompt_version=current_prompt.prompt_version,
        prompt_sha256=current_prompt.sha256,
        provider=request_settings.provider_name,
        base_url=request_settings.base_url,
        local_generation=request_settings.local_generation,
        external_network_used=not request_settings.local_generation,
        think_enabled=request_settings.think,
        num_ctx=request_settings.num_ctx,
        num_predict=request_settings.num_predict,
        model_requested=request_settings.model_name,
        model_returned=final_parsed.model,
        language=language,
        analysis_date=validated.analysis_date,
        response_id=final_parsed.response_id,
        latency_seconds=time.perf_counter() - started,
        usage=usage_total,
        retry_count=retries + transient_retry_count,
        repair_count=repairs,
        validation_status=final_report.final_status,
        validation_messages=validation_messages,
        provider_metadata=response.provider_metadata,
        generation_call_count=calls,
        response_sha256=hashlib.sha256(response.content.encode("utf-8")).hexdigest(),
        transport_attempt_count=transport_attempt_count,
        transient_retry_count=transient_retry_count,
        transport_retry_delays_seconds=transport_retry_delays_seconds,
        final_provider_status=final_provider_status,
        disclaimer_audit=disclaimer_audit.to_dict(),
        facts_used_source=FACTS_USED_SOURCE,
    )
    LOGGER.info(
        "Scientific brief generation completed: provider=%s model=%s language=%s "
        "duration_seconds=%.3f validation=%s repair_count=%d",
        request_settings.provider_name,
        request_settings.model_name,
        language,
        metadata.latency_seconds,
        final_report.final_status,
        repairs,
    )
    return LanguageGeneration(language, final_parsed.payload, markdown, final_report, metadata)


def _output_paths(
    output_directory: Path,
    *,
    analysis_date: str,
    languages: tuple[str, ...],
    latest: bool,
) -> dict[str, Path]:
    paths: dict[str, Path] = {}
    for language in languages:
        prefix = output_directory / f"scientific_brief_{analysis_date}_{language}"
        paths[f"{language}_json"] = prefix.with_suffix(".json")
        paths[f"{language}_markdown"] = prefix.with_suffix(".md")
        paths[f"{language}_validation"] = output_directory / f"scientific_brief_{analysis_date}_{language}_validation.json"
        paths[f"{language}_generation"] = output_directory / f"scientific_brief_{analysis_date}_{language}_generation.json"
        if latest:
            paths[f"{language}_latest_json"] = output_directory / f"scientific_brief_latest_{language}.json"
            paths[f"{language}_latest_markdown"] = output_directory / f"scientific_brief_latest_{language}.md"
    if len(languages) == 2:
        paths["cross_language_validation"] = output_directory / f"scientific_brief_{analysis_date}_cross_language_validation.json"
    paths["generation_summary"] = output_directory / "scientific_brief_generation_summary.json"
    return paths


def _contents_for_publish(
    paths: Mapping[str, Path],
    generations: Mapping[str, LanguageGeneration],
    *,
    cross_report: dict[str, Any] | None,
    validated: ValidatedContext,
    request_settings: LLMProviderSettings,
) -> dict[Path, str]:
    contents: dict[Path, str] = {}
    for language, result in generations.items():
        json_text = dumps_json_safe(result.payload) + "\n"
        contents[paths[f"{language}_json"]] = json_text
        contents[paths[f"{language}_markdown"]] = result.markdown
        contents[paths[f"{language}_validation"]] = dumps_json_safe(result.validation.to_dict()) + "\n"
        contents[paths[f"{language}_generation"]] = dumps_json_safe(result.metadata.to_dict()) + "\n"
        latest_json = paths.get(f"{language}_latest_json")
        latest_markdown = paths.get(f"{language}_latest_markdown")
        if latest_json:
            contents[latest_json] = json_text
        if latest_markdown:
            contents[latest_markdown] = result.markdown
    if cross_report is not None:
        contents[paths["cross_language_validation"]] = dumps_json_safe(cross_report) + "\n"
    summary = {
        "analysis_date": validated.analysis_date,
        "languages": sorted(generations),
        "context_sha256": validated.sha256,
        "provider": request_settings.provider_name,
        "model_requested": request_settings.model_name,
        "base_url": request_settings.base_url,
        "local_generation": request_settings.local_generation,
        "reasoning_effort": request_settings.reasoning_effort,
        "text_verbosity": request_settings.text_verbosity,
        "store": False,
        "tools": [],
        "validation_status": {
            language: result.validation.final_status for language, result in generations.items()
        },
        "cross_language_status": cross_report.get("final_status") if cross_report else None,
    }
    contents[paths["generation_summary"]] = dumps_json_safe(summary) + "\n"
    return contents


def generate_scientific_briefs(
    *,
    context_path: Path,
    config: Mapping[str, Any],
    root: Path,
    language: str = "both",
    analysis_date: str | None = None,
    provider_name: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    reasoning_effort: str | None = None,
    text_verbosity: str | None = None,
    output_directory: Path | None = None,
    overwrite: bool = False,
    allow_repair: bool = True,
    client: Any | None = None,
    llm_provider: LLMProvider | None = None,
    sleep: Callable[[float], None] = time.sleep,
) -> ScientificBriefArtifacts:
    settings = config["brief_generation"]
    if not settings.get("enabled", False):
        raise ScientificBriefGenerationError("generation_disabled", "Scientific brief generation is disabled")
    languages = language_codes(language)
    try:
        for language_code in languages:
            get_mandatory_disclaimer(language_code)
    except MandatoryDisclaimerConfigurationError as exc:
        raise ScientificBriefGenerationError(exc.code, str(exc)) from exc
    validated = load_validated_context(
        context_path,
        settings=settings,
        expected_analysis_date=analysis_date,
    )
    request_settings = _request_settings(
        settings,
        provider=provider_name or getattr(llm_provider, "provider_name", None),
        model=model or getattr(llm_provider, "model_name", None),
        base_url=base_url,
    )
    directory = output_directory or (root / str(settings["output_directory"]))
    if not directory.is_absolute():
        directory = root / directory
    latest = validated.analysis_date == validated.payload.get("analysis", {}).get("available_end_date")
    paths = _output_paths(directory, analysis_date=validated.analysis_date, languages=languages, latest=latest)
    existing = [path for path in paths.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Valid brief output may already exist; use --overwrite: " + ", ".join(map(str, existing)))

    active_provider = llm_provider or get_llm_provider(
        request_settings.provider_name,
        request_settings.model_name,
        settings,
        base_url=request_settings.base_url,
        client=client,
    )
    if llm_provider is None and client is None:
        local_validator = getattr(active_provider, "validate_local_environment", None)
        if callable(local_validator):
            try:
                local_validator(output_directory=directory)
            except (LLMConfigurationError, OSError) as exc:
                error_code = getattr(exc, "code", "generation_environment_invalid")
                message = (
                    str(exc)
                    if isinstance(exc, LLMConfigurationError)
                    else "The configured brief output directory is not writable"
                )
                raise ScientificBriefGenerationError(
                    error_code,
                    message,
                ) from exc
        LOGGER.info(
            "Normal generation uses local readiness checks only; run "
            "scripts/check_llm_access.py for the explicit full provider preflight"
        )

    generations: dict[str, LanguageGeneration] = {}
    for language_code in languages:
        try:
            generations[language_code] = _generate_language(
                provider=active_provider,
                validated=validated,
                language=language_code,
                settings=settings,
                request_settings=request_settings,
                output_directory=directory,
                allow_repair=allow_repair,
                sleep=sleep,
            )
        except ScientificBriefGenerationError as exc:
            failure = _failure_report(
                code=exc.code,
                message=str(exc),
                language=language_code,
                validated=validated,
                model=request_settings.model_name,
                provider=request_settings.provider_name,
                calls_made=int(exc.details.get("calls_made", settings["maximum_generation_attempts"])),
            )
            failure.update(exc.details)
            _write_failure(failure, directory, language=language_code)
            raise

    cross_report = None
    if len(languages) == 2 and settings.get("require_cross_language_consistency", True):
        cross_report = validate_cross_language(
            generations["es"].payload,
            generations["en"].payload,
        )
        if cross_report["final_status"] == "failed":
            failure = _failure_report(
                code="cross_language_validation_failed",
                message="Spanish and English briefs are scientifically inconsistent",
                language="both",
                validated=validated,
                model=request_settings.model_name,
                provider=request_settings.provider_name,
                calls_made=sum(result.metadata.retry_count + result.metadata.repair_count + 1 for result in generations.values()),
            )
            failure["validation_errors"] = cross_report["errors"]
            _write_failure(failure, directory, language="both")
            raise ScientificBriefGenerationError(
                "cross_language_validation_failed",
                "Spanish and English briefs are scientifically inconsistent",
            )

    contents = _contents_for_publish(
        paths,
        generations,
        cross_report=cross_report,
        validated=validated,
        request_settings=request_settings,
    )
    try:
        _atomic_write_group(contents, overwrite=overwrite)
    except OSError as exc:
        failure = _failure_report(
            code="output_write_failed",
            message="Validated brief outputs could not be written atomically",
            language="both" if len(languages) == 2 else languages[0],
            validated=validated,
            model=request_settings.model_name,
            provider=request_settings.provider_name,
            calls_made=sum(
                result.metadata.retry_count + result.metadata.repair_count + 1
                for result in generations.values()
            ),
        )
        try:
            _write_failure(failure, directory, language=failure["language"])
        except OSError:
            LOGGER.error("The sanitized output-write failure report could not be saved")
        raise ScientificBriefGenerationError(
            "output_write_failed", "Validated brief outputs could not be written atomically"
        ) from exc
    LOGGER.info(
        "Published validated scientific brief for %s (%s)",
        validated.analysis_date,
        ", ".join(languages),
    )
    return ScientificBriefArtifacts(validated.analysis_date, languages, paths, cross_report, generations)


def dry_run_scientific_brief(
    *,
    context_path: Path,
    config: Mapping[str, Any],
    root: Path,
    language: str,
    analysis_date: str | None,
    provider_name: str | None = None,
    model: str | None = None,
    base_url: str | None = None,
    reasoning_effort: str | None = None,
    text_verbosity: str | None = None,
    output_directory: Path | None,
    overwrite: bool,
) -> tuple[dict[str, Any], Path]:
    settings = config["brief_generation"]
    languages = language_codes(language)
    try:
        for language_code in languages:
            get_mandatory_disclaimer(language_code)
    except MandatoryDisclaimerConfigurationError as exc:
        raise ScientificBriefGenerationError(exc.code, str(exc)) from exc
    validated = load_validated_context(context_path, settings=settings, expected_analysis_date=analysis_date)
    request_settings = _request_settings(
        settings,
        provider=provider_name,
        model=model,
        base_url=base_url,
    )
    directory = output_directory or (root / str(settings["output_directory"]))
    if not directory.is_absolute():
        directory = root / directory
    preview = build_dry_run_preview(
        validated,
        languages=languages,
        settings=settings,
        request_settings=request_settings,
        root=root,
    )
    configured_debug = Path(str(settings["debug_directory"]))
    debug = configured_debug if configured_debug.is_absolute() else root / configured_debug
    path = debug / "brief_request_preview.json"
    try:
        # This sanitized diagnostic is not a validated final brief and is safe to refresh.
        write_dry_run_preview(preview, path, overwrite=True)
    except OSError as exc:
        raise ScientificBriefGenerationError(
            "output_write_failed", "The sanitized dry-run preview could not be written"
        ) from exc
    return preview, path
