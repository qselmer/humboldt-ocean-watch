"""Strict structural and scientific validation for brief-context JSON."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import date
import math
from pathlib import Path
import re
from typing import Any

from src.brief_fact_registry import validate_repository_relative_path
from src.brief_schema import (
    CONTEXT_STATUSES,
    CONTROLLED_UNITS,
    FLAG_SEVERITIES,
    REPRESENTATIVENESS_CLASSES,
    SCHEMA_VERSION,
    missing_top_level_sections,
)
from src.export_utils import dumps_json_safe, to_json_compatible


_SECRET_PATTERN = re.compile(r"(credential|password|api[_-]?key|secret|token)", re.IGNORECASE)
_ENVIRONMENT_PATH_PATTERN = re.compile(r"(?:\$\{|\$[A-Za-z_]|%[A-Za-z_][A-Za-z0-9_]*%|^~[/\\])")


@dataclass(frozen=True)
class ValidationIssue:
    code: str
    section: str
    message: str


@dataclass
class BriefValidationReport:
    result: str
    strict: bool
    schema_version: str | None
    payload_bytes: int
    maximum_payload_bytes: int
    errors: list[dict[str, str]] = field(default_factory=list)
    warnings: list[dict[str, str]] = field(default_factory=list)
    checks: dict[str, bool] = field(default_factory=dict)
    statistics: dict[str, int] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        converted = to_json_compatible(asdict(self))
        assert isinstance(converted, dict)
        return converted


def _iso_date(value: Any) -> bool:
    if not isinstance(value, str):
        return False
    try:
        date.fromisoformat(value[:10])
    except ValueError:
        return False
    return len(value) >= 10


def _walk(value: Any, path: str = "root") -> list[tuple[str, Any]]:
    items = [(path, value)]
    if isinstance(value, dict):
        for key, item in value.items():
            items.extend(_walk(item, f"{path}.{key}"))
    elif isinstance(value, list):
        for index, item in enumerate(value):
            items.extend(_walk(item, f"{path}[{index}]"))
    return items


def validate_brief_context(
    context: dict[str, Any],
    *,
    maximum_payload_kb: int = 150,
    strict: bool = True,
) -> BriefValidationReport:
    """Return a complete report; callers decide whether to raise or exit."""
    errors: list[ValidationIssue] = []
    warnings: list[ValidationIssue] = []

    def error(code: str, section: str, message: str) -> None:
        errors.append(ValidationIssue(code, section, message))

    def warning(code: str, section: str, message: str) -> None:
        warnings.append(ValidationIssue(code, section, message))

    missing = missing_top_level_sections(context)
    if missing:
        error("missing_top_level_sections", "root", f"Missing sections: {missing}")
    if context.get("schema_version") != SCHEMA_VERSION:
        error("invalid_schema_version", "root", f"Expected schema version {SCHEMA_VERSION}")

    for key in (
        "product_name", "product_short_name", "product_version", "product_status",
        "experimental_product", "region_name", "region_code", "spatial_bounds",
        "source_dataset", "source_variable", "institution", "scientific_scope",
    ):
        if key not in context.get("product", {}):
            error("missing_product_metadata", "product", f"Missing product.{key}")
    for key in (
        "requested_analysis_date", "resolved_analysis_date", "generated_at",
        "available_start_date", "available_end_date", "data_latency_days",
        "temporal_resolution", "spatial_resolution", "data_mode", "analysis_status",
    ):
        if key not in context.get("analysis", {}):
            error("missing_analysis_metadata", "analysis", f"Missing analysis.{key}")
    for key in ("resolved_analysis_date", "available_start_date", "available_end_date"):
        if key in context.get("analysis", {}) and not _iso_date(context["analysis"][key]):
            error("invalid_iso_date", "analysis", f"analysis.{key} is not ISO-8601")
    analysis = context.get("analysis", {})
    if all(_iso_date(analysis.get(key)) for key in ("available_start_date", "resolved_analysis_date", "available_end_date")):
        if not analysis["available_start_date"] <= analysis["resolved_analysis_date"] <= analysis["available_end_date"]:
            error("analysis_date_outside_period", "analysis", "Resolved date is outside the available period")
    if analysis.get("analysis_status") not in CONTEXT_STATUSES:
        error("unsupported_status", "analysis", "analysis_status is unsupported")

    for path, value in _walk(context):
        if isinstance(value, float) and not math.isfinite(value):
            error("nonfinite_number", path, "NaN and Infinity are forbidden")
        if isinstance(value, dict) and "status" in value and value["status"] not in CONTEXT_STATUSES:
            error("unsupported_status", path, f"Unsupported status {value['status']!r}")

    facts = context.get("fact_registry", [])
    fact_ids: list[str] = []
    for index, fact in enumerate(facts if isinstance(facts, list) else []):
        location = f"fact_registry[{index}]"
        if not isinstance(fact, dict):
            error("invalid_fact", location, "Fact must be an object")
            continue
        fact_id = fact.get("fact_id")
        if not isinstance(fact_id, str):
            error("invalid_fact_id", location, "fact_id must be a string")
        else:
            fact_ids.append(fact_id)
        if fact.get("unit") not in CONTROLLED_UNITS:
            error("invalid_unit", location, f"Unsupported unit {fact.get('unit')!r}")
        if fact.get("status") not in CONTEXT_STATUSES:
            error("unsupported_status", location, f"Unsupported fact status {fact.get('status')!r}")
        if fact.get("value") is None and fact.get("status") == "valid":
            error("null_valid_fact", location, "Unavailable facts cannot be valid")
        try:
            validate_repository_relative_path(str(fact.get("source_path", "")))
        except ValueError as exc:
            error("invalid_source_path", location, str(exc))
        if _ENVIRONMENT_PATH_PATTERN.search(str(fact.get("source_path", ""))):
            error("environment_path", location, "Environment-variable paths are forbidden")
        if fact.get("allowed_in_brief") and not fact.get("source_id"):
            error("missing_fact_provenance", location, "Allowed facts require source_id")
        unit = fact.get("unit")
        value = fact.get("value")
        name = str(fact.get("name", ""))
        if value is not None and unit == "fraction" and not 0 <= float(value) <= 1:
            error("invalid_fraction", location, f"Fraction outside [0, 1]: {value}")
        if value is not None and unit in {"degree_latitude", "degree_longitude"}:
            low, high = (-90, 90) if unit == "degree_latitude" else (-180, 360)
            if not low <= float(value) <= high:
                error("invalid_coordinate", location, f"Coordinate outside [{low}, {high}]")
        if value is not None and (unit in {"km2", "count", "day", "km2_day"} or "duration" in name):
            if float(value) < 0:
                error("negative_value", location, f"{name} cannot be negative")
    if len(fact_ids) != len(set(fact_ids)):
        error("duplicate_fact_id", "fact_registry", "fact_id values must be unique")

    registry_ids = set(fact_ids)
    registry_by_id = {
        str(fact["fact_id"]): fact
        for fact in facts if isinstance(fact, dict) and isinstance(fact.get("fact_id"), str)
    }
    selected_date = context.get("analysis", {}).get("resolved_analysis_date")
    for fact in facts if isinstance(facts, list) else []:
        if isinstance(fact, dict) and fact.get("selected_date") != selected_date:
            error("selected_date_mismatch", "fact_registry", f"Fact {fact.get('fact_id')} uses a different selected date")
    for path, value in _walk(context):
        if isinstance(value, dict) and {"value", "unit", "status", "fact_id"} <= set(value):
            if value["fact_id"] not in registry_ids:
                error("unknown_fact_reference", path, f"Unknown fact_id {value['fact_id']!r}")
            else:
                registered = registry_by_id[str(value["fact_id"])]
                for attribute in ("value", "unit", "status"):
                    if value.get(attribute) != registered.get(attribute):
                        error(
                            "fact_reference_mismatch",
                            path,
                            f"Compact reference {attribute} disagrees with fact_registry",
                        )

    for section, records_key in (
        ("daily_patches", "top_patches"),
        ("spatiotemporal_activity", "top_tracks"),
        ("spatiotemporal_activity", "top_event_families"),
    ):
        records = context.get(section, {}).get(records_key, [])
        for index, record in enumerate(records if isinstance(records, list) else []):
            fact_map = record.get("fact_ids", {}) if isinstance(record, dict) else {}
            for name, fact_id in fact_map.items():
                location = f"{section}.{records_key}[{index}].{name}"
                if fact_id not in registry_by_id:
                    error("unknown_fact_reference", location, f"Unknown fact_id {fact_id!r}")
                    continue
                if record.get(name) != registry_by_id[fact_id].get("value"):
                    error("fact_reference_mismatch", location, "Record value disagrees with fact_registry")

    regional = context.get("regional_state", {})
    p10 = regional.get("spatial_p10_anomaly_c", {}).get("value") if isinstance(regional.get("spatial_p10_anomaly_c"), dict) else None
    p90 = regional.get("spatial_p90_anomaly_c", {}).get("value") if isinstance(regional.get("spatial_p90_anomaly_c"), dict) else None
    if p10 is not None and p90 is not None and p10 > p90:
        error("invalid_percentile_order", "regional_state", "P10 cannot exceed P90")
    evidence = context.get("representativeness", {}).get("evidence_score", {})
    evidence_value = evidence.get("value") if isinstance(evidence, dict) else evidence
    if evidence_value is not None and not 0 <= float(evidence_value) <= 1:
        error("invalid_evidence_score", "representativeness", "evidence_score must be within [0, 1]")
    class_code = context.get("representativeness", {}).get("class_code")
    if class_code not in REPRESENTATIVENESS_CLASSES:
        error("invalid_representativeness_class", "representativeness", f"Unsupported class {class_code!r}")

    event = context.get("univariate_event", {})
    if event.get("active"):
        start = event.get("event_start_date")
        end = event.get("event_end_date")
        if not (_iso_date(start) and _iso_date(end) and _iso_date(selected_date)):
            error("invalid_event_dates", "univariate_event", "Active event dates must use ISO-8601")
        else:
            if start > selected_date:
                error("event_starts_after_analysis", "univariate_event", "Event start is after selected date")
            if end < selected_date:
                error("event_ends_before_analysis", "univariate_event", "Active event ends before selected date")

    def ref_value(section: str, name: str) -> Any:
        value = context.get(section, {}).get(name)
        return value.get("value") if isinstance(value, dict) else value

    patch_count = ref_value("daily_patches", "patch_count")
    observation_count = ref_value("spatiotemporal_activity", "active_patch_observation_count")
    if (
        context.get("daily_patches", {}).get("product_available")
        and context.get("spatiotemporal_activity", {}).get("product_available")
        and patch_count is not None
        and observation_count is not None
        and int(patch_count) != int(observation_count)
    ):
        error("patch_observation_count_mismatch", "spatiotemporal_activity", "Daily patch and tracked observation counts disagree")
    active_track_count = ref_value("spatiotemporal_activity", "active_track_count")
    top_tracks = context.get("spatiotemporal_activity", {}).get("top_tracks", [])
    if active_track_count is not None and int(active_track_count) < len(top_tracks):
        error("active_track_count_mismatch", "spatiotemporal_activity", "Top-track list exceeds active-track count")
    active_family_count = ref_value("spatiotemporal_activity", "active_family_count")
    top_families = context.get("spatiotemporal_activity", {}).get("top_event_families", [])
    if active_family_count is not None and int(active_family_count) < len(top_families):
        error("active_family_count_mismatch", "spatiotemporal_activity", "Top-family list exceeds active-family count")

    provenance = context.get("provenance", {}).get("sources", [])
    for index, source in enumerate(provenance if isinstance(provenance, list) else []):
        try:
            validate_repository_relative_path(str(source.get("path", "")))
        except ValueError as exc:
            error("invalid_provenance_path", f"provenance.sources[{index}]", str(exc))
        if _ENVIRONMENT_PATH_PATTERN.search(str(source.get("path", ""))):
            error("environment_path", f"provenance.sources[{index}]", "Environment-variable paths are forbidden")
        if any(_SECRET_PATTERN.search(str(key)) for key in source):
            error("credential_like_field", f"provenance.sources[{index}]", "Credential-like fields are forbidden")

    expected_availability = {
        "regional_metrics", "quality_control", "climatology", "representativeness",
        "temporal_features", "spatial_features", "univariate_events", "daily_patches",
        "tracks", "event_families", "lineage_edges",
    }
    availability = context.get("availability", {})
    for key in expected_availability:
        state = availability.get(key)
        if not isinstance(state, dict) or not {"available", "reason", "expected_path"} <= set(state):
            error("invalid_availability", "availability", f"Availability state {key} is incomplete")
        elif not isinstance(state["available"], bool):
            error("invalid_availability", "availability", f"Availability state {key} is not Boolean")
        else:
            try:
                validate_repository_relative_path(str(state["expected_path"]))
            except ValueError as exc:
                error("invalid_availability_path", "availability", str(exc))

    climatology_file = context.get("climatology", {}).get("climatology_file")
    if climatology_file is not None:
        try:
            validate_repository_relative_path(str(climatology_file))
        except ValueError as exc:
            error("invalid_climatology_path", "climatology", str(exc))

    for index, flag in enumerate(context.get("quality_flags", [])):
        required = {"flag_id", "severity", "section", "message_code", "message_es", "message_en", "affected_fact_ids", "suppress_dependent_claims"}
        if not isinstance(flag, dict) or not required <= set(flag):
            error("invalid_quality_flag", f"quality_flags[{index}]", "Quality flag schema is incomplete")
        elif flag["severity"] not in FLAG_SEVERITIES:
            error("invalid_flag_severity", f"quality_flags[{index}]", "Unsupported flag severity")
        elif flag["severity"] == "error":
            error(
                "error_quality_flag",
                str(flag.get("section", "quality_flags")),
                f"Context contains error quality flag {flag.get('flag_id')}",
            )
        if isinstance(flag, dict):
            unknown = sorted(set(flag.get("affected_fact_ids", [])) - registry_ids)
            if unknown:
                error("unknown_affected_fact", f"quality_flags[{index}]", f"Unknown affected fact IDs: {unknown}")
    for index, limitation in enumerate(context.get("limitations", [])):
        required = {"limitation_id", "category", "statement_es", "statement_en", "affected_sections", "severity"}
        if not isinstance(limitation, dict) or not required <= set(limitation):
            error("invalid_limitation", f"limitations[{index}]", "Limitation schema is incomplete")
        elif limitation["severity"] not in FLAG_SEVERITIES:
            error("invalid_limitation_severity", f"limitations[{index}]", "Unsupported limitation severity")

    constraints = context.get("generation_constraints", {})
    prohibited = " ".join(map(str, constraints.get("prohibited_claims", []))).lower()
    for phrase in ("official coastal el niño", "forecasts", "invented numerical", "individual water"):
        if phrase not in prohibited:
            error("missing_prohibited_claim", "generation_constraints", f"Missing prohibition covering {phrase!r}")
    if "valid fact_id" not in str(constraints.get("numerical_fact_policy", "")):
        error("missing_numerical_fact_policy", "generation_constraints", "Numerical fact policy is incomplete")
    if "must not be inferred" not in str(constraints.get("missing_data_policy", "")):
        error("missing_data_policy", "generation_constraints", "Missing-data policy is incomplete")
    disclaimer_present = any(
        "official" in str(item.get("statement_en", "")).lower()
        and "coastal el niño" in str(item.get("statement_en", "")).lower()
        for item in context.get("limitations", []) if isinstance(item, dict)
    )
    if not disclaimer_present:
        error("missing_official_disclaimer", "limitations", "Official-classification disclaimer is required")

    payload_bytes = len(dumps_json_safe(context).encode("utf-8"))
    maximum_payload_bytes = int(maximum_payload_kb) * 1024
    if payload_bytes > maximum_payload_bytes:
        error("oversized_payload", "root", f"Payload {payload_bytes} exceeds {maximum_payload_bytes} bytes")
    if context.get("spatiotemporal_activity", {}).get("lineage_edges"):
        error("lineage_catalogue_in_payload", "spatiotemporal_activity", "Complete lineage edges are forbidden")

    required_availability = context.get("availability", {})
    for key in ("regional_metrics", "quality_control", "representativeness"):
        state = required_availability.get(key, {})
        if not isinstance(state, dict) or not state.get("available"):
            error("core_input_unavailable", "availability", f"Core input {key} is unavailable")

    checks = {
        "structure": not any(issue.section in {"root", "product", "analysis"} for issue in errors),
        "numbers": not any(issue.code in {"nonfinite_number", "invalid_fraction", "invalid_coordinate", "negative_value", "invalid_percentile_order"} for issue in errors),
        "provenance": not any("provenance" in issue.code or "source_path" in issue.code for issue in errors),
        "generation_safety": not any(issue.section in {"generation_constraints", "limitations"} for issue in errors),
        "payload": not any(issue.code == "oversized_payload" for issue in errors),
    }
    has_flag_warning = any(
        isinstance(flag, dict) and flag.get("severity") in {"warning", "error"}
        for flag in context.get("quality_flags", [])
    )
    result = "failed" if errors else ("warning" if warnings or has_flag_warning else "passed")
    return BriefValidationReport(
        result=result,
        strict=strict,
        schema_version=context.get("schema_version"),
        payload_bytes=payload_bytes,
        maximum_payload_bytes=maximum_payload_bytes,
        errors=[asdict(issue) for issue in errors],
        warnings=[asdict(issue) for issue in warnings],
        checks=checks,
        statistics={
            "fact_count": len(facts) if isinstance(facts, list) else 0,
            "valid_fact_count": sum(1 for fact in facts if isinstance(fact, dict) and fact.get("status") == "valid"),
            "unavailable_fact_count": sum(1 for fact in facts if isinstance(fact, dict) and fact.get("value") is None),
            "quality_flag_count": len(context.get("quality_flags", [])),
        },
    )
