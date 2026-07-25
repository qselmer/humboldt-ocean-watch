"""Strict structured-output contract for bilingual scientific briefs."""

from __future__ import annotations

from collections.abc import Mapping
from copy import deepcopy
from typing import Any

from jsonschema import Draft202012Validator


BRIEF_OUTPUT_SCHEMA_VERSION = "1.0.0"
NARRATIVE_SECTION_IDS = (
    "executive_summary",
    "regional_state",
    "recent_evolution",
    "spatial_structure",
    "event_status",
    "data_quality",
)
SECTION_STATUSES = ("available", "limited", "unavailable")
CLAIM_TYPES = (
    "observation",
    "comparison",
    "recent_change",
    "spatial_interpretation",
    "event_description",
    "quality_caveat",
    "limitation",
)


def _string_array(*, maximum: int, minimum: int = 0) -> dict[str, Any]:
    return {
        "type": "array",
        "items": {"type": "string", "maxLength": 160},
        "minItems": minimum,
        "maxItems": maximum,
    }


def _paragraph_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "paragraph_id",
            "text",
            "supporting_fact_ids",
            "quality_flag_ids",
            "claim_types",
        ],
        "properties": {
            "paragraph_id": {"type": "string", "maxLength": 80},
            "text": {"type": "string", "maxLength": 3000},
            "supporting_fact_ids": _string_array(maximum=40),
            "quality_flag_ids": _string_array(maximum=20),
            "claim_types": {
                "type": "array",
                "items": {"type": "string", "enum": list(CLAIM_TYPES)},
                "minItems": 1,
                "maxItems": len(CLAIM_TYPES),
            },
        },
    }


def _section_schema(*, maximum_paragraphs: int = 5) -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": ["heading", "paragraphs", "status", "reason"],
        "properties": {
            "heading": {"type": "string", "maxLength": 180},
            "paragraphs": {
                "type": "array",
                "items": {"$ref": "#/$defs/paragraph"},
                "maxItems": maximum_paragraphs,
            },
            "status": {"type": "string", "enum": list(SECTION_STATUSES)},
            "reason": {"type": ["string", "null"], "maxLength": 500},
        },
    }


def _limitation_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "limitation_id",
            "text",
            "supporting_fact_ids",
            "quality_flag_ids",
        ],
        "properties": {
            "limitation_id": {"type": "string", "maxLength": 100},
            "text": {"type": "string", "maxLength": 1200},
            "supporting_fact_ids": _string_array(maximum=20),
            "quality_flag_ids": _string_array(maximum=20),
        },
    }


def _key_message_schema() -> dict[str, Any]:
    return {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "message_id",
            "text",
            "supporting_fact_ids",
            "quality_flag_ids",
            "claim_types",
        ],
        "properties": {
            "message_id": {"type": "string", "maxLength": 80},
            "text": {"type": "string", "maxLength": 1000},
            "supporting_fact_ids": _string_array(maximum=30),
            "quality_flag_ids": _string_array(maximum=20),
            "claim_types": {
                "type": "array",
                "items": {"type": "string", "enum": list(CLAIM_TYPES)},
                "minItems": 1,
                "maxItems": len(CLAIM_TYPES),
            },
        },
    }


def scientific_brief_json_schema() -> dict[str, Any]:
    """Return the API and local-validation JSON Schema.

    All object properties are required and all objects reject extra fields so
    the same schema can be used as a strict Responses API ``text.format``.
    """
    properties: dict[str, Any] = {
        "schema_version": {"type": "string", "const": BRIEF_OUTPUT_SCHEMA_VERSION},
        "language": {"type": "string", "enum": ["es", "en"]},
        "analysis_date": {"type": "string", "maxLength": 10},
        "title": {"type": "string", "maxLength": 300},
    }
    for section_id in NARRATIVE_SECTION_IDS:
        properties[section_id] = _section_schema(
            maximum_paragraphs=3 if section_id == "executive_summary" else 5
        )
    properties.update(
        {
            "limitations": {
                "type": "array",
                "items": {"$ref": "#/$defs/limitation"},
                "minItems": 1,
                "maxItems": 12,
            },
            "key_messages": {
                "type": "array",
                "items": {"$ref": "#/$defs/key_message"},
                "maxItems": 5,
            },
            "disclaimer": {"type": "string", "maxLength": 500},
            "facts_used": _string_array(maximum=500),
            "generation_notes": _string_array(maximum=12),
        }
    )
    return {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "type": "object",
        "additionalProperties": False,
        "required": list(properties),
        "properties": properties,
        "$defs": {
            "paragraph": _paragraph_schema(),
            "limitation": _limitation_schema(),
            "key_message": _key_message_schema(),
        },
    }


def model_generated_brief_json_schema() -> dict[str, Any]:
    """Return the provider-facing schema without application-owned metadata."""
    schema = scientific_brief_json_schema()
    schema["properties"].pop("disclaimer", None)
    schema["required"] = [
        field for field in schema["required"] if field != "disclaimer"
    ]
    return schema


def responses_text_format(
    schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the exact strict format object accepted by openai 2.46.0."""
    schema = deepcopy(schema if schema is not None else scientific_brief_json_schema())
    schema.pop("$schema", None)
    return {
        "type": "json_schema",
        "name": "humboldt_ocean_watch_scientific_brief",
        "description": "A fact-grounded scientific SST monitoring brief.",
        "strict": True,
        "schema": schema,
    }


def _json_type_name(value: Any) -> str:
    if value is None:
        return "null"
    if isinstance(value, bool):
        return "boolean"
    if isinstance(value, Mapping):
        return "object"
    if isinstance(value, list):
        return "array"
    if isinstance(value, str):
        return "string"
    if isinstance(value, int):
        return "integer"
    if isinstance(value, float):
        return "number"
    return type(value).__name__


def _expected_type_text(value: Any) -> str:
    if isinstance(value, list):
        return " or ".join(str(item) for item in value)
    return str(value)


def _resolve_schema_fragment(fragment: Mapping[str, Any], root: Mapping[str, Any]) -> Mapping[str, Any]:
    node: Mapping[str, Any] = fragment
    visited: set[str] = set()
    while "$ref" in node:
        ref = str(node["$ref"])
        if ref in visited or not ref.startswith("#/"):
            break
        visited.add(ref)
        candidate: Any = root
        for raw_part in ref[2:].split("/"):
            part = raw_part.replace("~1", "/").replace("~0", "~")
            if not isinstance(candidate, Mapping) or part not in candidate:
                return node
            candidate = candidate[part]
        if not isinstance(candidate, Mapping):
            return node
        node = candidate
    return node


def validate_output_schema(payload: Any) -> list[dict[str, Any]]:
    """Return deterministic, value-free JSON-Schema diagnostics without raising."""
    schema = scientific_brief_json_schema()
    validator = Draft202012Validator(schema)
    issues: list[dict[str, Any]] = []
    errors = sorted(
        validator.iter_errors(payload),
        key=lambda item: (
            tuple(str(part) for part in item.absolute_path),
            str(item.validator),
            str(item.validator_value),
        ),
    )
    for error in errors:
        parts = list(error.absolute_path)
        path = ".".join(str(part) for part in parts) or "root"
        validator_name = str(error.validator)
        code, message = _schema_issue_details(
            validator_name=validator_name,
            validator_value=error.validator_value,
            path_parts=parts,
        )
        base: dict[str, Any] = {"code": code, "path": path, "message": message}
        if validator_name == "additionalProperties":
            allowed = sorted(str(key) for key in error.schema.get("properties", {}))
            actual = error.instance if isinstance(error.instance, Mapping) else {}
            unexpected = sorted(str(key) for key in actual if str(key) not in allowed)
            for field in unexpected or ["unknown"]:
                issues.append({
                    **base,
                    "unexpected_field": field,
                    "allowed_fields": allowed,
                })
            continue
        if validator_name == "required":
            required = [str(item) for item in error.validator_value]
            actual = error.instance if isinstance(error.instance, Mapping) else {}
            missing = [field for field in required if field not in actual]
            properties = error.schema.get("properties", {})
            for field in missing or ["unknown"]:
                property_schema = properties.get(field, {}) if isinstance(properties, Mapping) else {}
                if isinstance(property_schema, Mapping):
                    property_schema = _resolve_schema_fragment(property_schema, schema)
                expected = property_schema.get("type", "defined field") if isinstance(property_schema, Mapping) else "defined field"
                issues.append({
                    **base,
                    "missing_field": field,
                    "expected_type": _expected_type_text(expected),
                })
            continue
        if validator_name == "type":
            base["expected_type"] = _expected_type_text(error.validator_value)
            base["actual_type"] = _json_type_name(error.instance)
        elif validator_name == "enum":
            base["allowed_values"] = deepcopy(error.validator_value)
        elif validator_name == "const":
            base["expected_constant"] = deepcopy(error.validator_value)
        elif validator_name in {"minItems", "maxItems", "minLength", "maxLength"}:
            base["constraint"] = validator_name
            base["constraint_value"] = error.validator_value
        issues.append(base)

    unique: list[dict[str, Any]] = []
    signatures: set[str] = set()
    for issue in issues:
        signature = repr(sorted(issue.items(), key=lambda item: item[0]))
        if signature not in signatures:
            signatures.add(signature)
            unique.append(issue)
    return unique


def _schema_issue_details(
    *,
    validator_name: str,
    validator_value: Any,
    path_parts: list[Any],
) -> tuple[str, str]:
    """Map jsonschema errors to stable codes without echoing model content."""
    if validator_name == "type":
        last = path_parts[-1] if path_parts else None
        if len(path_parts) == 1 and last in NARRATIVE_SECTION_IDS:
            return "invalid_section_type", "A narrative section must be an object"
        if len(path_parts) >= 2 and path_parts[-2] == "paragraphs" and isinstance(last, int):
            return "invalid_paragraph_type", "A paragraph entry must be an object"
        if last == "paragraphs":
            return "invalid_paragraphs_type", "Section paragraphs must be an array"
        if last in {"heading", "text", "reason"}:
            return "invalid_text_field_type", "The text field has an invalid type"
        expected_text = _expected_type_text(validator_value)
        return "invalid_field_type", f"The field must have type {expected_text}"
    if validator_name == "required":
        return "missing_required_field", "A required field is missing"
    if validator_name == "additionalProperties":
        return "additional_property_not_allowed", "Unexpected fields are not allowed"
    if validator_name == "enum":
        return "invalid_enum_value", "The field contains a value outside the allowed set"
    if validator_name == "const":
        return "invalid_constant_value", "The field does not match the required constant"
    return "schema_constraint_error", f"The field violates the {validator_name} schema constraint"


def schema_copy() -> dict[str, Any]:
    """Return an isolated copy for callers and tests."""
    return deepcopy(scientific_brief_json_schema())
