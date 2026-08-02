"""Deterministic adaptation of the strict brief schema for Google GenAI."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from typing import Any, Mapping

from jsonschema import Draft202012Validator, SchemaError

from src.export_utils import to_json_compatible


ANNOTATION_KEYWORDS = frozenset(
    {
        "$schema",
        "examples",
        "default",
        "readOnly",
        "writeOnly",
        "deprecated",
    }
)
SUPPORTED_STRUCTURAL_KEYWORDS = frozenset(
    {
        "type",
        "properties",
        "required",
        "items",
        "enum",
        "minimum",
        "maximum",
        "minItems",
        "maxItems",
        "additionalProperties",
        "$id",
        "$anchor",
        "anyOf",
        "oneOf",
        "format",
        "title",
        "description",
    }
)


def _json_bytes(value: Mapping[str, Any]) -> int:
    return len(
        json.dumps(
            value,
            ensure_ascii=False,
            sort_keys=True,
            separators=(",", ":"),
            allow_nan=False,
        ).encode("utf-8")
    )


@dataclass
class GeminiSchemaDiagnostics:
    strict_schema_bytes: int
    provider_schema_bytes: int = 0
    removed_keyword_counts: dict[str, int] = field(default_factory=dict)
    converted_const_count: int = 0
    resolved_reference_count: int = 0
    unresolved_references: list[str] = field(default_factory=list)
    unsupported_keyword_counts: dict[str, int] = field(default_factory=dict)
    additional_properties_preserved_count: int = 0
    provider_schema_validation_result: str = "not_validated"
    adaptation_status: str = "not_adapted"
    validation_errors: list[str] = field(default_factory=list)

    @property
    def unresolved_reference_count(self) -> int:
        return len(self.unresolved_references)

    def to_dict(self) -> dict[str, Any]:
        data = asdict(self)
        data["unresolved_reference_count"] = self.unresolved_reference_count
        converted = to_json_compatible(data)
        assert isinstance(converted, dict)
        return converted


@dataclass(frozen=True)
class GeminiSchemaAdaptationResult:
    schema: dict[str, Any]
    diagnostics: GeminiSchemaDiagnostics


def _resolve_pointer(root: Mapping[str, Any], reference: str) -> Any:
    if not reference.startswith("#/"):
        raise KeyError(reference)
    current: Any = root
    for raw_token in reference[2:].split("/"):
        token = raw_token.replace("~1", "/").replace("~0", "~")
        if not isinstance(current, Mapping):
            raise KeyError(reference)
        current = current[token]
    return current


def _resolve_local_references(
    value: Any,
    *,
    root: Mapping[str, Any],
    diagnostics: GeminiSchemaDiagnostics,
    stack: tuple[str, ...] = (),
) -> Any:
    if isinstance(value, list):
        return [
            _resolve_local_references(
                item,
                root=root,
                diagnostics=diagnostics,
                stack=stack,
            )
            for item in value
        ]
    if not isinstance(value, dict):
        return deepcopy(value)
    if "$ref" in value:
        reference = str(value["$ref"])
        if not reference.startswith("#/") or reference in stack:
            diagnostics.unresolved_references.append(reference)
        else:
            try:
                target = deepcopy(_resolve_pointer(root, reference))
            except (KeyError, TypeError):
                diagnostics.unresolved_references.append(reference)
            else:
                diagnostics.resolved_reference_count += 1
                siblings = {key: item for key, item in value.items() if key != "$ref"}
                if isinstance(target, dict):
                    target.update(siblings)
                return _resolve_local_references(
                    target,
                    root=root,
                    diagnostics=diagnostics,
                    stack=stack + (reference,),
                )
    return {
        key: _resolve_local_references(
            item,
            root=root,
            diagnostics=diagnostics,
            stack=stack,
        )
        for key, item in value.items()
    }


def _adapt_keywords(value: Any, diagnostics: GeminiSchemaDiagnostics) -> Any:
    if isinstance(value, list):
        return [_adapt_keywords(item, diagnostics) for item in value]
    if not isinstance(value, dict):
        return value

    adapted: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"properties", "$defs", "definitions"} and isinstance(item, dict):
            adapted[key] = {
                property_name: _adapt_keywords(property_schema, diagnostics)
                for property_name, property_schema in item.items()
            }
            continue
        if key in ANNOTATION_KEYWORDS:
            diagnostics.removed_keyword_counts[key] = (
                diagnostics.removed_keyword_counts.get(key, 0) + 1
            )
            continue
        if key == "const":
            diagnostics.converted_const_count += 1
            adapted["enum"] = [_adapt_keywords(item, diagnostics)]
            continue
        if key == "additionalProperties":
            # response_json_schema accepts JSON Schema dictionaries. This
            # structural constraint is retained so provider and local schemas
            # both reject unexpected fields.
            diagnostics.additional_properties_preserved_count += 1
        elif key not in SUPPORTED_STRUCTURAL_KEYWORDS:
            diagnostics.unsupported_keyword_counts[key] = (
                diagnostics.unsupported_keyword_counts.get(key, 0) + 1
            )
        adapted[key] = _adapt_keywords(item, diagnostics)
    return adapted


def build_gemini_compatible_schema(
    strict_schema: Mapping[str, Any],
) -> GeminiSchemaAdaptationResult:
    """Build a native ``response_json_schema`` without mutating the source.

    Local references are inlined, annotations are removed, ``const`` becomes a
    one-item ``enum``, and structural constraints—including
    ``additionalProperties``—are retained. Unknown structural keywords are
    reported and preserved rather than silently weakened.
    """
    strict_copy = deepcopy(dict(strict_schema))
    diagnostics = GeminiSchemaDiagnostics(strict_schema_bytes=_json_bytes(strict_copy))
    resolved = _resolve_local_references(
        strict_copy,
        root=strict_copy,
        diagnostics=diagnostics,
    )
    assert isinstance(resolved, dict)
    if not diagnostics.unresolved_references:
        for keyword in ("$defs", "definitions"):
            if keyword in resolved:
                diagnostics.removed_keyword_counts[keyword] = (
                    diagnostics.removed_keyword_counts.get(keyword, 0) + 1
                )
                resolved.pop(keyword)
    adapted = _adapt_keywords(resolved, diagnostics)
    assert isinstance(adapted, dict)
    diagnostics.provider_schema_bytes = _json_bytes(adapted)

    if diagnostics.unresolved_references:
        diagnostics.provider_schema_validation_result = "unresolved_references"
        diagnostics.adaptation_status = "failed"
        diagnostics.validation_errors.append("Provider schema contains unresolved references")
    else:
        try:
            Draft202012Validator.check_schema(adapted)
            json.dumps(adapted, ensure_ascii=False, allow_nan=False)
        except (SchemaError, TypeError, ValueError) as exc:
            diagnostics.provider_schema_validation_result = "invalid"
            diagnostics.adaptation_status = "failed"
            diagnostics.validation_errors.append(type(exc).__name__)
        else:
            diagnostics.provider_schema_validation_result = "valid"
            diagnostics.adaptation_status = (
                "adapted_with_preserved_unsupported_constraints"
                if diagnostics.unsupported_keyword_counts
                else "adapted"
            )
    return GeminiSchemaAdaptationResult(adapted, diagnostics)
