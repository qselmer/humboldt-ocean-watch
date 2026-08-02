"""Deterministic adaptation of the strict brief schema for Ollama grammar."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import asdict, dataclass, field
import json
from typing import Any, Mapping

from jsonschema import Draft202012Validator, SchemaError

from src.export_utils import to_json_compatible


ANNOTATION_KEYWORDS = frozenset({
    "$schema", "$id", "title", "description", "examples", "default",
    "readOnly", "writeOnly", "deprecated",
})
COMPOSITION_KEYWORDS = frozenset({"anyOf", "oneOf", "allOf"})


@dataclass
class OllamaSchemaDiagnostics:
    strict_schema_bytes: int
    provider_schema_bytes: int = 0
    removed_keyword_counts: dict[str, int] = field(default_factory=dict)
    converted_const_count: int = 0
    resolved_reference_count: int = 0
    unresolved_references: list[str] = field(default_factory=list)
    composition_keyword_counts: dict[str, int] = field(default_factory=dict)
    nullable_type_array_count: int = 0
    provider_schema_validation_result: str = "not_validated"
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
class AdaptedOllamaSchema:
    schema: dict[str, Any]
    diagnostics: OllamaSchemaDiagnostics


def _json_bytes(value: Mapping[str, Any]) -> int:
    return len(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8"))


def _resolve_pointer(root: Mapping[str, Any], reference: str) -> Any:
    if not reference.startswith("#/"):
        raise KeyError(reference)
    current: Any = root
    for raw in reference[2:].split("/"):
        token = raw.replace("~1", "/").replace("~0", "~")
        current = current[token]
    return current


def _resolve_local_references(
    value: Any,
    *,
    root: Mapping[str, Any],
    diagnostics: OllamaSchemaDiagnostics,
    stack: tuple[str, ...] = (),
) -> Any:
    if isinstance(value, list):
        return [
            _resolve_local_references(item, root=root, diagnostics=diagnostics, stack=stack)
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
                    target, root=root, diagnostics=diagnostics, stack=stack + (reference,),
                )
    return {
        key: _resolve_local_references(item, root=root, diagnostics=diagnostics, stack=stack)
        for key, item in value.items()
    }


def _adapt_keywords(
    value: Any,
    *,
    diagnostics: OllamaSchemaDiagnostics,
) -> Any:
    if isinstance(value, list):
        return [_adapt_keywords(item, diagnostics=diagnostics) for item in value]
    if not isinstance(value, dict):
        return value
    result: dict[str, Any] = {}
    for key, item in value.items():
        if key in {"properties", "patternProperties", "$defs", "definitions"} and isinstance(item, dict):
            result[key] = {
                property_name: _adapt_keywords(property_schema, diagnostics=diagnostics)
                for property_name, property_schema in item.items()
            }
            continue
        if key in ANNOTATION_KEYWORDS:
            diagnostics.removed_keyword_counts[key] = diagnostics.removed_keyword_counts.get(key, 0) + 1
            continue
        if key in COMPOSITION_KEYWORDS:
            diagnostics.composition_keyword_counts[key] = diagnostics.composition_keyword_counts.get(key, 0) + 1
        if key == "type" and isinstance(item, list) and "null" in item:
            diagnostics.nullable_type_array_count += 1
        if key == "const":
            diagnostics.converted_const_count += 1
            result["enum"] = [_adapt_keywords(item, diagnostics=diagnostics)]
            continue
        result[key] = _adapt_keywords(item, diagnostics=diagnostics)
    return result


def adapt_ollama_schema(strict_schema: Mapping[str, Any]) -> AdaptedOllamaSchema:
    """Return an Ollama grammar schema plus reproducible diagnostics.

    Local JSON pointers are inlined before ``$defs`` is removed. Composition
    keywords and nullable unions are retained and reported, never discarded.
    """
    strict_copy = deepcopy(dict(strict_schema))
    diagnostics = OllamaSchemaDiagnostics(strict_schema_bytes=_json_bytes(strict_copy))
    resolved = _resolve_local_references(strict_copy, root=strict_copy, diagnostics=diagnostics)
    assert isinstance(resolved, dict)
    if not diagnostics.unresolved_references:
        for keyword in ("$defs", "definitions"):
            if keyword in resolved:
                diagnostics.removed_keyword_counts[keyword] = diagnostics.removed_keyword_counts.get(keyword, 0) + 1
                resolved.pop(keyword)
    adapted = _adapt_keywords(resolved, diagnostics=diagnostics)
    assert isinstance(adapted, dict)
    diagnostics.provider_schema_bytes = _json_bytes(adapted)
    if diagnostics.unresolved_references:
        diagnostics.provider_schema_validation_result = "unresolved_references"
        diagnostics.validation_errors.append("Provider schema contains unresolved references")
    else:
        try:
            Draft202012Validator.check_schema(adapted)
            json.dumps(adapted, ensure_ascii=False, allow_nan=False)
        except (SchemaError, TypeError, ValueError) as exc:
            diagnostics.provider_schema_validation_result = "invalid"
            diagnostics.validation_errors.append(type(exc).__name__)
        else:
            diagnostics.provider_schema_validation_result = "valid"
    return AdaptedOllamaSchema(adapted, diagnostics)


def build_ollama_compatible_schema(strict_schema: Mapping[str, Any]) -> dict[str, Any]:
    """Build the provider-facing schema without mutating the strict schema."""
    return adapt_ollama_schema(strict_schema).schema
