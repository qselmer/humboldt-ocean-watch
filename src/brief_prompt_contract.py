"""Compact, schema-derived response contracts for scientific-brief prompts.

The strict JSON Schema in :mod:`src.brief_output_schema` remains authoritative.
This module only produces a smaller prompt-oriented representation.  Repeated
object shapes are named once, and all local JSON-Schema references are resolved
before the contract is emitted.
"""

from __future__ import annotations

from copy import deepcopy
import json
import re
from typing import Any, Mapping

from src.brief_output_schema import scientific_brief_json_schema


_CONSTRAINT_KEYS = (
    "type",
    "const",
    "enum",
    "additionalProperties",
    "required",
    "properties",
    "items",
    "minItems",
    "maxItems",
    "minLength",
    "maxLength",
    "minimum",
    "maximum",
)


def _resolve_local_ref(ref: str, root: Mapping[str, Any]) -> Mapping[str, Any]:
    if not ref.startswith("#/"):
        raise ValueError(f"Only local JSON-Schema references are supported: {ref}")
    node: Any = root
    for raw_part in ref[2:].split("/"):
        part = raw_part.replace("~1", "/").replace("~0", "~")
        if not isinstance(node, Mapping) or part not in node:
            raise ValueError(f"Unresolvable local JSON-Schema reference: {ref}")
        node = node[part]
    if not isinstance(node, Mapping):
        raise ValueError(f"Local JSON-Schema reference does not identify an object: {ref}")
    return node


def _compact_inline(node: Mapping[str, Any], root: Mapping[str, Any]) -> dict[str, Any]:
    """Return relevant generation constraints with local references expanded."""
    if "$ref" in node:
        resolved = _resolve_local_ref(str(node["$ref"]), root)
        siblings = {key: value for key, value in node.items() if key != "$ref"}
        merged = {**deepcopy(resolved), **deepcopy(siblings)}
        return _compact_inline(merged, root)

    compact: dict[str, Any] = {}
    for key in _CONSTRAINT_KEYS:
        if key not in node:
            continue
        value = node[key]
        if key == "properties":
            compact[key] = {
                str(name): _compact_inline(schema, root)
                for name, schema in value.items()
            }
        elif key == "items" and isinstance(value, Mapping):
            compact[key] = _compact_inline(value, root)
        else:
            compact[key] = deepcopy(value)
    return compact


def _shape_reference_for(
    schema: Mapping[str, Any],
    *,
    root: Mapping[str, Any],
    shapes: dict[str, dict[str, Any]],
    signatures: dict[str, str],
    preferred_name: str,
) -> dict[str, str]:
    compact = _compact_with_named_refs(schema, root=root, shapes=shapes)
    signature = json.dumps(compact, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
    if signature in signatures:
        return {"shape": signatures[signature]}
    name = preferred_name
    suffix = 2
    while name in shapes:
        name = f"{preferred_name}_{suffix}"
        suffix += 1
    shapes[name] = compact
    signatures[signature] = name
    return {"shape": name}


def _compact_with_named_refs(
    node: Mapping[str, Any],
    *,
    root: Mapping[str, Any],
    shapes: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    if "$ref" in node:
        ref = str(node["$ref"])
        name = ref.rsplit("/", 1)[-1]
        if name not in shapes:
            shapes[name] = _compact_with_named_refs(
                _resolve_local_ref(ref, root), root=root, shapes=shapes
            )
        return {"shape": name}

    compact: dict[str, Any] = {}
    for key in _CONSTRAINT_KEYS:
        if key not in node:
            continue
        value = node[key]
        if key == "properties":
            compact[key] = {
                str(name): _compact_with_named_refs(schema, root=root, shapes=shapes)
                for name, schema in value.items()
            }
        elif key == "items" and isinstance(value, Mapping):
            compact[key] = _compact_with_named_refs(value, root=root, shapes=shapes)
        else:
            compact[key] = deepcopy(value)
    return compact


def build_compact_response_contract(
    schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Build a deterministic compact contract without mutating ``schema``.

    ``shape`` is prompt notation, not a JSON-Schema reference: it means that
    the value must have exactly the named object shape declared in ``shapes``.
    All object shapes retain their required-field lists and
    ``additionalProperties`` rules.
    """
    source = deepcopy(schema if schema is not None else scientific_brief_json_schema())
    properties = source.get("properties")
    if not isinstance(properties, Mapping):
        raise ValueError("Scientific brief schema must define root properties")

    shapes: dict[str, dict[str, Any]] = {}
    signatures: dict[str, str] = {}
    root_properties: dict[str, Any] = {}
    for name, property_schema in properties.items():
        if not isinstance(property_schema, Mapping):
            raise ValueError(f"Schema property {name} must be an object")
        if property_schema.get("type") == "object":
            preferred = (
                "executive_summary_section"
                if name == "executive_summary"
                else "narrative_section"
            )
            root_properties[str(name)] = _shape_reference_for(
                property_schema,
                root=source,
                shapes=shapes,
                signatures=signatures,
                preferred_name=preferred,
            )
        else:
            root_properties[str(name)] = _compact_with_named_refs(
                property_schema, root=source, shapes=shapes
            )

    root_contract: dict[str, Any] = {
        "type": source.get("type"),
        "additionalProperties": source.get("additionalProperties"),
        "required": deepcopy(source.get("required", [])),
        "properties": root_properties,
    }
    return {
        "contract_version": str(source.get("properties", {}).get("schema_version", {}).get("const", "unknown")),
        "notation": "shape means the exact object definition in shapes; no unlisted object fields are allowed",
        "root": root_contract,
        "shapes": shapes,
    }


def _path_tokens(path: str) -> list[str | int]:
    normalized = str(path or "root").strip()
    if normalized == "root":
        return []
    if normalized.startswith("root."):
        normalized = normalized[5:]
    tokens: list[str | int] = []
    for name, bracket_index in re.findall(r"([^.\[\]]+)|\[(\d+)\]", normalized):
        token = name or bracket_index
        tokens.append(int(token) if token.isdigit() else token)
    return tokens


def _schema_at_path(
    path: str,
    *,
    root: Mapping[str, Any],
    missing_field: str | None = None,
) -> Mapping[str, Any]:
    node: Mapping[str, Any] = root
    for token in _path_tokens(path):
        while "$ref" in node:
            node = _resolve_local_ref(str(node["$ref"]), root)
        if isinstance(token, int):
            items = node.get("items")
            if not isinstance(items, Mapping):
                raise ValueError(f"Path does not identify an array item: {path}")
            node = items
        else:
            properties = node.get("properties")
            if not isinstance(properties, Mapping) or token not in properties:
                raise ValueError(f"Path is not present in the response schema: {path}")
            child = properties[token]
            if not isinstance(child, Mapping):
                raise ValueError(f"Schema path is not an object: {path}")
            node = child
    while "$ref" in node:
        node = _resolve_local_ref(str(node["$ref"]), root)
    if missing_field:
        properties = node.get("properties")
        if not isinstance(properties, Mapping) or missing_field not in properties:
            raise ValueError(f"Missing field is not present in the response schema: {missing_field}")
        child = properties[missing_field]
        if not isinstance(child, Mapping):
            raise ValueError(f"Missing-field schema is not an object: {missing_field}")
        node = child
    return node


def compact_contract_for_path(
    path: str,
    *,
    missing_field: str | None = None,
    schema: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Return the exact expanded schema fragment for one validation path."""
    source = deepcopy(schema if schema is not None else scientific_brief_json_schema())
    node = _schema_at_path(path, root=source, missing_field=missing_field)
    return _compact_inline(node, source)


def build_repair_contract(
    validation_errors: list[Mapping[str, Any]],
    *,
    schema: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Build sanitized, path-specific repair requirements from schema errors."""
    allowed_detail_keys = (
        "unexpected_field",
        "allowed_fields",
        "missing_field",
        "expected_type",
        "actual_type",
        "allowed_values",
        "expected_constant",
        "fact_id",
        "available",
        "allowed_in_brief",
        "value_available",
        "authoritative_value_available",
        "valid_zero",
        "required_representation",
        "claim_support_field",
        "suggested_sections",
        "present_in_facts_used",
        "present_in_claim_support",
        "claim_support_locations_inspected",
        "present_in_each_inspected_location",
        "derived_facts_used_membership",
        "repair_instruction_generated",
    )
    repairs: list[dict[str, Any]] = []
    seen: set[str] = set()
    for issue in validation_errors:
        code = re.sub(r"[^A-Za-z0-9_.-]", "_", str(issue.get("code", "validation_error")))[:120]
        path = re.sub(r"[^A-Za-z0-9_.\[\]-]", "_", str(issue.get("path", "root")))[:240]
        item: dict[str, Any] = {
            "code": code or "validation_error",
            "path": path or "root",
        }
        for key in allowed_detail_keys:
            if key in issue:
                item[key] = deepcopy(issue[key])
        try:
            item["expected_shape"] = compact_contract_for_path(
                item["path"],
                missing_field=(
                    str(item["missing_field"])
                    if "missing_field" in item
                    else None
                ),
                schema=schema,
            )
        except ValueError:
            item["expected_shape"] = {"type": "object"} if item["path"] == "root" else None
        signature = json.dumps(item, ensure_ascii=False, sort_keys=True, separators=(",", ":"))
        if signature not in seen:
            seen.add(signature)
            repairs.append(item)
    return repairs
