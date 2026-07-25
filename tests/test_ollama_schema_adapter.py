from copy import deepcopy
import json

from src.brief_output_schema import scientific_brief_json_schema
from src.ollama_schema_adapter import (
    adapt_ollama_schema,
    build_ollama_compatible_schema,
)


def test_strict_schema_is_not_mutated_and_adaptation_is_stable() -> None:
    strict = scientific_brief_json_schema()
    original = deepcopy(strict)
    first = build_ollama_compatible_schema(strict)
    second = build_ollama_compatible_schema(strict)
    assert strict == original
    assert first == second
    assert json.dumps(first, sort_keys=True, allow_nan=False)


def test_metadata_removed_const_converted_and_structure_preserved() -> None:
    strict = {
        "$schema": "https://json-schema.org/draft/2020-12/schema",
        "title": "Annotated",
        "description": "not required by grammar",
        "type": "object",
        "required": ["status", "items"],
        "additionalProperties": False,
        "properties": {
            "status": {"const": "OK", "description": "annotation"},
            "items": {
                "type": "array",
                "minItems": 1,
                "maxItems": 2,
                "items": {
                    "type": "object",
                    "required": ["value"],
                    "additionalProperties": False,
                    "properties": {"value": {"type": "number", "minimum": 0}},
                },
            },
        },
    }
    result = adapt_ollama_schema(strict)
    adapted = result.schema
    serialized = json.dumps(adapted)
    assert "$schema" not in adapted
    assert "title" not in adapted
    assert "description" not in serialized
    assert adapted["properties"]["status"] == {"enum": ["OK"]}
    assert adapted["required"] == ["status", "items"]
    assert adapted["additionalProperties"] is False
    nested = adapted["properties"]["items"]
    assert nested["minItems"] == 1 and nested["maxItems"] == 2
    assert nested["items"]["required"] == ["value"]
    assert result.diagnostics.converted_const_count == 1


def test_local_references_are_inlined_deterministically() -> None:
    strict = {
        "$defs": {
            "entry": {
                "type": "object",
                "required": ["name"],
                "additionalProperties": False,
                "properties": {"name": {"type": "string", "minLength": 1}},
            }
        },
        "type": "array",
        "items": {"$ref": "#/$defs/entry"},
    }
    result = adapt_ollama_schema(strict)
    assert "$defs" not in result.schema
    assert "$ref" not in json.dumps(result.schema)
    assert result.schema["items"]["required"] == ["name"]
    assert result.diagnostics.resolved_reference_count == 1
    assert result.diagnostics.unresolved_reference_count == 0


def test_unresolved_references_are_reported_not_silently_removed() -> None:
    strict = {"type": "object", "properties": {"x": {"$ref": "#/missing/value"}}}
    result = adapt_ollama_schema(strict)
    assert result.schema["properties"]["x"]["$ref"] == "#/missing/value"
    assert result.diagnostics.unresolved_reference_count == 1
    assert result.diagnostics.provider_schema_validation_result == "unresolved_references"


def test_composition_and_nullable_arrays_are_retained_and_reported() -> None:
    strict = {
        "anyOf": [
            {"type": ["string", "null"]},
            {"allOf": [{"type": "number"}, {"minimum": 0}]},
        ]
    }
    result = adapt_ollama_schema(strict)
    assert "anyOf" in result.schema
    assert "allOf" in result.schema["anyOf"][1]
    assert result.schema["anyOf"][0]["type"] == ["string", "null"]
    assert result.diagnostics.composition_keyword_counts == {"anyOf": 1, "allOf": 1}
    assert result.diagnostics.nullable_type_array_count == 1


def test_real_scientific_schema_preserves_required_constraints() -> None:
    strict = scientific_brief_json_schema()
    result = adapt_ollama_schema(strict)
    assert result.diagnostics.provider_schema_validation_result == "valid"
    assert result.diagnostics.resolved_reference_count > 0
    assert result.diagnostics.unresolved_reference_count == 0
    assert result.schema["required"] == strict["required"]
    assert set(result.schema["properties"]) == set(strict["properties"])
    assert result.schema["additionalProperties"] is False

