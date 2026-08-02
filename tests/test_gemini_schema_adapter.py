from copy import deepcopy
import json

from src.brief_output_schema import scientific_brief_json_schema
from src.gemini_schema_adapter import build_gemini_compatible_schema


def test_adapter_is_stable_and_does_not_mutate_strict_schema() -> None:
    strict = scientific_brief_json_schema()
    original = deepcopy(strict)
    first = build_gemini_compatible_schema(strict)
    second = build_gemini_compatible_schema(strict)

    assert strict == original
    assert first.schema == second.schema
    assert first.diagnostics.to_dict() == second.diagnostics.to_dict()
    assert first.diagnostics.provider_schema_validation_result == "valid"
    assert first.diagnostics.adaptation_status == "adapted_with_preserved_unsupported_constraints"
    assert first.diagnostics.unsupported_keyword_counts["maxLength"] > 0
    assert first.diagnostics.provider_schema_bytes > 0
    json.dumps(first.schema, allow_nan=False)


def test_references_and_const_are_resolved_deterministically() -> None:
    result = build_gemini_compatible_schema(scientific_brief_json_schema())
    serialized = json.dumps(result.schema, sort_keys=True)

    assert "$ref" not in serialized
    assert "$defs" not in result.schema
    assert "const" not in serialized
    assert result.diagnostics.resolved_reference_count > 0
    assert result.diagnostics.converted_const_count == 1
    assert result.schema["properties"]["schema_version"]["enum"] == ["1.0.0"]


def test_required_nested_objects_arrays_and_additional_properties_are_preserved() -> None:
    result = build_gemini_compatible_schema(scientific_brief_json_schema())
    schema = result.schema
    section = schema["properties"]["executive_summary"]
    paragraph = section["properties"]["paragraphs"]["items"]

    assert set(schema["required"]) >= {"executive_summary", "limitations", "disclaimer"}
    assert schema["additionalProperties"] is False
    assert section["type"] == "object"
    assert section["additionalProperties"] is False
    assert section["properties"]["paragraphs"]["type"] == "array"
    assert paragraph["required"] == [
        "paragraph_id",
        "text",
        "supporting_fact_ids",
        "quality_flag_ids",
        "claim_types",
    ]
    assert result.diagnostics.additional_properties_preserved_count > 0


def test_unsupported_structural_keyword_is_reported_and_not_discarded() -> None:
    strict = {
        "type": "object",
        "properties": {"value": {"type": "number", "exclusiveMinimum": 0}},
        "required": ["value"],
        "additionalProperties": False,
    }
    result = build_gemini_compatible_schema(strict)

    assert result.schema["properties"]["value"]["exclusiveMinimum"] == 0
    assert result.diagnostics.unsupported_keyword_counts == {"exclusiveMinimum": 1}
    assert result.diagnostics.adaptation_status == "adapted_with_preserved_unsupported_constraints"
    assert result.diagnostics.provider_schema_validation_result == "valid"


def test_unresolved_reference_is_an_explicit_failure() -> None:
    result = build_gemini_compatible_schema(
        {"type": "object", "properties": {"value": {"$ref": "#/missing"}}}
    )
    assert result.diagnostics.unresolved_reference_count == 1
    assert result.diagnostics.adaptation_status == "failed"
