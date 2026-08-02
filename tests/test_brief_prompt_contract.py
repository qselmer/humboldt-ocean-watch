from copy import deepcopy
import json

from src.brief_output_schema import scientific_brief_json_schema
from src.brief_prompt_contract import (
    build_compact_response_contract,
    build_repair_contract,
    compact_contract_for_path,
)


def _shape(contract, field_name):
    reference = contract["root"]["properties"][field_name]
    return contract["shapes"][reference["shape"]]


def test_compact_contract_is_deterministic_and_substantially_smaller() -> None:
    schema = scientific_brief_json_schema()
    first = build_compact_response_contract(schema)
    second = build_compact_response_contract(schema)
    assert first == second
    compact_bytes = len(json.dumps(first, separators=(",", ":")))
    schema_bytes = len(json.dumps(schema, separators=(",", ":")))
    assert compact_bytes < schema_bytes * 0.85
    assert "$ref" not in json.dumps(first)
    assert "$schema" not in first


def test_compact_contract_does_not_mutate_authoritative_schema() -> None:
    schema = scientific_brief_json_schema()
    original = deepcopy(schema)
    build_compact_response_contract(schema)
    assert schema == original


def test_contract_contains_every_required_root_field() -> None:
    schema = scientific_brief_json_schema()
    contract = build_compact_response_contract(schema)
    assert contract["root"]["required"] == schema["required"]
    assert set(contract["root"]["properties"]) == set(schema["properties"])
    assert contract["root"]["additionalProperties"] is False


def test_contract_contains_exact_section_and_paragraph_shapes() -> None:
    contract = build_compact_response_contract()
    section = _shape(contract, "regional_state")
    assert section["type"] == "object"
    assert section["additionalProperties"] is False
    assert section["required"] == ["heading", "paragraphs", "status", "reason"]
    assert set(section["properties"]) == set(section["required"])
    paragraph_ref = section["properties"]["paragraphs"]["items"]
    paragraph = contract["shapes"][paragraph_ref["shape"]]
    assert paragraph["required"] == [
        "paragraph_id", "text", "supporting_fact_ids", "quality_flag_ids", "claim_types"
    ]
    assert paragraph["additionalProperties"] is False
    assert paragraph["properties"]["claim_types"]["minItems"] == 1
    assert paragraph["properties"]["claim_types"]["items"]["enum"]


def test_contract_keeps_distinct_executive_paragraph_limit() -> None:
    contract = build_compact_response_contract()
    executive = _shape(contract, "executive_summary")
    narrative = _shape(contract, "regional_state")
    assert executive["properties"]["paragraphs"]["maxItems"] == 3
    assert narrative["properties"]["paragraphs"]["maxItems"] == 5


def test_contract_contains_exact_key_message_item_shape() -> None:
    item = compact_contract_for_path("key_messages.0")
    assert item["type"] == "object"
    assert item["additionalProperties"] is False
    assert item["required"] == [
        "message_id", "text", "supporting_fact_ids", "quality_flag_ids", "claim_types"
    ]
    assert "message" not in item["properties"]


def test_contract_contains_exact_limitation_item_shape() -> None:
    item = compact_contract_for_path("limitations[0]")
    assert item["required"] == [
        "limitation_id", "text", "supporting_fact_ids", "quality_flag_ids"
    ]
    assert item["additionalProperties"] is False
    assert "statement" not in item["properties"]


def test_repair_contract_uses_exact_schema_fragment_for_each_path() -> None:
    repairs = build_repair_contract([
        {"code": "invalid_section_type", "path": "regional_state", "expected_type": "object", "actual_type": "string"},
        {"code": "additional_property_not_allowed", "path": "key_messages.0", "unexpected_field": "message"},
        {"code": "missing_required_field", "path": "limitations.0", "missing_field": "text", "expected_type": "string"},
    ])
    assert repairs[0]["expected_shape"]["required"] == ["heading", "paragraphs", "status", "reason"]
    assert repairs[1]["expected_shape"]["required"][0] == "message_id"
    assert repairs[2]["expected_shape"] == {"type": "string", "maxLength": 1200}


def test_repair_contract_retains_only_safe_operational_fact_diagnostics() -> None:
    repairs = build_repair_contract(
        [
            {
                "code": "required_operational_fact_not_represented",
                "path": "facts_used",
                "message": "generated prose must not be copied",
                "fact_id": "event.event_day",
                "available": True,
                "allowed_in_brief": True,
                "value_available": True,
                "authoritative_value_available": True,
                "valid_zero": False,
                "required_representation": "paragraph_or_key_message_claim_support",
                "claim_support_field": "supporting_fact_ids",
                "suggested_sections": ["event_status"],
                "present_in_facts_used": True,
                "present_in_claim_support": False,
                "claim_support_locations_inspected": [
                    "event_status.paragraphs[0].supporting_fact_ids"
                ],
                "present_in_each_inspected_location": {
                    "event_status.paragraphs[0].supporting_fact_ids": False
                },
                "derived_facts_used_membership": False,
                "repair_instruction_generated": False,
                "provider_secret": "must-not-appear",
            }
        ]
    )
    repair = repairs[0]

    assert repair["fact_id"] == "event.event_day"
    assert repair["suggested_sections"] == ["event_status"]
    assert repair["present_in_facts_used"] is True
    assert repair["present_in_claim_support"] is False
    assert repair["authoritative_value_available"] is True
    assert repair["valid_zero"] is False
    assert repair["derived_facts_used_membership"] is False
    assert repair["claim_support_locations_inspected"]
    assert "message" not in repair
    assert "provider_secret" not in repair
