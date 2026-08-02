from copy import deepcopy
import json

import pytest

from src.brief_output_schema import (
    model_generated_brief_json_schema,
    responses_text_format,
    scientific_brief_json_schema,
    validate_output_schema,
)
from tests.scientific_brief_test_data import valid_brief


@pytest.mark.parametrize("language", ["es", "en"])
def test_valid_bilingual_schema(language: str) -> None:
    assert validate_output_schema(valid_brief(language)) == []


@pytest.mark.parametrize(
    ("mutation", "expected_path"),
    [
        (lambda payload: payload.pop("title"), "root"),
        (lambda payload: payload.update({"extra": True}), "root"),
        (lambda payload: payload["regional_state"].update({"status": "bad"}), "regional_state.status"),
        (lambda payload: payload["regional_state"]["paragraphs"][0].update({"claim_types": ["prediction"]}), "regional_state.paragraphs.0.claim_types.0"),
    ],
)
def test_schema_rejects_missing_extra_and_invalid_enums(mutation, expected_path: str) -> None:
    payload = deepcopy(valid_brief("en"))
    mutation(payload)
    assert expected_path in {issue["path"] for issue in validate_output_schema(payload)}


def test_responses_format_is_strict_json_schema() -> None:
    value = responses_text_format()
    assert value["type"] == "json_schema"
    assert value["strict"] is True
    assert value["schema"]["additionalProperties"] is False


def test_provider_schema_omits_only_application_owned_disclaimer() -> None:
    final_schema = scientific_brief_json_schema()
    model_schema = model_generated_brief_json_schema()
    assert "disclaimer" in final_schema["required"]
    assert "disclaimer" in final_schema["properties"]
    assert "disclaimer" not in model_schema["required"]
    assert "disclaimer" not in model_schema["properties"]
    assert set(final_schema["required"]) - set(model_schema["required"]) == {"disclaimer"}


def test_additional_property_diagnostic_names_field_and_allowed_fields_without_value() -> None:
    payload = deepcopy(valid_brief("en"))
    secret_prose = "generated paragraph text API-key-secret prompt context"
    payload["regional_state"] = {"text": secret_prose}
    issues = validate_output_schema(payload)
    unexpected = next(
        issue for issue in issues
        if issue["code"] == "additional_property_not_allowed"
        and issue["path"] == "regional_state"
    )
    assert unexpected["unexpected_field"] == "text"
    assert unexpected["allowed_fields"] == ["heading", "paragraphs", "reason", "status"]
    assert secret_prose not in json.dumps(issues)


def test_missing_field_diagnostics_name_each_field_and_expected_type() -> None:
    payload = deepcopy(valid_brief("en"))
    payload["regional_state"] = {"text": "not retained"}
    issues = validate_output_schema(payload)
    missing = {
        (issue["missing_field"], issue["expected_type"])
        for issue in issues
        if issue["code"] == "missing_required_field"
        and issue["path"] == "regional_state"
    }
    assert missing == {
        ("heading", "string"),
        ("paragraphs", "array"),
        ("status", "string"),
        ("reason", "string or null"),
    }


def test_wrong_type_diagnostic_has_expected_and_actual_types() -> None:
    payload = deepcopy(valid_brief("en"))
    payload["regional_state"] = "flattened section"
    issue = next(
        issue for issue in validate_output_schema(payload)
        if issue["path"] == "regional_state"
    )
    assert issue["code"] == "invalid_section_type"
    assert issue["expected_type"] == "object"
    assert issue["actual_type"] == "string"


def test_invalid_array_item_diagnostic_keeps_exact_item_path() -> None:
    payload = deepcopy(valid_brief("en"))
    payload["key_messages"] = [{"message": "invalid wrapper"}]
    issues = validate_output_schema(payload)
    item_issues = [issue for issue in issues if issue["path"] == "key_messages.0"]
    assert item_issues
    assert any(issue.get("unexpected_field") == "message" for issue in item_issues)
    assert {issue.get("missing_field") for issue in item_issues} >= {
        "message_id", "text", "supporting_fact_ids", "quality_flag_ids", "claim_types"
    }


@pytest.mark.parametrize(
    ("field", "invalid_item", "unexpected"),
    [
        ("key_messages", {"message": "invalid"}, "message"),
        ("limitations", {"limitation_id": "limit-1", "statement": "invalid"}, "statement"),
    ],
)
def test_guessed_wrapper_shapes_remain_strictly_rejected(
    field: str, invalid_item: dict, unexpected: str,
) -> None:
    payload = deepcopy(valid_brief("en"))
    payload[field] = [invalid_item]
    issues = validate_output_schema(payload)
    assert any(issue.get("unexpected_field") == unexpected for issue in issues)
    assert issues
