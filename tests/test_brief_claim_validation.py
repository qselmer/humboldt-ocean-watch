from copy import deepcopy

import pytest

from src.brief_claim_validation import validate_cross_language, validate_scientific_brief
from src.brief_fact_coverage import derive_facts_used_from_claim_support
from tests.scientific_brief_test_data import generation_settings, valid_brief, validated_context


def _codes(payload, context=None, language="en") -> set[str]:
    context = context or validated_context()
    report = _report(payload, context=context, language=language)
    return {item["code"] for item in report.errors}


def _report(payload, context=None, language="en"):
    context = context or validated_context()
    return validate_scientific_brief(
        payload, context, language=language, settings=generation_settings()
    )


def test_valid_observation_and_boundary_warning_pass() -> None:
    assert _codes(valid_brief("en")) == set()


def test_wrong_analysis_date_and_word_limit_fail() -> None:
    payload = deepcopy(valid_brief("en"))
    payload["analysis_date"] = "2026-07-18"
    assert "analysis_date_mismatch" in _codes(payload)
    payload = deepcopy(valid_brief("en"))
    payload["executive_summary"]["paragraphs"][0]["text"] = "regional " * 181
    assert "section_word_limit" in _codes(payload)


@pytest.mark.parametrize(
    ("mutate", "expected_code"),
    [
        (lambda payload: payload.update({"executive_summary": "unexpected text"}), "invalid_section_type"),
        (
            lambda payload: payload["executive_summary"].update({"paragraphs": ["unexpected text"]}),
            "invalid_paragraph_type",
        ),
        (
            lambda payload: payload["executive_summary"].update({"paragraphs": {"text": "unexpected"}}),
            "invalid_paragraphs_type",
        ),
        (
            lambda payload: payload["executive_summary"]["paragraphs"][0].update({"text": ["unexpected"]}),
            "invalid_text_field_type",
        ),
        (lambda payload: payload.update({"executive_summary": None}), "invalid_section_type"),
    ],
)
def test_structurally_invalid_model_output_is_a_validation_failure(
    mutate, expected_code: str,
) -> None:
    payload = deepcopy(valid_brief("en"))
    mutate(payload)
    report = validate_scientific_brief(
        payload,
        validated_context(),
        language="en",
        settings=generation_settings(),
    )
    assert report.final_status == "failed"
    assert expected_code in {item["code"] for item in report.errors}
    assert report.checks == {"schema_validation": False, "semantic_validation": False}


def test_schema_gate_runs_before_semantic_language_traversal(monkeypatch) -> None:
    payload = deepcopy(valid_brief("en"))
    payload["executive_summary"] = "regression payload matching the runtime failure"

    def unsafe_semantic_validator(*args, **kwargs):
        raise AssertionError("semantic validation must not run after schema failure")

    monkeypatch.setattr(
        "src.brief_claim_validation.validate_brief_language",
        unsafe_semantic_validator,
    )
    report = validate_scientific_brief(
        payload,
        validated_context(),
        language="en",
        settings=generation_settings(),
    )
    assert report.final_status == "failed"
    assert report.errors[0]["code"] == "invalid_section_type"


@pytest.mark.parametrize(
    ("text", "code"),
    [
        ("This warming was caused by an external mechanism.", "prohibited_causal_attribution"),
        ("The anomaly will increase tomorrow.", "prohibited_forecast"),
        ("This creates a biological impact.", "prohibited_biological_fisheries_impact"),
        ("This creates a fisheries impact.", "prohibited_biological_fisheries_impact"),
        ("This is an official classification.", "prohibited_official_classification"),
        ("Tracks are individual water parcels.", "prohibited_track_water_parcel"),
        ("This is a representativeness severity ranking.", "prohibited_ordinal_representativeness"),
        ("According to a citation [1], conditions changed.", "prohibited_external_reference"),
    ],
)
def test_prohibited_claims_are_rejected(text: str, code: str) -> None:
    payload = deepcopy(valid_brief("en"))
    payload["event_status"]["paragraphs"][0]["text"] = text
    assert code in _codes(payload)


@pytest.mark.parametrize(
    ("kind", "expected"),
    [
        ("unknown", "unknown_fact_id"),
        ("unavailable", "unavailable_fact_used"),
        ("disallowed", "disallowed_fact_used"),
        ("suppressed", "suppressed_fact_used"),
        ("missing", "missing_supporting_fact"),
    ],
)
def test_fact_registry_enforcement(kind: str, expected: str) -> None:
    context = deepcopy(validated_context())
    payload = deepcopy(valid_brief("en", context))
    paragraph = payload["regional_state"]["paragraphs"][0]
    target = "regional.mean_sst_c"
    if kind == "unknown":
        paragraph["supporting_fact_ids"].append("unknown.fact")
        payload["facts_used"].append("unknown.fact")
    elif kind == "missing":
        paragraph["supporting_fact_ids"] = []
        payload["facts_used"].remove(target)
    else:
        fact = next(item for item in context["fact_registry"] if item["fact_id"] == target)
        if kind == "unavailable":
            fact["status"] = "unavailable"
            fact["value"] = None
        elif kind == "disallowed":
            fact["allowed_in_brief"] = False
        else:
            context["quality_flags"].append(
                {
                    "flag_id": "suppress_sst",
                    "severity": "warning",
                    "section": "regional_state",
                    "message_code": "suppress_sst",
                    "message_en": "SST is suppressed.",
                    "message_es": "La TSM está suprimida.",
                    "affected_fact_ids": [target],
                    "suppress_dependent_claims": True,
                }
            )
            payload["data_quality"]["paragraphs"][0]["quality_flag_ids"].append("suppress_sst")
            payload["regional_state"]["status"] = "limited"
            payload["regional_state"]["reason"] = "The SST fact is suppressed."
    assert expected in _codes(payload, context)


def test_required_limitations_cannot_be_dropped() -> None:
    payload = deepcopy(valid_brief("en"))
    payload["limitations"] = payload["limitations"][:-1]
    assert "required_limitations_missing" in _codes(payload)


def test_insufficient_coverage_monthly_fallback_and_stale_product_are_disclosed() -> None:
    context = deepcopy(validated_context())
    context["quality"]["valid_coverage"]["value"] = 0.5
    assert "insufficient_coverage_not_limited" in _codes(valid_brief("en", context), context)

    context = deepcopy(validated_context())
    context["climatology"]["fallback_used"] = True
    assert "monthly_fallback_not_disclosed" in _codes(valid_brief("en", context), context)

    context = deepcopy(validated_context())
    context["quality_flags"].append(
        {
            "flag_id": "stale_optional_product", "severity": "warning", "section": "quality",
            "message_code": "stale_optional_product", "affected_fact_ids": [],
            "suppress_dependent_claims": False,
        }
    )
    payload = valid_brief("en", context)
    payload["data_quality"]["paragraphs"][0]["quality_flag_ids"].remove("stale_optional_product")
    assert "material_quality_flag_omitted" in _codes(payload, context)


def test_no_active_event_and_zero_patches_are_valid_states() -> None:
    context = deepcopy(validated_context())
    context["univariate_event"]["active"] = False
    patch_fact = next(item for item in context["fact_registry"] if item["fact_id"] == "patches.patch_count")
    patch_fact["value"] = 0
    assert "prohibited_claim" not in " ".join(_codes(valid_brief("en", context), context))


def test_facts_used_alone_does_not_satisfy_mandatory_claim_coverage() -> None:
    payload = deepcopy(valid_brief("en"))
    payload["event_status"]["paragraphs"][0]["text"] = (
        "The active univariate event is on day 69."
    )
    payload["event_status"]["paragraphs"][0]["supporting_fact_ids"].remove(
        "event.event_day"
    )

    report = _report(payload)
    issue = next(
        item
        for item in report.errors
        if item["code"] == "required_operational_fact_not_represented"
        and item["fact_id"] == "event.event_day"
    )

    assert issue["present_in_facts_used"] is True
    assert issue["present_in_claim_support"] is False
    assert issue["claim_support_field"] == "supporting_fact_ids"
    assert issue["suggested_sections"] == ["event_status"]


def test_missing_operational_fact_diagnostics_are_one_per_fact() -> None:
    payload = deepcopy(valid_brief("en"))
    missing = {
        "activity.active_family_count",
        "event.event_day",
        "patches.patch_count",
    }
    paragraph = payload["event_status"]["paragraphs"][0]
    paragraph["supporting_fact_ids"] = [
        fact_id
        for fact_id in paragraph["supporting_fact_ids"]
        if fact_id not in missing
    ]
    payload["facts_used"] = [
        fact_id for fact_id in payload["facts_used"] if fact_id not in missing
    ]

    report = _report(payload)
    issues = [
        item
        for item in report.errors
        if item["code"] == "required_operational_fact_not_represented"
    ]

    assert {item["fact_id"] for item in issues} == missing
    assert all(item["available"] is True for item in issues)
    assert all(item["allowed_in_brief"] is True for item in issues)
    assert all(item["value_available"] is True for item in issues)
    assert all(item["present_in_facts_used"] is False for item in issues)
    assert all(item["present_in_claim_support"] is False for item in issues)
    assert all(item["authoritative_value_available"] is True for item in issues)
    assert all(item["valid_zero"] is False for item in issues)
    assert all(item["claim_support_locations_inspected"] for item in issues)
    assert all(
        not any(item["present_in_each_inspected_location"].values())
        for item in issues
    )
    assert all(item["derived_facts_used_membership"] is False for item in issues)
    assert all(item["repair_instruction_generated"] is False for item in issues)
    assert all("text" not in item and "prompt" not in item for item in issues)


def test_non_substantive_limitation_reference_does_not_satisfy_coverage() -> None:
    payload = deepcopy(valid_brief("en"))
    payload["event_status"]["paragraphs"][0]["supporting_fact_ids"].remove(
        "event.event_day"
    )
    payload["limitations"][0]["supporting_fact_ids"].append("event.event_day")

    report = _report(payload)

    assert any(
        item["code"] == "required_operational_fact_not_represented"
        and item["fact_id"] == "event.event_day"
        for item in report.errors
    )


def test_zero_patch_and_activity_counts_still_require_and_pass_claim_coverage() -> None:
    context = deepcopy(validated_context())
    for fact_id in (
        "patches.patch_count",
        "activity.active_track_count",
        "activity.active_family_count",
    ):
        next(
            item for item in context["fact_registry"] if item["fact_id"] == fact_id
        )["value"] = 0

    report = _report(valid_brief("en", context), context=context)

    assert not any(
        item["code"] == "required_operational_fact_not_represented"
        for item in report.errors
    )


def test_missing_valid_zero_family_count_is_identified_as_valid_zero() -> None:
    context = deepcopy(validated_context())
    next(
        item
        for item in context["fact_registry"]
        if item["fact_id"] == "activity.active_family_count"
    )["value"] = 0
    payload = valid_brief("en", context)
    payload["event_status"]["paragraphs"][0]["supporting_fact_ids"].remove(
        "activity.active_family_count"
    )
    payload["facts_used"].remove("activity.active_family_count")

    report = _report(payload, context=context)
    issue = next(
        item
        for item in report.errors
        if item.get("fact_id") == "activity.active_family_count"
    )

    assert issue["valid_zero"] is True
    assert issue["authoritative_value_available"] is True


def test_derived_unknown_support_id_is_rejected_not_silently_dropped() -> None:
    payload = deepcopy(valid_brief("en"))
    payload["event_status"]["paragraphs"][0]["supporting_fact_ids"].append(
        "unknown.operational_fact"
    )
    derive_facts_used_from_claim_support(payload)

    report = _report(payload)

    assert "unknown.operational_fact" in payload["facts_used"]
    assert any(item["code"] == "unknown_fact_id" for item in report.errors)
    assert not any(item["code"] == "facts_used_mismatch" for item in report.errors)


def test_cross_language_consistency_and_failures() -> None:
    es, en = valid_brief("es"), valid_brief("en")
    assert validate_cross_language(es, en)["final_status"] == "passed"

    inconsistent = deepcopy(en)
    inconsistent["recent_evolution"]["paragraphs"][0]["text"] = "The anomaly is increasing."
    assert validate_cross_language(es, inconsistent)["final_status"] == "failed"

    inconsistent = deepcopy(en)
    inconsistent["event_status"]["paragraphs"][0]["supporting_fact_ids"].remove("patches.patch_count")
    inconsistent["facts_used"].remove("patches.patch_count")
    assert not validate_cross_language(es, inconsistent)["checks"]["patch_count"]

    inconsistent = deepcopy(en)
    inconsistent["limitations"] = inconsistent["limitations"][:-1]
    assert not validate_cross_language(es, inconsistent)["checks"]["limitations"]
