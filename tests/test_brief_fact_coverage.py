from copy import deepcopy

from src.brief_fact_coverage import (
    FACTS_USED_SOURCE,
    derive_facts_used_from_claim_support,
    derive_mandatory_fact_coverage_plan,
)
from tests.scientific_brief_test_data import valid_brief, validated_context


EXPECTED_CURRENT_REQUIRED_FACTS = {
    "regional.valid_coverage",
    "event.event_day",
    "patches.patch_count",
    "activity.active_track_count",
    "activity.active_family_count",
}


def _fact(context, fact_id: str):
    return next(
        item for item in context["fact_registry"] if item["fact_id"] == fact_id
    )


def test_current_coverage_plan_is_derived_from_active_context_state() -> None:
    plan = derive_mandatory_fact_coverage_plan(validated_context())
    assert set(plan.required_fact_ids) == EXPECTED_CURRENT_REQUIRED_FACTS
    assert len(plan.required_facts) == 5
    for entry in plan.required_facts:
        assert entry.available is True
        assert entry.allowed_in_brief is True
        assert entry.value_available is True
        assert entry.required_representation == (
            "paragraph_or_key_message_claim_support"
        )
        assert entry.claim_support_field == "supporting_fact_ids"
        assert entry.suggested_sections


def test_plan_uses_context_fact_ids_instead_of_universal_literal_ids() -> None:
    context = deepcopy(validated_context())
    original = "regional.valid_coverage"
    replacement = "quality.context_supplied_coverage"
    context["quality"]["valid_coverage"]["fact_id"] = replacement
    _fact(context, original)["fact_id"] = replacement

    plan = derive_mandatory_fact_coverage_plan(context)

    assert replacement in plan.required_fact_ids
    assert original not in plan.required_fact_ids


def test_inactive_or_unavailable_candidates_are_not_required() -> None:
    context = deepcopy(validated_context())
    context["univariate_event"]["active"] = False
    patch = _fact(context, "patches.patch_count")
    patch["status"] = "unavailable"
    patch["value"] = None

    plan = derive_mandatory_fact_coverage_plan(context)

    assert "event.event_day" not in plan.required_fact_ids
    assert "patches.patch_count" not in plan.required_fact_ids
    excluded = {entry.fact_id: entry for entry in plan.excluded_candidates}
    assert excluded["patches.patch_count"].exclusion_reason == "unavailable"


def test_zero_is_an_available_mandatory_operational_value() -> None:
    context = deepcopy(validated_context())
    for fact_id in ("patches.patch_count", "activity.active_family_count"):
        _fact(context, fact_id)["value"] = 0

    plan = derive_mandatory_fact_coverage_plan(context)
    entries = {entry.fact_id: entry for entry in plan.required_facts}

    assert entries["patches.patch_count"].value == 0
    assert entries["patches.patch_count"].value_available is True
    assert entries["patches.patch_count"].valid_zero is True
    assert entries["activity.active_family_count"].value == 0
    assert entries["activity.active_family_count"].available is True
    assert entries["activity.active_family_count"].valid_zero is True


def test_disallowed_and_suppressed_candidates_follow_existing_policy() -> None:
    context = deepcopy(validated_context())
    _fact(context, "patches.patch_count")["allowed_in_brief"] = False
    context["quality_flags"].append(
        {
            "flag_id": "suppress_active_families",
            "severity": "warning",
            "section": "spatiotemporal_activity",
            "message_code": "suppress_active_families",
            "affected_fact_ids": ["activity.active_family_count"],
            "suppress_dependent_claims": True,
        }
    )

    plan = derive_mandatory_fact_coverage_plan(context)
    excluded = {entry.fact_id: entry for entry in plan.excluded_candidates}

    assert excluded["patches.patch_count"].exclusion_reason == (
        "not_allowed_in_brief"
    )
    assert excluded["activity.active_family_count"].exclusion_reason == (
        "suppressed_by_quality_control"
    )


def test_plan_preserves_authoritative_value_unit_precision_and_category() -> None:
    plan = derive_mandatory_fact_coverage_plan(validated_context())
    entries = {entry.fact_id: entry for entry in plan.required_facts}
    event_day = entries["event.event_day"]

    assert event_day.value == 69
    assert event_day.unit == "day"
    assert event_day.precision == 3
    assert event_day.fact_category == "univariate_event"
    assert event_day.suggested_sections == ("event_status",)
    assert event_day.represented_in_claim_support == "required"


def test_facts_used_is_derived_from_paragraph_claim_support_only() -> None:
    payload = {
        "executive_summary": {
            "paragraphs": [
                {
                    "text": "Scientific text remains unchanged.",
                    "supporting_fact_ids": ["regional.mean_sst_c"],
                    "claim_types": ["observation"],
                }
            ]
        },
        "facts_used": ["model.supplied.index"],
    }
    original_text = payload["executive_summary"]["paragraphs"][0]["text"]
    original_claim_types = list(
        payload["executive_summary"]["paragraphs"][0]["claim_types"]
    )

    audit = derive_facts_used_from_claim_support(payload)

    assert payload["facts_used"] == ["regional.mean_sst_c"]
    assert audit.facts_used_source == FACTS_USED_SOURCE
    assert payload["executive_summary"]["paragraphs"][0]["text"] == original_text
    assert (
        payload["executive_summary"]["paragraphs"][0]["claim_types"]
        == original_claim_types
    )


def test_facts_used_is_derived_from_key_message_and_limitation_support() -> None:
    payload = {
        "key_messages": [
            {
                "supporting_fact_ids": ["event.event_day"],
                "claim_types": ["event_description"],
            }
        ],
        "limitations": [
            {"supporting_fact_ids": ["regional.valid_coverage"]}
        ],
        "facts_used": [],
    }

    audit = derive_facts_used_from_claim_support(payload)

    assert payload["facts_used"] == [
        "event.event_day",
        "regional.valid_coverage",
    ]
    assert audit.claim_support_locations_inspected == (
        "key_messages[0].supporting_fact_ids",
        "limitations[0].supporting_fact_ids",
    )


def test_facts_used_deduplicates_and_uses_registry_compatible_order() -> None:
    payload = valid_brief("en")
    payload["key_messages"][0]["supporting_fact_ids"].extend(
        ["regional.valid_coverage", "event.event_day", "event.event_day"]
    )
    payload["facts_used"] = ["not", "authoritative"]

    audit = derive_facts_used_from_claim_support(payload)

    assert payload["facts_used"] == sorted(set(payload["facts_used"]))
    assert payload["facts_used"] == list(audit.fact_ids)
    assert payload["facts_used"].count("event.event_day") == 1


def test_absent_support_id_is_not_added_and_unknown_support_is_not_dropped() -> None:
    payload = {
        "executive_summary": {
            "paragraphs": [
                {
                    "supporting_fact_ids": ["unknown.fact"],
                    "claim_types": ["observation"],
                }
            ]
        },
        "facts_used": ["event.event_day"],
    }

    derive_facts_used_from_claim_support(payload)

    assert payload["facts_used"] == ["unknown.fact"]
    assert "event.event_day" not in payload["facts_used"]
