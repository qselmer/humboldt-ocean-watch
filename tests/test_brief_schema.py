from src.brief_schema import (
    CONTEXT_STATUSES,
    CONTROLLED_UNITS,
    SCHEMA_VERSION,
    TOP_LEVEL_SECTIONS,
    new_context,
)
from src.export_utils import dumps_json_safe
from src.i18n import human_label


def test_required_top_level_sections_and_version() -> None:
    context = new_context("1.0.0")
    assert tuple(context) == TOP_LEVEL_SECTIONS
    assert context["schema_version"] == SCHEMA_VERSION


def test_schema_rejects_unsupported_version() -> None:
    try:
        new_context("2.0.0")
    except ValueError as exc:
        assert "Unsupported brief schema" in str(exc)
    else:
        raise AssertionError("unsupported schema version was accepted")


def test_statuses_units_and_bilingual_codes_are_stable() -> None:
    assert {"valid", "warning", "invalid", "unavailable"} <= CONTEXT_STATUSES
    assert {"degC", "fraction", "km2", "degC_km2_day"} <= CONTROLLED_UNITS
    assert human_label("strong_heterogeneous", "es") != human_label("strong_heterogeneous", "en")


def test_empty_schema_serialization_is_deterministic() -> None:
    assert dumps_json_safe(new_context()) == dumps_json_safe(new_context())
