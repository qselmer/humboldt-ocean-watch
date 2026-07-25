from copy import deepcopy

import pytest

from src.brief_numeric_validation import validate_numeric_claims
from tests.scientific_brief_test_data import valid_brief, validated_context


def _numeric_payload(text: str, facts: list[str]):
    payload = deepcopy(valid_brief("en"))
    paragraph = payload["regional_state"]["paragraphs"][0]
    paragraph["text"] = text
    paragraph["supporting_fact_ids"] = facts
    return payload


@pytest.mark.parametrize(
    ("text", "facts"),
    [
        ("Mean anomaly was 3.944 °C.", ["regional.mean_anomaly_c"]),
        ("Mean anomaly was 3.94 °C.", ["regional.mean_anomaly_c"]),
        ("Valid coverage was 95.4%.", ["regional.valid_coverage"]),
        ("The analysis date was 19 July 2026.", []),
        ("The 7-day anomaly change was 0.364 °C.", ["recent.7d.change_mean_anomaly_c"]),
    ],
)
def test_supported_numeric_transformations(text: str, facts: list[str]) -> None:
    assert validate_numeric_claims(_numeric_payload(text, facts), validated_context()) == []


@pytest.mark.parametrize(
    ("text", "facts"),
    [
        ("Mean anomaly was 88.8 °C.", ["regional.mean_anomaly_c"]),
        ("Valid coverage was 50%.", ["regional.valid_coverage"]),
        ("The derived anomaly was 7.89 °C.", ["regional.mean_anomaly_c", "regional.maximum_anomaly_c"]),
        ("Mean anomaly covered 3.944 km.", ["regional.mean_anomaly_c"]),
        ("The evidence score means 90% probability.", ["representativeness.evidence_score"]),
    ],
)
def test_unsupported_numbers_arithmetic_units_and_probability_fail(text: str, facts: list[str]) -> None:
    issues = validate_numeric_claims(_numeric_payload(text, facts), validated_context())
    assert any(issue["code"] == "unsupported_numeric_claim" for issue in issues)


def test_temperature_over_window_is_not_misread_as_compound_unit() -> None:
    payload = _numeric_payload(
        "Mean anomaly was 3.944 °C over the 7-day window.",
        ["regional.mean_anomaly_c"],
    )
    assert validate_numeric_claims(payload, validated_context()) == []
