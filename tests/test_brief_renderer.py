from copy import deepcopy

import pytest

from src.brief_renderer import render_scientific_brief_markdown
from tests.scientific_brief_test_data import valid_brief


@pytest.mark.parametrize(
    ("language", "heading"),
    [("es", "## Resumen ejecutivo"), ("en", "## Executive summary")],
)
def test_renderer_is_deterministic_bilingual_and_ordered(language: str, heading: str) -> None:
    payload = valid_brief(language)
    first = render_scientific_brief_markdown(payload)
    second = render_scientific_brief_markdown(payload)
    assert first == second
    assert heading in first
    assert "1991–2020" in first
    assert first.index(heading) < first.index("## " + ("Limitaciones" if language == "es" else "Limitations"))
    assert payload["disclaimer"] in first
    assert "`regional.mean_anomaly_c`" in first
    primary_prose = first.split("## " + headings_for_facts(language), 1)[0]
    assert "mean_anomaly_c" not in primary_prose


def test_renderer_preserves_validated_units_and_has_no_raw_json() -> None:
    payload = valid_brief("en")
    payload["regional_state"]["paragraphs"][0]["text"] = "Mean anomaly was 3.944 °C and coverage was 95.4%."
    text = render_scientific_brief_markdown(payload)
    assert "3.944 °C" in text
    assert "95.4%" in text
    assert '"schema_version"' not in text


def test_renderer_rejects_invalid_schema() -> None:
    payload = deepcopy(valid_brief("en"))
    payload["unexpected"] = True
    with pytest.raises(ValueError, match="invalid structured brief"):
        render_scientific_brief_markdown(payload)


def test_renderer_rejects_non_authoritative_disclaimer() -> None:
    payload = deepcopy(valid_brief("en"))
    payload["disclaimer"] = "Experimental only."
    with pytest.raises(ValueError, match="non-authoritative disclaimer"):
        render_scientific_brief_markdown(payload)


def headings_for_facts(language: str) -> str:
    return "Identificadores de hechos de respaldo" if language == "es" else "Supporting fact IDs"
