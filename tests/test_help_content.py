import pytest

from src.help_content import (
    event_concept_hierarchy,
    get_context_help,
    metric_tooltip,
    representativeness_definition,
    scientific_disclaimer,
)


@pytest.mark.parametrize(
    "tab",
    [
        "Overview", "Maps", "Time series", "Spatial behaviour",
        "Quality and representativeness", "Thermal events", "Data and methods", "Export",
    ],
)
def test_each_tab_has_six_contextual_sections(tab: str) -> None:
    assert [section.key for section in get_context_help(tab, "es")] == [
        "quick_start", "what_seeing", "key_concepts", "controls", "interpretation", "limitations"
    ]


def test_contextual_help_selects_only_requested_language() -> None:
    spanish = get_context_help("Thermal events", "es")
    english = get_context_help("Thermal events", "en")
    assert spanish[0].title == "Inicio rápido"
    assert english[0].title == "Quick start"
    assert spanish[0].body != english[0].body


def test_english_context_help_uses_progressive_disclosure_not_view_modes() -> None:
    english_text = " ".join(
        section.body
        for tab in (
            "Overview", "Maps", "Time series", "Spatial behaviour",
            "Quality and representativeness", "Thermal events",
            "Data and methods", "Export",
        )
        for section in get_context_help(tab, "en")
    )
    assert "Advanced mode" not in english_text
    assert "Basic mode" not in english_text


def test_representativeness_definitions_are_bilingual_and_nominal() -> None:
    assert representativeness_definition("insufficient_coverage", "es").startswith("Cobertura insuficiente")
    assert representativeness_definition("strong_coherent", "en").startswith("Strong regional signal")


def test_event_hierarchy_keeps_concepts_distinct() -> None:
    spanish = " ".join(event_concept_hierarchy("es"))
    english = " ".join(event_concept_hierarchy("en"))
    assert "evento univariado != parche diario != track != familia de eventos" in spanish.lower()
    assert "univariate event != daily patch != track != event family" in english.lower()
    assert "parcelas individuales de agua" in spanish
    assert "water parcels" in english


@pytest.mark.parametrize(
    "metric",
    [
        "evidence_score", "valid_coverage", "mean_anomaly", "spatial_standard_deviation",
        "sign_coherence", "signal_heterogeneity_ratio", "positive_fraction",
        "negative_fraction", "neutral_fraction", "dominant_patch_fraction", "track",
        "event_family", "iou", "link_score", "cumulative_severity", "trajectory_length",
        "tortuosity", "expansion_rate", "contraction_rate",
    ],
)
def test_every_required_tooltip_is_bilingual(metric: str) -> None:
    assert metric_tooltip(metric, "es")
    assert metric_tooltip(metric, "en")
    assert metric_tooltip(metric, "es") != metric_tooltip(metric, "en")


def test_scientific_disclaimer_is_exact_in_both_languages() -> None:
    assert scientific_disclaimer("es") == (
        "Este producto es experimental y no constituye una clasificación oficial de la magnitud de El Niño Costero."
    )
    assert scientific_disclaimer("en") == (
        "This is an experimental product and does not constitute an official classification of Coastal El Niño magnitude."
    )
