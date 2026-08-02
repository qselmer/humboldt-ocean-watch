from __future__ import annotations

import ast
import inspect
from pathlib import Path

import yaml

from src.event_dashboard import render_thermal_events_tab


ROOT = Path(__file__).resolve().parents[1]


def _config() -> dict:
    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))


def _app_source() -> str:
    return (ROOT / "app.py").read_text(encoding="utf-8")


def test_interface_configuration_is_english_interpretability_first() -> None:
    interface = _config()["interface"]
    assert interface == {
        "language": "en",
        "mode": "interpretability_first",
        "progressive_disclosure": True,
        "show_internal_ids": False,
        "show_language_selector": False,
        "show_view_mode_selector": False,
    }
    assert "available_languages" not in interface
    assert "default_language" not in interface
    assert "default_view_mode" not in interface


def test_brief_languages_and_nino12_domain_are_unchanged() -> None:
    config = _config()
    assert config["brief_generation"]["default_language"] == "both"
    assert config["region"] == {
        "longitude": [-90.0, -80.0],
        "latitude": [-10.0, 0.0],
    }


def test_eight_tabs_remain_in_the_required_order() -> None:
    tree = ast.parse(_app_source())
    assigned_lists: dict[str, list[str]] = {}
    for node in ast.walk(tree):
        if (
            isinstance(node, ast.Assign)
            and len(node.targets) == 1
            and isinstance(node.targets[0], ast.Name)
            and isinstance(node.value, ast.List)
        ):
            values = [item.value for item in node.value.elts if isinstance(item, ast.Constant)]
            assigned_lists[node.targets[0].id] = values
    assert assigned_lists["tab_labels"] == [
        "Overview",
        "Maps",
        "Time series",
        "Spatial behaviour",
        "Quality and representativeness",
        "Thermal events",
        "Data and methods",
        "Export",
    ]


def test_required_controls_and_progressive_disclosure_are_present() -> None:
    source = _app_source()
    for label in (
        "Analysis date",
        "Time-series window",
        "Warm-anomaly threshold (°C)",
        "Persistence window",
        "Evidence and supporting diagnostics",
        "Technical identifiers",
        "Map interpretation and limitations",
        "Spatial diagnostic details",
    ):
        assert label in source
    assert "render_help_sidebar" not in source
    assert "is_advanced" not in source


def test_dashboard_retains_all_scientific_and_export_surfaces() -> None:
    source = _app_source()
    for symbol in (
        "diagnose_date(",
        "build_metrics_table(",
        "plot_spatial_field(",
        "temporal_chart(",
        "profile_chart(",
        "classify_representativeness(",
        "qc_report",
        "render_thermal_events_tab(",
        "plot_geographic_foundation(",
        "dumps_json_safe(",
    ):
        assert symbol in source


def test_no_legacy_spanish_interface_labels_are_rendered() -> None:
    source = _app_source()
    for obsolete_label in (
        "Idioma",
        "Español",
        "Modo básico",
        "Modo avanzado",
        "Ayuda y guía",
        "Fecha de análisis",
    ):
        assert obsolete_label not in source


def test_event_dashboard_has_one_fixed_interface_contract() -> None:
    parameters = inspect.signature(render_thermal_events_tab).parameters
    assert "language" not in parameters
    assert "view_mode" not in parameters
    assert "show_internal_ids" not in parameters
