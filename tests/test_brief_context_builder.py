import json

import pandas as pd
import pytest

import scripts.build_brief_context as builder
from scripts.validate_brief_context import validate_file
from tests.brief_test_data import synthetic_config, write_products


def _overrides(inputs):
    return {name: getattr(inputs, name) for name in (
        "daily_metrics", "qc_report", "representativeness", "series_bank",
        "temporal_features", "spatial_features", "univariate_events", "univariate_flags",
        "daily_patches", "daily_patch_summary", "patch_observations", "tracks",
        "event_families", "lineage_edges",
    )}


def test_builder_writes_dated_latest_validation_and_fact_outputs(tmp_path, monkeypatch) -> None:
    _, config_path = synthetic_config(tmp_path)
    inputs = write_products(tmp_path, active_event=True, patch_count=2)
    monkeypatch.setattr(builder, "PROJECT_ROOT", tmp_path)
    artifacts = builder.build(
        config_path=config_path, analysis_date="latest",
        output_directory=tmp_path / "outputs" / "briefs",
        input_overrides=_overrides(inputs), strict=True, overwrite=False,
    )
    assert artifacts.context_path.exists()
    assert artifacts.latest_path and artifacts.latest_path.exists()
    assert artifacts.validation_path.exists()
    assert artifacts.facts_path.exists()
    assert json.loads(artifacts.latest_path.read_text(encoding="utf-8"))["analysis"]["resolved_analysis_date"] == "2026-01-02"
    facts = pd.read_parquet(artifacts.facts_path)
    assert len(facts) == len(artifacts.context["fact_registry"])
    assert facts.fact_id.is_unique


def test_previous_outputs_are_preserved_without_overwrite(tmp_path, monkeypatch) -> None:
    _, config_path = synthetic_config(tmp_path)
    inputs = write_products(tmp_path, include_optional=False)
    monkeypatch.setattr(builder, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "outputs" / "briefs"
    output.mkdir(parents=True)
    existing = output / "brief_context_2026-01-02.json"
    existing.write_text("preserve-me", encoding="utf-8")
    with pytest.raises(FileExistsError):
        builder.build(
            config_path=config_path, analysis_date="latest", output_directory=output,
            input_overrides=_overrides(inputs), strict=True, overwrite=False,
        )
    assert existing.read_text(encoding="utf-8") == "preserve-me"


def test_validation_cli_helper_returns_status_and_writes_report(tmp_path, monkeypatch) -> None:
    _, config_path = synthetic_config(tmp_path)
    inputs = write_products(tmp_path, include_optional=False)
    monkeypatch.setattr(builder, "PROJECT_ROOT", tmp_path)
    artifacts = builder.build(
        config_path=config_path, analysis_date="2026-01-02",
        output_directory=tmp_path / "outputs" / "briefs",
        input_overrides=_overrides(inputs), strict=True, overwrite=False,
    )
    report_path = tmp_path / "validation.json"
    report, destination = validate_file(
        artifacts.context_path, config_path=config_path, output_path=report_path, strict=True,
    )
    assert report["result"] in {"passed", "warning"}
    assert destination == report_path
    assert destination.exists()


def test_builder_is_offline_and_has_no_llm_dependency() -> None:
    source = (builder.PROJECT_ROOT / "scripts" / "build_brief_context.py").read_text(encoding="utf-8")
    assert "openai" not in source.lower()
    assert "copernicusmarine" not in source.lower()
    assert "requests." not in source.lower()


def test_previous_output_is_preserved_after_validation_failure(tmp_path, monkeypatch) -> None:
    _, config_path = synthetic_config(tmp_path)
    inputs = write_products(tmp_path, active_event=True)
    events = pd.read_parquet(inputs.univariate_events)
    events.loc[:, "end_date"] = pd.Timestamp("2026-01-01")
    events.to_parquet(inputs.univariate_events, index=False)
    monkeypatch.setattr(builder, "PROJECT_ROOT", tmp_path)
    output = tmp_path / "outputs" / "briefs"
    output.mkdir(parents=True)
    existing = output / "brief_context_2026-01-02.json"
    existing.write_text("previous-valid-output", encoding="utf-8")

    with pytest.raises(ValueError, match="validation failed"):
        builder.build(
            config_path=config_path, analysis_date="latest", output_directory=output,
            input_overrides=_overrides(inputs), strict=True, overwrite=True,
        )

    assert existing.read_text(encoding="utf-8") == "previous-valid-output"
