import os

import pandas as pd
import pytest

from src.brief_context import build_brief_context, resolve_analysis_date
from src.brief_validation import validate_brief_context
from src.utils import load_config
from tests.brief_test_data import synthetic_config, write_products


def test_explicit_latest_and_invalid_date_selection() -> None:
    frame = pd.DataFrame({"date": pd.to_datetime(["2026-01-01", "2026-01-02"]), "mean_sst_c": [24.0, 24.2]})
    assert resolve_analysis_date(frame, "latest") == pd.Timestamp("2026-01-02")
    assert resolve_analysis_date(frame, "2026-01-01") == pd.Timestamp("2026-01-01")
    with pytest.raises(ValueError, match="unavailable"):
        resolve_analysis_date(frame, "2026-01-03")
    with pytest.raises(ValueError, match="ISO date"):
        resolve_analysis_date(frame, "02-01-2026")


def test_duplicate_dates_are_rejected() -> None:
    frame = pd.DataFrame({"date": pd.to_datetime(["2026-01-01", "2026-01-01"]), "mean_sst_c": [24.0, 24.2]})
    with pytest.raises(ValueError, match="duplicate dates"):
        resolve_analysis_date(frame, "latest")


def test_nonfinite_sst_is_not_a_valid_analysis_date() -> None:
    frame = pd.DataFrame(
        {"date": pd.to_datetime(["2026-01-01", "2026-01-02"]), "mean_sst_c": [24.0, float("inf")]}
    )
    assert resolve_analysis_date(frame, "latest") == pd.Timestamp("2026-01-01")
    with pytest.raises(ValueError, match="lacks valid regional SST"):
        resolve_analysis_date(frame, "2026-01-02")


def test_core_context_and_missing_optional_products(tmp_path) -> None:
    config, _ = synthetic_config(tmp_path)
    inputs = write_products(tmp_path, include_optional=False)
    context, _ = build_brief_context(config, analysis_date="latest", root=tmp_path, inputs=inputs)
    assert context["analysis"]["requested_analysis_date"] == "latest"
    assert context["analysis"]["resolved_analysis_date"] == "2026-01-02"
    assert context["regional_state"]["mean_sst_c"]["value"] == 24.3
    assert context["climatology"]["method"] == "daily_smoothed"
    assert context["representativeness"]["class_code"] == "strong_heterogeneous"
    assert context["availability"]["temporal_features"]["available"] is False
    assert any(flag["message_code"] == "missing_optional_product" for flag in context["quality_flags"])


def test_active_event_patches_tracks_and_families(tmp_path) -> None:
    config, _ = synthetic_config(tmp_path)
    inputs = write_products(tmp_path, active_event=True, patch_count=2)
    context, _ = build_brief_context(config, analysis_date="latest", root=tmp_path, inputs=inputs)
    assert context["univariate_event"]["active"] is True
    assert context["univariate_event"]["event_day"]["value"] == 2
    assert context["daily_patches"]["patch_count"]["value"] == 2
    assert len(context["daily_patches"]["top_patches"]) == 2
    assert context["spatiotemporal_activity"]["active_track_count"]["value"] == 2
    assert context["spatiotemporal_activity"]["active_family_count"]["value"] == 1
    assert context["spatiotemporal_activity"]["top_tracks"]
    assert context["spatiotemporal_activity"]["top_event_families"]
    report = validate_brief_context(context, maximum_payload_kb=150)
    assert report.result in {"passed", "warning"}


def test_valid_absence_of_event_and_patches_is_not_an_error(tmp_path) -> None:
    config, _ = synthetic_config(tmp_path)
    inputs = write_products(tmp_path, active_event=False, patch_count=0)
    context, _ = build_brief_context(config, analysis_date="latest", root=tmp_path, inputs=inputs)
    assert context["univariate_event"]["active"] is False
    assert context["daily_patches"]["patch_count"]["value"] == 0
    flags = {flag["flag_id"]: flag for flag in context["quality_flags"]}
    assert flags["no_active_event"]["severity"] == "info"
    assert flags["no_daily_patches"]["severity"] == "info"


def test_incompatible_climatology_and_data_mode_are_rejected(tmp_path) -> None:
    config, _ = synthetic_config(tmp_path)
    inputs = write_products(tmp_path, include_optional=False)
    rep = pd.read_parquet(inputs.representativeness)
    rep["climatology_method"] = "monthly"
    rep.to_parquet(inputs.representativeness, index=False)
    with pytest.raises(ValueError, match="incompatible climatology"):
        build_brief_context(config, analysis_date="latest", root=tmp_path, inputs=inputs)


def test_incompatible_data_mode_is_rejected(tmp_path) -> None:
    config, _ = synthetic_config(tmp_path)
    inputs = write_products(tmp_path)
    patches = pd.read_parquet(inputs.daily_patches)
    patches["data_mode"] = "Synthetic demonstration data"
    patches.to_parquet(inputs.daily_patches, index=False)
    with pytest.raises(ValueError, match="incompatible data modes"):
        build_brief_context(config, analysis_date="latest", root=tmp_path, inputs=inputs)


def test_patch_count_mismatch_is_explicit_and_fails_validation(tmp_path) -> None:
    config, _ = synthetic_config(tmp_path)
    inputs = write_products(tmp_path, patch_count=2)
    summary = pd.read_parquet(inputs.daily_patch_summary)
    summary.loc[:, "patch_count"] = 3
    summary.to_parquet(inputs.daily_patch_summary, index=False)
    context, _ = build_brief_context(config, analysis_date="latest", root=tmp_path, inputs=inputs)
    assert context["daily_patches"]["product_available"] is False
    assert any(flag["flag_id"] == "patch_count_mismatch" for flag in context["quality_flags"])
    assert validate_brief_context(context).result == "failed"


@pytest.mark.parametrize(
    ("catalogue", "id_column", "expected_flag"),
    [
        ("tracks", "track_id", "missing_referenced_track"),
        ("event_families", "event_family_id", "missing_referenced_family"),
    ],
)
def test_missing_referenced_track_or_family_is_detected(
    tmp_path, catalogue: str, id_column: str, expected_flag: str,
) -> None:
    config, _ = synthetic_config(tmp_path)
    inputs = write_products(tmp_path, patch_count=2)
    path = getattr(inputs, catalogue)
    table = pd.read_parquet(path)
    table = table.loc[table[id_column].astype(str) != table[id_column].astype(str).iloc[0]]
    table.to_parquet(path, index=False)
    context, _ = build_brief_context(config, analysis_date="latest", root=tmp_path, inputs=inputs)
    assert any(flag["flag_id"] == expected_flag for flag in context["quality_flags"])
    assert validate_brief_context(context).result == "failed"


def test_stale_optional_product_is_suppressed_with_warning(tmp_path) -> None:
    config, _ = synthetic_config(tmp_path)
    inputs = write_products(tmp_path)
    upstream_time = inputs.series_bank.stat().st_mtime
    os.utime(inputs.temporal_features, (upstream_time - 60, upstream_time - 60))
    context, _ = build_brief_context(config, analysis_date="latest", root=tmp_path, inputs=inputs)
    assert context["availability"]["temporal_features"]["available"] is False
    assert any(
        flag["flag_id"] == "stale_analytical_output.temporal_features"
        for flag in context["quality_flags"]
    )
    assert validate_brief_context(context).result == "warning"


def test_stale_core_product_fails_validation(tmp_path) -> None:
    config, _ = synthetic_config(tmp_path)
    inputs = write_products(tmp_path)
    upstream_time = inputs.spatial_features.stat().st_mtime
    os.utime(inputs.representativeness, (upstream_time - 60, upstream_time - 60))
    context, _ = build_brief_context(config, analysis_date="latest", root=tmp_path, inputs=inputs)
    assert context["availability"]["representativeness"]["available"] is False
    assert context["quality"]["core_inputs_valid"] is False
    assert validate_brief_context(context).result == "failed"


def test_insufficient_coverage_fallback_and_boundary_flags(tmp_path) -> None:
    config, _ = synthetic_config(tmp_path)
    inputs = write_products(tmp_path, patch_count=1)
    metrics = pd.read_parquet(inputs.daily_metrics)
    metrics.loc[metrics.index[-1], "valid_data_coverage_percent"] = 70.0
    metrics["climatology_method"] = "monthly"
    metrics["climatology_fallback_used"] = True
    metrics.to_parquet(inputs.daily_metrics, index=False)
    for path in (
        inputs.representativeness,
        inputs.univariate_events,
        inputs.daily_patches,
        inputs.patch_observations,
        inputs.event_families,
    ):
        table = pd.read_parquet(path)
        if "climatology_method" in table:
            table["climatology_method"] = "monthly"
            table.to_parquet(path, index=False)
    # Restore downstream modification times after the deliberate table edits.
    inputs.representativeness.touch()
    for path in (
        inputs.patch_observations, inputs.lineage_edges, inputs.tracks,
        inputs.event_families, inputs.tracking_summary,
    ):
        path.touch()
    context, _ = build_brief_context(config, analysis_date="latest", root=tmp_path, inputs=inputs)
    codes = {flag["message_code"] for flag in context["quality_flags"]}
    assert {"insufficient_coverage", "monthly_climatology_fallback", "track_touches_domain_boundary"} <= codes
    assert context["climatology"]["fallback_used"] is True
    assert context["climatology"]["compatibility_status"] == "warning"
