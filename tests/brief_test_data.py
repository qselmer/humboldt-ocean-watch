"""Deterministic cached-product fixtures for brief-context tests."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import yaml

from src.brief_context import BriefBuildInputs
from src.utils import load_config


def synthetic_config(root: Path) -> tuple[dict, Path]:
    config = deepcopy(load_config())
    config["brief_context"]["output_directory"] = "outputs/briefs"
    path = root / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False, allow_unicode=True), encoding="utf-8")
    return config, path


def write_products(
    root: Path,
    *,
    active_event: bool = True,
    patch_count: int = 2,
    include_optional: bool = True,
) -> BriefBuildInputs:
    analytics = root / "outputs" / "analytics"
    processed = root / "data" / "processed"
    analytics.mkdir(parents=True, exist_ok=True)
    processed.mkdir(parents=True, exist_ok=True)
    dates = pd.to_datetime(["2026-01-01", "2026-01-02"])
    metrics = pd.DataFrame(
        {
            "date": dates,
            "mean_sst_c": [24.0, 24.3],
            "seven_day_mean_sst_c": [24.0, 24.15],
            "mean_sst_anomaly_c": [0.8, 1.2],
            "mean_standardized_anomaly": [1.0, 1.5],
            "seven_day_mean_anomaly_c": [0.8, 1.0],
            "maximum_anomaly_c": [2.0, 2.6],
            "p90_anomaly_c": [1.5, 2.0],
            "mean_daily_sst_change_c": [None, 0.3],
            "seven_day_sst_change_c": [None, None],
            "area_anomaly_ge_1c_percent": [30.0, 60.0],
            "area_anomaly_ge_2c_percent": [5.0, 20.0],
            "area_anomaly_ge_3c_percent": [0.0, 2.0],
            "area_above_climatological_p90_percent": [20.0, 40.0],
            "area_zscore_ge_2_percent": [2.0, 10.0],
            "valid_data_coverage_percent": [95.0, 94.0],
            "warm_centroid_longitude": [-85.0, -84.8],
            "warm_centroid_latitude": [-5.0, -4.9],
            "climatology_method": ["daily_smoothed", "daily_smoothed"],
            "climatology_reference_period": ["1991-01-01 to 2020-12-31"] * 2,
            "climatology_fallback_used": [False, False],
        }
    )
    daily_metrics = processed / "nino12_daily_metrics.parquet"
    metrics.to_parquet(daily_metrics, index=False)
    qc = {
        "data_mode": "Copernicus cached data",
        "units": "degrees_Celsius",
        "start_date": "2026-01-01T00:00:00",
        "end_date": "2026-01-02T00:00:00",
        "number_of_dates": 2,
        "duplicated_dates": [],
        "temporal_interval_summary": {"regular": True, "minimum_days": 1, "median_days": 1, "maximum_days": 1},
        "expected_frequency": "1D",
        "spatial_dimensions": {"latitude": 2, "longitude": 2},
        "spatial_bounds": {"latitude": [-10.0, 0.0], "longitude": [-90.0, -80.0]},
        "valid_data_fraction": 0.94,
        "weighted_spatial_coverage": 0.94,
        "constant_field_dates": [],
        "all_nan_dates": [],
        "warnings": [],
        "errors": [],
        "overall_status": "valid",
    }
    qc_path = analytics / "qc_report.json"
    qc_path.write_text(json.dumps(qc), encoding="utf-8")
    rep = pd.DataFrame(
        {
            "date": dates,
            "class": ["weak_homogeneous", "strong_heterogeneous"],
            "status": ["valid", "valid"],
            "triggered_rule": ["weak_signal", "strong_signal_with_high_heterogeneity"],
            "evidence_score": [0.8, 0.9],
            "valid_coverage": [0.95, 0.94],
            "weighted_mean_anomaly": [0.8, 1.2],
            "weighted_median_anomaly": [0.75, 1.1],
            "spatial_standard_deviation": [0.2, 0.9],
            "weighted_iqr": [0.3, 1.0],
            "positive_fraction": [0.8, 0.9],
            "negative_fraction": [0.1, 0.05],
            "neutral_fraction": [0.1, 0.05],
            "sign_coherence": [0.7, 0.85],
            "signal_heterogeneity_ratio": [4.0, 1.33],
            "mean_median_difference": [0.1, 0.1],
            "dominant_patch_fraction": [0.8, 0.7],
            "patch_density": [0.01, 0.02],
            "coexistence_index": [0.1, 0.1],
            "spatial_compensation_index": [0.05, 0.05],
            "climatology_method": ["daily_smoothed"] * 2,
            "data_mode": ["Copernicus cached data"] * 2,
        }
    )
    rep_path = analytics / "representativeness.parquet"
    rep.to_parquet(rep_path, index=False)

    paths = BriefBuildInputs(
        daily_metrics=daily_metrics,
        qc_report=qc_path,
        representativeness=rep_path,
        series_bank=analytics / "daily_series_bank.parquet",
        temporal_features=analytics / "temporal_features.parquet",
        spatial_features=analytics / "spatial_features.parquet",
        univariate_events=analytics / "univariate_events.parquet",
        univariate_flags=analytics / "univariate_event_daily_flags.parquet",
        univariate_summary=analytics / "univariate_event_summary.json",
        daily_patches=analytics / "daily_patches.parquet",
        daily_patch_summary=analytics / "daily_patch_summary.parquet",
        daily_patch_build_summary=analytics / "daily_patch_build_summary.json",
        patch_observations=analytics / "spatiotemporal_patch_observations.parquet",
        lineage_edges=analytics / "patch_lineage_edges.parquet",
        tracks=analytics / "spatiotemporal_tracks.parquet",
        event_families=analytics / "spatiotemporal_event_families.parquet",
        tracking_summary=analytics / "spatiotemporal_tracking_summary.json",
    )
    if not include_optional:
        return paths

    bank_rows = []
    for date_value in dates:
        for metric, value, unit in (
            ("sst.valid_cell_fraction", 0.95, "1"),
            ("anomaly.weighted_median", 1.1, "degrees_Celsius"),
            ("anomaly.spatial_p10", 0.1, "degrees_Celsius"),
            ("zscore.spatial_maximum", 3.0, "1"),
            ("anomaly.weighted_mad", 0.5, "degrees_Celsius"),
            ("anomaly.weighted_iqr", 1.0, "degrees_Celsius"),
            ("anomaly.p90_minus_p10", 1.9, "degrees_Celsius"),
            ("anomaly.latitudinal_gradient", 0.1, "1"),
            ("anomaly.longitudinal_gradient", 0.2, "1"),
        ):
            bank_rows.append({"date": date_value, "family": "state", "metric": metric, "value": value, "unit": unit, "status": "valid", "reason": None, "n_observations": 4, "valid_coverage": 0.95, "window_days": None, "climatology_method": "daily_smoothed", "data_mode": "Copernicus cached data"})
    pd.DataFrame(bank_rows).to_parquet(paths.series_bank, index=False)
    temporal = pd.DataFrame([
        {"analysis_end_date": dates[-1], "window_start_date": dates[0], "window_end_date": dates[-1], "window_days": 1, "source_metric": "anomaly.weighted_mean", "family": "trend", "metric": "ols_slope", "value": 0.4, "unit": "degrees_Celsius day-1", "status": "valid", "reason": None, "n_observations": 2, "valid_coverage": 1.0}
    ])
    temporal.to_parquet(paths.temporal_features, index=False)
    spatial = pd.DataFrame([
        {"date": dates[-1], "variable": "anomaly", "family": "spatial_autocorrelation", "metric": "moran_i_queen", "value": 0.5, "unit": "1", "status": "valid", "reason": None, "n_observations": 4, "valid_coverage": 0.94, "threshold": None, "connectivity": "queen"},
        {"date": dates[-1], "variable": "anomaly", "family": "texture", "metric": "local_spatial_standard_deviation", "value": 0.4, "unit": "degrees_Celsius", "status": "valid", "reason": None, "n_observations": 4, "valid_coverage": 0.94, "threshold": None, "connectivity": None},
        {"date": dates[-1], "variable": "anomaly", "family": "gradient", "metric": "thermal_gradient_magnitude", "value": 0.2, "unit": "1", "status": "valid", "reason": None, "n_observations": 4, "valid_coverage": 0.94, "threshold": None, "connectivity": None},
    ])
    spatial.to_parquet(paths.spatial_features, index=False)

    event_id = "regional-mean-sst-E0001" if active_event else None
    events = pd.DataFrame([
        {"event_id": "regional-mean-sst-E0001", "source_variable": "regional_mean_sst", "threshold_type": "daily_climatological", "threshold_description": "daily P90", "direction": "above", "start_date": dates[0], "end_date": dates[-1], "peak_date": dates[-1], "duration_calendar_days": 2, "duration_observed_days": 2, "mean_valid_coverage": 0.94, "mean_intensity": 0.4, "maximum_intensity": 0.6, "cumulative_intensity": 0.8, "severity_class": "moderate thermal exceedance", "status": "valid", "reason": None, "climatology_method": "daily_smoothed", "data_mode": "Copernicus cached data"}
    ])
    events.to_parquet(paths.univariate_events, index=False)
    flags = pd.DataFrame([
        {"date": dates[0], "source_variable": "regional_mean_sst", "value": 24.0, "threshold": 23.8, "exceeds_threshold": active_event, "observed_interruption": False, "event_id": event_id if active_event else None, "event_day": 1 if active_event else None, "event_age_calendar_days": 1 if active_event else None, "intensity": 0.2 if active_event else None, "valid_coverage": 0.95, "status": "valid", "reason": None},
        {"date": dates[-1], "source_variable": "regional_mean_sst", "value": 24.3, "threshold": 23.7, "exceeds_threshold": active_event, "observed_interruption": False, "event_id": event_id, "event_day": 2 if active_event else None, "event_age_calendar_days": 2 if active_event else None, "intensity": 0.6 if active_event else None, "valid_coverage": 0.94, "status": "valid", "reason": None},
    ])
    flags.to_parquet(paths.univariate_flags, index=False)
    paths.univariate_summary.write_text(json.dumps({"number_of_events": int(active_event)}), encoding="utf-8")

    patch_rows = []
    for patch_id in range(1, patch_count + 1):
        patch_rows.append({
            "date": dates[-1], "patch_id": patch_id, "source_variable": "anomaly", "threshold_type": "fixed", "threshold_value": 2.0, "direction": "above", "connectivity": "queen", "cell_count": 4, "area_km2": float(100 * patch_id), "area_fraction_of_threshold_total": patch_id / max(sum(range(1, patch_count + 1)), 1), "centroid_latitude": -5.0 + patch_id / 10, "centroid_longitude": -85.0 + patch_id / 10, "mean_source_value": 2.5, "maximum_source_value": 3.0, "mean_exceedance": 0.5, "maximum_exceedance": 1.0, "compactness": 0.8, "elongation": 1.2, "touches_north_boundary": patch_id == 1, "touches_south_boundary": False, "touches_east_boundary": False, "touches_west_boundary": False, "valid_coverage": 0.94, "climatology_method": "daily_smoothed", "data_mode": "Copernicus cached data", "status": "valid", "reason": None,
        })
    patch_table = pd.DataFrame(patch_rows, columns=["date", "patch_id", "source_variable", "threshold_type", "threshold_value", "direction", "connectivity", "cell_count", "area_km2", "area_fraction_of_threshold_total", "centroid_latitude", "centroid_longitude", "mean_source_value", "maximum_source_value", "mean_exceedance", "maximum_exceedance", "compactness", "elongation", "touches_north_boundary", "touches_south_boundary", "touches_east_boundary", "touches_west_boundary", "valid_coverage", "climatology_method", "data_mode", "status", "reason"])
    patch_table.to_parquet(paths.daily_patches, index=False)
    total_area = float(patch_table.area_km2.sum()) if patch_count else 0.0
    summary = pd.DataFrame([{"date": dates[-1], "patch_count": patch_count, "total_patch_area_km2": total_area, "threshold_area_fraction": 0.2 if patch_count else 0.0, "largest_patch_area_km2": float(patch_table.area_km2.max()) if patch_count else 0.0, "largest_patch_fraction": float(patch_table.area_km2.max() / total_area) if patch_count else 0.0, "mean_patch_area_km2": float(patch_table.area_km2.mean()) if patch_count else 0.0, "fragmentation_index": 0.2 if patch_count > 1 else 0.0, "dominant_patch_fraction": 0.7 if patch_count else 0.0, "maximum_patch_intensity": 1.0 if patch_count else None, "area_weighted_centroid_latitude": -4.8 if patch_count else None, "area_weighted_centroid_longitude": -84.8 if patch_count else None, "valid_coverage": 0.94, "status": "valid", "reason": None}])
    summary.to_parquet(paths.daily_patch_summary, index=False)
    paths.daily_patch_build_summary.write_text(json.dumps({"overall_status": "valid"}), encoding="utf-8")

    observations = []
    for patch_id in range(1, patch_count + 1):
        observations.append({"node_id": f"2026-01-02::P{patch_id:06d}", "date": dates[-1], "local_patch_id": patch_id, "track_id": f"TRK{patch_id:06d}", "event_family_id": "FAM000001", "node_event_type": "termination", "area_km2": float(100 * patch_id), "valid_coverage": 0.94, "climatology_method": "daily_smoothed", "data_mode": "Copernicus cached data", "status": "valid", "reason": None})
    pd.DataFrame(observations, columns=["node_id", "date", "local_patch_id", "track_id", "event_family_id", "node_event_type", "area_km2", "valid_coverage", "climatology_method", "data_mode", "status", "reason"]).to_parquet(paths.patch_observations, index=False)
    pd.DataFrame(columns=["predecessor_node_id", "successor_node_id", "predecessor_date", "successor_date", "status"]).to_parquet(paths.lineage_edges, index=False)
    track_rows = []
    for patch_id in range(1, patch_count + 1):
        track_rows.append({"track_id": f"TRK{patch_id:06d}", "event_family_id": "FAM000001", "start_date": dates[0], "end_date": dates[-1], "observed_days": 2, "duration_calendar_days": 2, "maximum_area_km2": float(100 * patch_id), "cumulative_severity": float(50 * patch_id), "trajectory_length_km": 10.0, "net_displacement_km": 8.0, "mean_speed_km_per_day": 8.0, "maximum_speed_km_per_day": 10.0, "touched_north_boundary": patch_id == 1, "status": "valid", "reason": None})
    pd.DataFrame(track_rows, columns=["track_id", "event_family_id", "start_date", "end_date", "observed_days", "duration_calendar_days", "maximum_area_km2", "cumulative_severity", "trajectory_length_km", "net_displacement_km", "mean_speed_km_per_day", "maximum_speed_km_per_day", "touched_north_boundary", "status", "reason"]).to_parquet(paths.tracks, index=False)
    family_rows = [] if patch_count == 0 else [{"event_family_id": "FAM000001", "start_date": dates[0], "end_date": dates[-1], "duration_calendar_days": 2, "track_count": patch_count, "unique_patch_observation_count": patch_count, "split_count": 0, "merge_count": 0, "complex_branch_count": 0, "maximum_total_daily_area_km2": total_area, "family_area_days_km2_days": total_area, "cumulative_severity": 50.0, "family_trajectory_length_km": 10.0, "maximum_family_extent_km": 20.0, "boundary_contact_day_count": 1, "climatology_method": "daily_smoothed", "data_mode": "Copernicus cached data", "status": "valid", "reason": None}]
    pd.DataFrame(family_rows, columns=["event_family_id", "start_date", "end_date", "duration_calendar_days", "track_count", "unique_patch_observation_count", "split_count", "merge_count", "complex_branch_count", "maximum_total_daily_area_km2", "family_area_days_km2_days", "cumulative_severity", "family_trajectory_length_km", "maximum_family_extent_km", "boundary_contact_day_count", "climatology_method", "data_mode", "status", "reason"]).to_parquet(paths.event_families, index=False)
    paths.tracking_summary.write_text(json.dumps({"overall_status": "valid"}), encoding="utf-8")
    # Representativeness is downstream of spatial features.  Keep fixture
    # modification times consistent with that documented provenance relation.
    rep_path.touch()
    return paths
