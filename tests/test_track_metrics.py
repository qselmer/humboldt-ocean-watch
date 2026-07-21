"""Track and event-family aggregation tests."""

import numpy as np
import pandas as pd
import pytest

from src.patch_tracking import OBSERVATION_COLUMNS
from src.track_metrics import (
    FAMILY_COLUMNS,
    TRACK_COLUMNS,
    calculate_family_metrics,
    calculate_track_metrics,
)


def _observation(
    date: str,
    node: str,
    *,
    track="TRK000001",
    family="FAM000001",
    area=10.0,
    latitude=0.0,
    longitude=0.0,
    event_type="continuation",
    boundary=False,
):
    row = {column: None for column in OBSERVATION_COLUMNS}
    row.update({
        "node_id": node, "date": pd.Timestamp(date), "local_patch_id": 1,
        "track_id": track, "event_family_id": family,
        "node_event_type": event_type, "source_variable": "anomaly",
        "threshold_type": "fixed", "direction": "above", "area_km2": area,
        "centroid_latitude": latitude, "centroid_longitude": longitude,
        "mean_source_value": 3.0, "maximum_source_value": 4.0,
        "mean_exceedance": 1.0, "maximum_exceedance": 2.0,
        "parent_node_ids": [], "child_node_ids": [], "parent_track_ids": [],
        "child_track_ids": [], "predecessor_count": 0, "successor_count": 0,
        "link_crossed_temporal_gap": False, "valid_coverage": 1.0,
        "climatology_method": "daily_smoothed", "data_mode": "Copernicus cached data",
        "status": "valid", "reason": None,
        "touches_north_boundary": boundary, "touches_south_boundary": False,
        "touches_east_boundary": False, "touches_west_boundary": False,
    })
    return row


def test_track_duration_area_days_severity_movement_and_area_change() -> None:
    observations = pd.DataFrame([
        _observation("2024-01-01", "a", area=10, longitude=0, event_type="appearance"),
        _observation("2024-01-02", "b", area=20, longitude=1),
        _observation("2024-01-04", "c", area=10, longitude=2, event_type="termination", boundary=True),
    ], columns=OBSERVATION_COLUMNS)
    tracks = calculate_track_metrics(observations)
    row = tracks.iloc[0]
    assert tracks.columns.tolist() == TRACK_COLUMNS
    assert row.observed_days == 3
    assert row.duration_calendar_days == 4
    assert row.internal_gap_count == 1
    assert row.observed_area_days_km2_days == 40
    assert row.cumulative_severity == 40
    assert row.trajectory_length_km > 200
    assert row.net_displacement_km == pytest.approx(row.trajectory_length_km, rel=1e-6)
    assert row.mean_speed_km_per_day == pytest.approx(row.trajectory_length_km / 3)
    assert row.maximum_expansion_rate_km2_per_day == 10
    assert row.maximum_contraction_rate_km2_per_day == -5
    assert row.expansion_day_count == 1
    assert row.contraction_day_count == 1
    assert row.boundary_contact_day_count == 1
    assert row.trajectory_tortuosity == pytest.approx(1.0)


def test_zero_displacement_and_one_day_tracks_use_nan_safeguards() -> None:
    stationary = pd.DataFrame([
        _observation("2024-01-01", "a", event_type="appearance"),
        _observation("2024-01-02", "b", event_type="termination"),
    ], columns=OBSERVATION_COLUMNS)
    row = calculate_track_metrics(stationary).iloc[0]
    assert row.net_displacement_km == 0
    assert np.isnan(row.trajectory_tortuosity)
    assert row.status == "warning"
    one_day = pd.DataFrame([
        _observation("2024-01-01", "c", event_type="isolated_single_day")
    ], columns=OBSERVATION_COLUMNS)
    row = calculate_track_metrics(one_day).iloc[0]
    assert row.observed_days == 1
    assert np.isnan(row.mean_speed_km_per_day)
    assert np.isnan(row.mean_area_change_km2_per_day)
    assert row.status == "warning"


def test_family_with_split_counts_unique_observations_and_daily_centroid() -> None:
    observations = pd.DataFrame([
        _observation("2024-01-01", "a", track="TRK000001", area=20, longitude=0, event_type="split_parent"),
        _observation("2024-01-02", "b", track="TRK000002", area=10, longitude=-1, event_type="split_child"),
        _observation("2024-01-02", "c", track="TRK000003", area=30, longitude=1, event_type="split_child"),
    ], columns=OBSERVATION_COLUMNS)
    observations.loc[1, "local_patch_id"] = 1
    observations.loc[2, "local_patch_id"] = 2
    families = calculate_family_metrics(observations)
    row = families.iloc[0]
    assert families.columns.tolist() == FAMILY_COLUMNS
    assert row.track_count == 3
    assert row.unique_patch_observation_count == 3
    assert row.observed_patch_days == 3
    assert row.split_count == 1
    assert row.maximum_simultaneous_patch_count == 2
    assert row.family_area_days_km2_days == 60
    assert row.maximum_total_daily_area_km2 == 40
    assert row.end_centroid_longitude == pytest.approx(0.5, abs=0.01)
    assert row.maximum_family_extent_km > 200


def test_family_merge_and_complex_counts_are_explicit() -> None:
    observations = pd.DataFrame([
        _observation("2024-01-01", "a", track="TRK000001", event_type="merge_parent"),
        _observation("2024-01-01", "b", track="TRK000002", event_type="complex_branch"),
        _observation("2024-01-02", "c", track="TRK000003", event_type="merge_child"),
    ], columns=OBSERVATION_COLUMNS)
    observations.loc[1, "local_patch_id"] = 2
    row = calculate_family_metrics(observations).iloc[0]
    assert row.merge_count == 1
    assert row.complex_branch_count == 1


def test_family_metrics_reject_duplicate_patch_observations() -> None:
    observations = pd.DataFrame([
        _observation("2024-01-01", "duplicate"),
        _observation("2024-01-02", "duplicate"),
    ], columns=OBSERVATION_COLUMNS)
    with pytest.raises(ValueError, match="double-count"):
        calculate_family_metrics(observations)


def test_one_day_family_marks_undefined_movement_rate() -> None:
    observations = pd.DataFrame([
        _observation("2024-01-01", "single", event_type="isolated_single_day")
    ], columns=OBSERVATION_COLUMNS)
    row = calculate_family_metrics(observations).iloc[0]
    assert row.status == "warning"
    assert "at least two" in row.reason
    assert np.isnan(row.mean_family_speed_km_per_day)
