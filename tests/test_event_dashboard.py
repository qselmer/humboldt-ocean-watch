from __future__ import annotations

from pathlib import Path

import inspect
import json
import numpy as np
import pandas as pd

from src.event_dashboard import (
    consistent_track_family_selection,
    prepare_event_overview,
    prepare_family_daily_series,
    prepare_track_daily_series,
    render_thermal_events_tab,
)
from src.event_data_loader import EventProducts, ProductState
from src.event_tables import filter_catalogue, prepare_family_tables
from src.event_ui_utils import dataframe_to_iso_csv, record_to_json


def _state(frame: pd.DataFrame) -> ProductState[pd.DataFrame]:
    return ProductState(frame, Path("cached.parquet"), "empty" if frame.empty else "available")


def _products(*, active: bool = True, patch_count: int = 1, observation_count: int = 1) -> EventProducts:
    event_id = "EVT0001" if active else None
    flags = pd.DataFrame({
        "date": [pd.Timestamp("2026-01-02")], "source_variable": ["anomaly"], "value": [2.5],
        "threshold": [2.0], "exceeds_threshold": [active], "event_id": [event_id],
        "event_day": [2 if active else np.nan], "intensity": [0.5 if active else np.nan],
        "valid_coverage": [0.95], "status": ["valid"],
    })
    events = pd.DataFrame({
        "event_id": ["EVT0001"], "source_variable": ["anomaly"], "threshold_type": ["fixed"],
        "direction": ["above"], "start_date": [pd.Timestamp("2026-01-01")],
        "end_date": [pd.Timestamp("2026-01-03")], "peak_date": [pd.Timestamp("2026-01-02")],
        "duration_calendar_days": [3], "mean_intensity": [0.4], "maximum_intensity": [0.5],
        "cumulative_intensity": [1.2], "severity_class": ["moderate thermal exceedance"], "status": ["valid"],
    })
    patch_summary = pd.DataFrame({
        "date": [pd.Timestamp("2026-01-02")], "patch_count": [patch_count],
        "total_patch_area_km2": [100.0 if patch_count else 0.0], "largest_patch_area_km2": [100.0 if patch_count else 0.0],
        "threshold_area_fraction": [0.1 if patch_count else 0.0], "valid_coverage": [0.95], "status": ["valid"],
    })
    observations = pd.DataFrame({
        "node_id": [f"N{i}" for i in range(observation_count)], "date": [pd.Timestamp("2026-01-02")] * observation_count,
        "local_patch_id": list(range(1, observation_count + 1)), "track_id": [f"TRK{i:06d}" for i in range(1, observation_count + 1)],
        "event_family_id": [f"FAM{i:06d}" for i in range(1, observation_count + 1)], "node_event_type": ["isolated_single_day"] * observation_count,
        "area_km2": [100.0] * observation_count, "centroid_latitude": [-5.0] * observation_count,
        "centroid_longitude": [-85.0] * observation_count, "mean_exceedance": [0.5] * observation_count,
        "maximum_exceedance": [1.0] * observation_count, "predecessor_count": [0] * observation_count,
        "successor_count": [0] * observation_count, "status": ["valid"] * observation_count,
    })
    tracks = pd.DataFrame({"track_id": observations.track_id, "event_family_id": observations.event_family_id})
    empty = pd.DataFrame()
    return EventProducts(
        _state(events), _state(flags), _state(empty), _state(patch_summary), _state(observations),
        _state(empty), _state(tracks), _state(empty), _state(empty), _state(empty),
        ProductState({}, Path("summary.json"), "empty"),
    )


def test_event_overview_active_and_no_active_states() -> None:
    representative = pd.DataFrame({"date": [pd.Timestamp("2026-01-02")], "class": ["strong_coherent"], "valid_coverage": [0.96]})
    active = prepare_event_overview(_products(active=True), "2026-01-02", representative)
    assert active.active_event and active.event_id == "EVT0001" and active.event_day == 2
    assert active.patch_count == 1 and active.active_track_count == 1 and active.active_family_count == 1
    inactive = prepare_event_overview(_products(active=False, patch_count=0, observation_count=0), "2026-01-02", representative)
    assert not inactive.active_event and inactive.event_id is None
    assert inactive.patch_count == 0 and inactive.active_track_count == 0


def test_event_overview_counts_multiple_active_tracks_and_families() -> None:
    overview = prepare_event_overview(_products(observation_count=3), "2026-01-02")
    assert overview.active_track_count == 3
    assert overview.active_family_count == 3


def test_track_family_selection_rejects_stale_ids_and_restricts_family() -> None:
    tracks = pd.DataFrame({"track_id": ["TRK1", "TRK2"], "event_family_id": ["FAM1", "FAM2"]})
    track, family, options = consistent_track_family_selection(tracks, requested_track="TRK1", requested_family="FAM1")
    assert (track, family, options) == ("TRK1", "FAM1", ["TRK1"])
    assert consistent_track_family_selection(tracks, requested_track="STALE", requested_family="FAM1")[0] is None


def test_track_daily_rates_use_actual_elapsed_days() -> None:
    observations = pd.DataFrame({
        "node_id": ["A", "B"], "date": pd.to_datetime(["2026-01-01", "2026-01-03"]),
        "track_id": ["TRK1", "TRK1"], "area_km2": [10.0, 14.0],
        "centroid_latitude": [-5.0, -5.0], "centroid_longitude": [-85.0, -84.0],
    })
    edges = pd.DataFrame({"predecessor_node_id": ["A"], "successor_node_id": ["B"], "centroid_distance_km": [20.0]})
    result = prepare_track_daily_series(observations, edges, "TRK1")
    assert result.area_change_km2_per_day.iloc[1] == 2.0
    assert result.speed_km_per_day.iloc[1] == 10.0


def test_family_daily_series_does_not_double_count_and_separates_centroids() -> None:
    observations = pd.DataFrame({
        "event_family_id": ["FAM1", "FAM1", "FAM1"],
        "date": pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02"]),
        "area_km2": [10.0, 30.0, 20.0], "centroid_latitude": [-6.0, -4.0, -5.0],
        "centroid_longitude": [-86.0, -84.0, -85.0], "mean_exceedance": [1.0, 3.0, 2.0],
        "maximum_exceedance": [2.0, 4.0, 3.0],
    })
    result = prepare_family_daily_series(observations, "FAM1")
    assert result.total_family_area_km2.tolist() == [40.0, 20.0]
    assert result.patch_count.tolist() == [2, 1]
    assert result.centroid_latitude.iloc[0] == -4.5
    assert result.centroid_longitude.iloc[0] == -84.5


def test_family_tables_support_empty_edges() -> None:
    observations = _products().patch_observations.value.assign(
        predecessor_count=0, successor_count=0
    )
    tracks = pd.DataFrame({"track_id": ["TRK000001"], "event_family_id": ["FAM000001"], "start_date": [pd.Timestamp("2026-01-02")]})
    observation_table, edge_table, track_table = prepare_family_tables("FAM000001", observations, pd.DataFrame(), tracks)
    assert len(observation_table) == 1 and edge_table.empty and len(track_table) == 1


def test_catalogue_filters_and_exports_are_stable_strict_and_iso() -> None:
    tracks = pd.DataFrame({
        "track_id": ["TRK1", "TRK2"], "start_date": pd.to_datetime(["2026-01-01", "2026-02-01"]),
        "end_date": pd.to_datetime(["2026-01-05", "2026-02-02"]), "duration_calendar_days": [5, 2],
        "maximum_area_km2": [100.0, 20.0], "cumulative_severity": [50.0, np.inf],
        "boundary_contact_day_count": [1, 0],
    })
    filtered = filter_catalogue(tracks, minimum_duration=3, minimum_area=50, require_boundary_contact=True, search="TRK1")
    assert filtered.track_id.tolist() == ["TRK1"]
    csv_text = dataframe_to_iso_csv(filtered)
    assert "2026-01-01" in csv_text and "00:00:00" not in csv_text
    json_text = record_to_json(tracks)
    parsed = json.loads(json_text)
    assert parsed[1]["cumulative_severity"] is None
    assert "NaN" not in json_text and "Infinity" not in json_text


def test_empty_selection_json_is_valid() -> None:
    assert json.loads(record_to_json(pd.DataFrame())) == []


def test_event_dashboard_uses_progressive_disclosure_without_view_modes() -> None:
    source = inspect.getsource(render_thermal_events_tab)
    assert "view_mode" not in source
    assert "is_advanced" not in source
    assert "Additional event filters" in source

