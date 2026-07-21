"""Stable table and catalogue preparation for Increment 4 products."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import numpy as np
import pandas as pd


PATCH_DISPLAY_COLUMNS = [
    "patch_id", "area_km2", "area_fraction_of_threshold_total",
    "centroid_latitude", "centroid_longitude", "mean_source_value",
    "maximum_source_value", "mean_exceedance", "maximum_exceedance",
    "equivalent_radius_km", "compactness", "elongation", "orientation_degrees",
    "touches_north_boundary", "touches_south_boundary",
    "touches_east_boundary", "touches_west_boundary",
]

OBSERVATION_DISPLAY_COLUMNS = [
    "date", "local_patch_id", "track_id", "node_event_type", "area_km2",
    "centroid_latitude", "centroid_longitude", "mean_exceedance",
    "predecessor_count", "successor_count",
]

EDGE_DISPLAY_COLUMNS = [
    "predecessor_node_id", "successor_node_id", "elapsed_days",
    "area_weighted_iou", "predecessor_overlap_fraction",
    "successor_overlap_fraction", "centroid_distance_km", "link_score",
    "link_basis", "lineage_relation", "is_continuation_backbone",
]

TRACK_DISPLAY_COLUMNS = [
    "track_id", "start_date", "end_date", "duration_calendar_days",
    "observed_days", "maximum_area_km2", "cumulative_severity",
    "trajectory_length_km", "mean_speed_km_per_day", "maximum_speed_km_per_day",
    "start_event_type", "end_event_type",
]

EVENT_DISPLAY_COLUMNS = [
    "event_id", "start_date", "end_date", "duration_calendar_days",
    "mean_intensity", "maximum_intensity", "cumulative_intensity",
    "severity_class", "mean_valid_coverage",
]


def available_columns(data: pd.DataFrame, requested: Sequence[str]) -> list[str]:
    return [column for column in requested if column in data.columns]


def rows_for_date(data: pd.DataFrame, selected_date: str | pd.Timestamp) -> pd.DataFrame:
    if data.empty or "date" not in data.columns:
        return data.iloc[0:0].copy()
    target = pd.Timestamp(selected_date).normalize()
    dates = pd.to_datetime(data["date"], errors="coerce").dt.normalize()
    return data.loc[dates == target].copy()


def prepare_patch_table(data: pd.DataFrame, selected_date: str | pd.Timestamp) -> pd.DataFrame:
    current = rows_for_date(data, selected_date)
    columns = available_columns(current, PATCH_DISPLAY_COLUMNS)
    return current[columns].sort_values("patch_id", kind="mergesort").reset_index(drop=True)


def prepare_family_tables(
    family_id: str,
    observations: pd.DataFrame,
    edges: pd.DataFrame,
    tracks: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    family_observations = observations.loc[observations.get("event_family_id", pd.Series(dtype=object)).astype(str) == family_id].copy()
    node_ids = set(family_observations.get("node_id", pd.Series(dtype=object)).dropna().astype(str))
    if edges.empty or not node_ids:
        family_edges = edges.iloc[0:0].copy()
    else:
        family_edges = edges.loc[
            edges["predecessor_node_id"].astype(str).isin(node_ids)
            & edges["successor_node_id"].astype(str).isin(node_ids)
        ].copy()
    family_tracks = tracks.loc[tracks.get("event_family_id", pd.Series(dtype=object)).astype(str) == family_id].copy()
    return (
        family_observations[available_columns(family_observations, OBSERVATION_DISPLAY_COLUMNS)],
        family_edges[available_columns(family_edges, EDGE_DISPLAY_COLUMNS)],
        family_tracks[available_columns(family_tracks, TRACK_DISPLAY_COLUMNS)],
    )


def filter_catalogue(
    data: pd.DataFrame,
    *,
    start_date: str | pd.Timestamp | None = None,
    end_date: str | pd.Timestamp | None = None,
    minimum_duration: float = 0,
    minimum_area: float = 0,
    minimum_severity: float = 0,
    require_split: bool = False,
    require_merge: bool = False,
    require_boundary_contact: bool = False,
    search: str = "",
) -> pd.DataFrame:
    """Filter an existing catalogue without calculating new metrics."""
    frame = data.copy()
    if frame.empty:
        return frame
    if start_date is not None and "end_date" in frame:
        frame = frame[pd.to_datetime(frame.end_date, errors="coerce") >= pd.Timestamp(start_date)]
    if end_date is not None and "start_date" in frame:
        frame = frame[pd.to_datetime(frame.start_date, errors="coerce") <= pd.Timestamp(end_date)]
    duration_column = next((c for c in ("duration_calendar_days", "observed_days") if c in frame), None)
    if duration_column:
        frame = frame[pd.to_numeric(frame[duration_column], errors="coerce").fillna(-np.inf) >= minimum_duration]
    area_column = next((c for c in ("maximum_area_km2", "maximum_total_daily_area_km2", "largest_patch_area_km2") if c in frame), None)
    if area_column:
        frame = frame[pd.to_numeric(frame[area_column], errors="coerce").fillna(-np.inf) >= minimum_area]
    severity_column = next((c for c in ("cumulative_severity", "cumulative_intensity") if c in frame), None)
    if severity_column:
        frame = frame[pd.to_numeric(frame[severity_column], errors="coerce").fillna(-np.inf) >= minimum_severity]
    if require_split and "split_count" in frame:
        frame = frame[pd.to_numeric(frame.split_count, errors="coerce").fillna(0) > 0]
    if require_merge and "merge_count" in frame:
        frame = frame[pd.to_numeric(frame.merge_count, errors="coerce").fillna(0) > 0]
    if require_boundary_contact:
        if "boundary_contact_day_count" in frame:
            frame = frame[pd.to_numeric(frame.boundary_contact_day_count, errors="coerce").fillna(0) > 0]
        else:
            boundary_columns = [column for column in frame if column.startswith("touches_")]
            if boundary_columns:
                frame = frame[frame[boundary_columns].fillna(False).astype(bool).any(axis=1)]
    if search.strip():
        needle = search.strip().casefold()
        text = frame.astype(str).agg(" ".join, axis=1).str.casefold()
        frame = frame[text.str.contains(needle, regex=False)]
    return frame.reset_index(drop=True)


def stable_table(data: pd.DataFrame, columns: Sequence[str], maximum_rows: int) -> pd.DataFrame:
    frame = data[available_columns(data, columns)].copy()
    return frame.head(maximum_rows).reset_index(drop=True)


def selected_record(data: pd.DataFrame, column: str, value: Any) -> dict[str, Any] | None:
    if data.empty or column not in data or value is None:
        return None
    selected = data.loc[data[column].astype(str) == str(value)]
    return selected.iloc[0].to_dict() if not selected.empty else None

