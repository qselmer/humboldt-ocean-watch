"""Metrics for non-branching patch tracks and complete event families.

Movement uses great-circle distances and initial bearings.  Areas are never
interpolated across missing dates: an observed area-day is one patch area
multiplied by one observed day, and cumulative severity is the sum of
``mean_exceedance * area_km2 * one observed day``.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd

from src.patch_linking import haversine_distance_km, initial_bearing_degrees


TRACK_COLUMNS = [
    "track_id", "event_family_id", "start_date", "end_date",
    "observed_days", "duration_calendar_days", "internal_gap_count",
    "start_event_type", "end_event_type", "mean_area_km2",
    "median_area_km2", "maximum_area_km2", "minimum_area_km2",
    "area_standard_deviation_km2", "observed_area_days_km2_days",
    "area_weighted_mean_exceedance", "maximum_exceedance",
    "mean_source_value", "maximum_source_value", "cumulative_severity",
    "start_centroid_latitude", "start_centroid_longitude",
    "end_centroid_latitude", "end_centroid_longitude",
    "net_displacement_km", "trajectory_length_km", "mean_speed_km_per_day",
    "maximum_speed_km_per_day", "mean_bearing_degrees",
    "net_bearing_degrees", "trajectory_tortuosity",
    "mean_area_change_km2_per_day", "maximum_expansion_rate_km2_per_day",
    "maximum_contraction_rate_km2_per_day", "relative_area_change",
    "expansion_day_count", "contraction_day_count", "touched_north_boundary",
    "touched_south_boundary", "touched_east_boundary",
    "touched_west_boundary", "boundary_contact_day_count",
    "area_unit", "observed_area_days_unit", "exceedance_unit",
    "cumulative_severity_unit", "movement_unit", "speed_unit",
    "status", "reason",
]

FAMILY_COLUMNS = [
    "event_family_id", "start_date", "end_date", "duration_calendar_days",
    "observed_patch_days", "track_count", "unique_patch_observation_count",
    "split_count", "merge_count", "complex_branch_count",
    "maximum_simultaneous_patch_count", "mean_total_daily_area_km2",
    "maximum_total_daily_area_km2", "family_area_days_km2_days",
    "maximum_patch_area_km2", "area_weighted_mean_exceedance",
    "maximum_exceedance", "cumulative_severity", "start_centroid_latitude",
    "start_centroid_longitude", "end_centroid_latitude",
    "end_centroid_longitude", "family_net_displacement_km",
    "family_trajectory_length_km", "mean_family_speed_km_per_day",
    "maximum_family_extent_km", "boundary_contact_day_count",
    "climatology_method", "data_mode", "area_unit",
    "family_area_days_unit", "cumulative_severity_unit", "movement_unit",
    "speed_unit", "status", "reason",
]


def _weighted_mean(values: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(weights) & (weights > 0)
    denominator = float(weights[valid].sum())
    return float(np.sum(values[valid] * weights[valid]) / denominator) if denominator > 0 else np.nan


def _weighted_centroid(frame: pd.DataFrame) -> tuple[float, float]:
    weights = frame.area_km2.to_numpy(dtype=float)
    latitude = _weighted_mean(frame.centroid_latitude.to_numpy(dtype=float), weights)
    longitude = frame.centroid_longitude.to_numpy(dtype=float)
    valid = np.isfinite(longitude) & np.isfinite(weights) & (weights > 0)
    if not valid.any():
        return latitude, np.nan
    sine = np.sum(weights[valid] * np.sin(np.deg2rad(longitude[valid])))
    cosine = np.sum(weights[valid] * np.cos(np.deg2rad(longitude[valid])))
    result = float(np.rad2deg(np.arctan2(sine, cosine)))
    if bool(((longitude[valid] >= 0) & (longitude[valid] <= 360)).all()):
        result %= 360.0
    return latitude, result


def _movement(frame: pd.DataFrame) -> dict[str, float | int]:
    ordered = frame.sort_values("date", kind="mergesort")
    latitude = ordered.centroid_latitude.to_numpy(dtype=float)
    longitude = ordered.centroid_longitude.to_numpy(dtype=float)
    dates = pd.DatetimeIndex(ordered.date)
    net = haversine_distance_km(latitude[0], longitude[0], latitude[-1], longitude[-1])
    net_bearing = initial_bearing_degrees(latitude[0], longitude[0], latitude[-1], longitude[-1])
    if len(ordered) < 2:
        return {
            "net_displacement_km": 0.0,
            "trajectory_length_km": 0.0,
            "mean_speed_km_per_day": np.nan,
            "maximum_speed_km_per_day": np.nan,
            "mean_bearing_degrees": np.nan,
            "net_bearing_degrees": np.nan,
            "trajectory_tortuosity": np.nan,
        }
    elapsed = np.diff(dates.values).astype("timedelta64[D]").astype(int)
    distances = np.asarray([
        haversine_distance_km(latitude[i], longitude[i], latitude[i + 1], longitude[i + 1])
        for i in range(len(ordered) - 1)
    ])
    bearings = np.asarray([
        initial_bearing_degrees(latitude[i], longitude[i], latitude[i + 1], longitude[i + 1])
        for i in range(len(ordered) - 1)
    ])
    speeds = distances / elapsed
    valid_bearing = np.isfinite(bearings)
    if valid_bearing.any():
        radians = np.deg2rad(bearings[valid_bearing])
        mean_bearing = float((np.degrees(np.arctan2(np.mean(np.sin(radians)), np.mean(np.cos(radians)))) + 360) % 360)
    else:
        mean_bearing = np.nan
    trajectory = float(np.sum(distances))
    total_elapsed = int((dates[-1] - dates[0]).days)
    return {
        "net_displacement_km": net,
        "trajectory_length_km": trajectory,
        "mean_speed_km_per_day": trajectory / total_elapsed if total_elapsed > 0 else np.nan,
        "maximum_speed_km_per_day": float(np.max(speeds)),
        "mean_bearing_degrees": mean_bearing,
        "net_bearing_degrees": net_bearing,
        "trajectory_tortuosity": trajectory / net if np.isfinite(net) and net > 1e-12 else np.nan,
    }


def calculate_track_metrics(observations: pd.DataFrame) -> pd.DataFrame:
    """Return one deterministic metrics row per maximal non-branching track."""
    if observations.empty:
        return pd.DataFrame(columns=TRACK_COLUMNS)
    rows: list[dict[str, Any]] = []
    for track_id, group in observations.groupby("track_id", sort=False):
        track = group.sort_values("date", kind="mergesort").reset_index(drop=True)
        families = track.event_family_id.dropna().unique()
        if len(families) != 1:
            raise ValueError("Every track must belong to exactly one event family")
        dates = pd.DatetimeIndex(track.date)
        if dates.has_duplicates or not dates.is_monotonic_increasing:
            raise ValueError("Track observations must have sorted unique dates")
        areas = track.area_km2.to_numpy(dtype=float)
        if not np.isfinite(areas).all() or bool((areas <= 0).any()):
            raise ValueError("Track areas must be finite and positive")
        elapsed = np.diff(dates.values).astype("timedelta64[D]").astype(int)
        changes = np.diff(areas) / elapsed if len(areas) > 1 else np.asarray([], dtype=float)
        mean_exceedance = track.mean_exceedance.to_numpy(dtype=float)
        source = track.mean_source_value.to_numpy(dtype=float)
        movement = _movement(track)
        warnings: list[str] = []
        if len(track) == 1:
            warnings.append("Movement and area-change rates require at least two observations")
        elif not np.isfinite(float(movement["trajectory_tortuosity"])):
            warnings.append("Trajectory tortuosity is not calculated for zero net displacement")
        boundary = track[[
            "touches_north_boundary", "touches_south_boundary",
            "touches_east_boundary", "touches_west_boundary",
        ]].fillna(False).astype(bool)
        rows.append({
            "track_id": track_id,
            "event_family_id": families[0],
            "start_date": dates[0],
            "end_date": dates[-1],
            "observed_days": len(track),
            "duration_calendar_days": int((dates[-1] - dates[0]).days) + 1,
            "internal_gap_count": int(np.sum(elapsed > 1)),
            "start_event_type": track.iloc[0].node_event_type,
            "end_event_type": track.iloc[-1].node_event_type,
            "mean_area_km2": float(np.mean(areas)),
            "median_area_km2": float(np.median(areas)),
            "maximum_area_km2": float(np.max(areas)),
            "minimum_area_km2": float(np.min(areas)),
            "area_standard_deviation_km2": float(np.std(areas)),
            "observed_area_days_km2_days": float(np.sum(areas)),
            "area_weighted_mean_exceedance": _weighted_mean(mean_exceedance, areas),
            "maximum_exceedance": float(np.nanmax(track.maximum_exceedance)),
            "mean_source_value": _weighted_mean(source, areas),
            "maximum_source_value": float(np.nanmax(track.maximum_source_value)),
            "cumulative_severity": float(np.nansum(mean_exceedance * areas)),
            "start_centroid_latitude": float(track.iloc[0].centroid_latitude),
            "start_centroid_longitude": float(track.iloc[0].centroid_longitude),
            "end_centroid_latitude": float(track.iloc[-1].centroid_latitude),
            "end_centroid_longitude": float(track.iloc[-1].centroid_longitude),
            **movement,
            "mean_area_change_km2_per_day": float(np.mean(changes)) if changes.size else np.nan,
            "maximum_expansion_rate_km2_per_day": (
                float(np.max(changes[changes > 0])) if bool((changes > 0).any()) else np.nan
            ),
            "maximum_contraction_rate_km2_per_day": (
                float(np.min(changes[changes < 0])) if bool((changes < 0).any()) else np.nan
            ),
            "relative_area_change": float((areas[-1] - areas[0]) / areas[0]),
            "expansion_day_count": int(np.sum(changes > 0)),
            "contraction_day_count": int(np.sum(changes < 0)),
            "touched_north_boundary": bool(boundary.touches_north_boundary.any()),
            "touched_south_boundary": bool(boundary.touches_south_boundary.any()),
            "touched_east_boundary": bool(boundary.touches_east_boundary.any()),
            "touched_west_boundary": bool(boundary.touches_west_boundary.any()),
            "boundary_contact_day_count": int(boundary.any(axis=1).sum()),
            "area_unit": "km2",
            "observed_area_days_unit": "km2 day",
            "exceedance_unit": "source-variable unit",
            "cumulative_severity_unit": "source-variable unit km2 day",
            "movement_unit": "km",
            "speed_unit": "km day-1",
            "status": "warning" if warnings else "valid",
            "reason": "; ".join(warnings) if warnings else None,
        })
    return pd.DataFrame(rows, columns=TRACK_COLUMNS).sort_values("track_id").reset_index(drop=True)


def _maximum_daily_extent(family: pd.DataFrame) -> float:
    maximum = 0.0
    for _, daily in family.groupby("date", sort=False):
        coordinates = daily[["centroid_latitude", "centroid_longitude"]].to_numpy(dtype=float)
        for first in range(len(coordinates)):
            for second in range(first + 1, len(coordinates)):
                maximum = max(maximum, haversine_distance_km(*coordinates[first], *coordinates[second]))
    return float(maximum)


def calculate_family_metrics(observations: pd.DataFrame) -> pd.DataFrame:
    """Return unique-observation metrics for each weak lineage component."""
    if observations.empty:
        return pd.DataFrame(columns=FAMILY_COLUMNS)
    if observations.node_id.duplicated().any():
        raise ValueError("Family metrics cannot double-count patch observations")
    rows: list[dict[str, Any]] = []
    for family_id, group in observations.groupby("event_family_id", sort=False):
        family = group.sort_values(["date", "local_patch_id"], kind="mergesort")
        dates = pd.DatetimeIndex(family.date)
        daily_rows: list[dict[str, Any]] = []
        for date, daily in family.groupby("date", sort=True):
            latitude, longitude = _weighted_centroid(daily)
            daily_rows.append({
                "date": pd.Timestamp(date),
                "area_km2": float(daily.area_km2.sum()),
                "centroid_latitude": latitude,
                "centroid_longitude": longitude,
            })
        daily_frame = pd.DataFrame(daily_rows)
        movement = _movement(daily_frame)
        areas = family.area_km2.to_numpy(dtype=float)
        exceedance = family.mean_exceedance.to_numpy(dtype=float)
        climate = family.climatology_method.dropna().astype(str).unique()
        modes = family.data_mode.dropna().astype(str).unique()
        if len(climate) > 1 or len(modes) > 1:
            raise ValueError("An event family cannot mix climatology methods or data modes")
        boundary = family[[
            "touches_north_boundary", "touches_south_boundary",
            "touches_east_boundary", "touches_west_boundary",
        ]].fillna(False).astype(bool)
        boundary_dates = family.loc[boundary.any(axis=1), "date"].nunique()
        first, last = daily_frame.iloc[0], daily_frame.iloc[-1]
        warnings: list[str] = []
        if len(daily_frame) < 2:
            warnings.append("Family movement rates require at least two observed dates")
        rows.append({
            "event_family_id": family_id,
            "start_date": pd.Timestamp(daily_frame.date.iloc[0]),
            "end_date": pd.Timestamp(daily_frame.date.iloc[-1]),
            "duration_calendar_days": int((daily_frame.date.iloc[-1] - daily_frame.date.iloc[0]).days) + 1,
            "observed_patch_days": len(family),
            "track_count": int(family.track_id.nunique()),
            "unique_patch_observation_count": int(family.node_id.nunique()),
            "split_count": int((family.node_event_type == "split_parent").sum()),
            "merge_count": int((family.node_event_type == "merge_child").sum()),
            "complex_branch_count": int((family.node_event_type == "complex_branch").sum()),
            "maximum_simultaneous_patch_count": int(family.groupby("date").size().max()),
            "mean_total_daily_area_km2": float(daily_frame.area_km2.mean()),
            "maximum_total_daily_area_km2": float(daily_frame.area_km2.max()),
            "family_area_days_km2_days": float(areas.sum()),
            "maximum_patch_area_km2": float(areas.max()),
            "area_weighted_mean_exceedance": _weighted_mean(exceedance, areas),
            "maximum_exceedance": float(np.nanmax(family.maximum_exceedance)),
            "cumulative_severity": float(np.nansum(exceedance * areas)),
            "start_centroid_latitude": float(first.centroid_latitude),
            "start_centroid_longitude": float(first.centroid_longitude),
            "end_centroid_latitude": float(last.centroid_latitude),
            "end_centroid_longitude": float(last.centroid_longitude),
            "family_net_displacement_km": float(movement["net_displacement_km"]),
            "family_trajectory_length_km": float(movement["trajectory_length_km"]),
            "mean_family_speed_km_per_day": float(movement["mean_speed_km_per_day"]),
            "maximum_family_extent_km": _maximum_daily_extent(family),
            "boundary_contact_day_count": int(boundary_dates),
            "climatology_method": climate[0] if len(climate) else None,
            "data_mode": modes[0] if len(modes) else None,
            "area_unit": "km2",
            "family_area_days_unit": "km2 day",
            "cumulative_severity_unit": "source-variable unit km2 day",
            "movement_unit": "km",
            "speed_unit": "km day-1",
            "status": "warning" if warnings else "valid",
            "reason": "; ".join(warnings) if warnings else None,
        })
    return pd.DataFrame(rows, columns=FAMILY_COLUMNS).sort_values("event_family_id").reset_index(drop=True)
