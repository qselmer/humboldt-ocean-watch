"""Structural validation for cached daily-patch tracking inputs."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from src.grid_geometry import GridGeometry, build_grid_geometry, unwrap_longitudes
from src.quality_control import coordinate_names


@dataclass(frozen=True)
class ValidatedTrackingInputs:
    patches: pd.DataFrame
    daily_summary: pd.DataFrame
    labels: xr.Dataset
    geometry: GridGeometry
    dates: pd.DatetimeIndex
    warnings: tuple[str, ...] = ()


def _synthetic(value: Any) -> bool:
    return "synthetic" in str(value or "").lower()


def _real_mode(value: Any) -> bool:
    text = str(value or "").lower()
    return "copernicus" in text or "cached data" in text


def validate_tracking_inputs(
    patches: pd.DataFrame,
    labels: xr.Dataset,
    daily_summary: pd.DataFrame | None = None,
) -> ValidatedTrackingInputs:
    """Validate table/cube agreement without assigning temporal meaning to IDs."""
    if not isinstance(patches, pd.DataFrame):
        raise TypeError("Daily patches must be supplied as a pandas DataFrame")
    if not isinstance(labels, xr.Dataset):
        raise TypeError("Daily patch labels must be supplied as an xarray Dataset")
    required_patch = {
        "date", "patch_id", "area_km2", "centroid_latitude",
        "centroid_longitude", "mean_source_value", "maximum_source_value",
        "mean_exceedance", "maximum_exceedance", "source_variable",
        "threshold_type", "direction", "valid_coverage", "climatology_method",
        "data_mode",
    }
    missing = sorted(required_patch.difference(patches.columns))
    if missing:
        raise ValueError("Daily patch table is missing required columns: " + ", ".join(missing))
    if "patch_id" not in labels:
        raise ValueError("Daily label cube must contain patch_id")
    if "time" not in labels.dims or "time" not in labels.coords:
        raise ValueError("Daily label cube must contain a time dimension and coordinate")
    latitude_name, longitude_name = coordinate_names(labels)
    patch_labels = labels.patch_id
    expected_dims = {"time", latitude_name, longitude_name}
    if set(patch_labels.dims) != expected_dims or patch_labels.ndim != 3:
        raise ValueError("patch_id must have time, latitude, and longitude dimensions")
    patch_labels = patch_labels.transpose("time", latitude_name, longitude_name)
    if not np.issubdtype(patch_labels.dtype, np.integer):
        raise ValueError("patch_id labels must use an integer dtype")
    if bool((patch_labels < 0).any()):
        raise ValueError("patch_id labels cannot be negative")
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(labels.time.values, errors="raise")).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError("Label-cube dates are unreadable") from exc
    if dates.has_duplicates:
        raise ValueError("Tracking dates cannot contain duplicates")
    if not dates.is_monotonic_increasing:
        raise ValueError("Label-cube dates must be ordered chronologically")

    canonical = patches.copy()
    try:
        canonical["date"] = pd.to_datetime(canonical.date, errors="raise").dt.normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError("Daily patch-table dates are unreadable") from exc
    if not canonical.date.is_monotonic_increasing:
        raise ValueError("Daily patch-table dates must be ordered chronologically")
    if canonical.duplicated(["date", "patch_id"]).any():
        raise ValueError("Local patch IDs must be unique within each date")
    identifiers = pd.to_numeric(canonical.patch_id, errors="coerce")
    if identifiers.isna().any() or bool((identifiers < 1).any()) or not np.allclose(identifiers, np.round(identifiers)):
        raise ValueError("Local patch IDs must be positive integers")
    canonical["patch_id"] = identifiers.astype(int)
    if not canonical.empty and not set(canonical.date).issubset(set(dates)):
        raise ValueError("Patch-table dates must exist in the label cube")

    first_field = patch_labels.isel(time=0, drop=True)
    geometry = build_grid_geometry(first_field)
    cell_areas = np.asarray(geometry.area_km2.values, dtype=float)
    if not np.isfinite(cell_areas).all() or bool((cell_areas <= 0).any()):
        raise ValueError("Tracking cell areas must be finite and positive")
    latitude = np.asarray(labels[latitude_name].values, dtype=float)
    longitude = unwrap_longitudes(np.asarray(labels[longitude_name].values, dtype=float))
    if not np.isfinite(latitude).all() or not np.isfinite(longitude).all():
        raise ValueError("Tracking spatial coordinates must be finite")
    area_values = pd.to_numeric(canonical.area_km2, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(area_values).all() or bool((area_values < 0).any()):
        raise ValueError("No patch may have a negative or non-finite area")
    if bool((area_values == 0).any()):
        raise ValueError("Every retained patch must have positive area")
    latitude_values = pd.to_numeric(canonical.centroid_latitude, errors="coerce").to_numpy(dtype=float)
    longitude_values = pd.to_numeric(canonical.centroid_longitude, errors="coerce").to_numpy(dtype=float)
    if not np.isfinite(latitude_values).all() or not np.isfinite(longitude_values).all():
        raise ValueError("Patch centroid coordinates must be finite")
    if bool(((latitude_values < latitude.min()) | (latitude_values > latitude.max())).any()):
        raise ValueError("Patch centroid latitude falls outside the spatial domain")
    unwrapped_centroids = np.rad2deg(np.unwrap(np.deg2rad(longitude_values))) if len(longitude_values) else longitude_values
    if len(unwrapped_centroids) and bool(((unwrapped_centroids < longitude.min() - 1e-6) | (unwrapped_centroids > longitude.max() + 1e-6)).any()):
        raise ValueError("Patch centroid longitude falls outside the spatial domain")

    table_dates: set[pd.Timestamp] = set()
    for position, date in enumerate(dates):
        label_values = np.asarray(patch_labels.isel(time=position).values, dtype=np.int64)
        label_ids = set(np.unique(label_values).tolist()) - {0}
        daily = canonical.loc[canonical.date == date]
        table_ids = set(daily.patch_id.astype(int).tolist())
        if label_ids != table_ids:
            raise ValueError(
                f"Patch-table IDs do not match label cube on {date.date().isoformat()}: "
                f"table={sorted(table_ids)}, labels={sorted(label_ids)}"
            )
        if label_ids:
            table_dates.add(date)
        for row in daily.itertuples(index=False):
            cells = label_values == int(row.patch_id)
            if not cells.any():
                raise ValueError("Every local patch must have at least one labelled cell")
            calculated_area = float(cell_areas[cells].sum())
            if not np.isclose(calculated_area, float(row.area_km2), rtol=1e-5, atol=1e-5):
                raise ValueError(
                    f"Patch area does not match spherical label area for "
                    f"{date.date().isoformat()} patch {int(row.patch_id)}"
                )
    if table_dates != set(canonical.date.unique()):
        raise ValueError("Nonzero label-cube dates must match patch-table dates")

    warnings: list[str] = []
    if daily_summary is None:
        summary = pd.DataFrame({"date": dates})
        warnings.append("Daily summary was not supplied; zero-patch counts were validated from labels")
    else:
        if not isinstance(daily_summary, pd.DataFrame) or "date" not in daily_summary:
            raise ValueError("Daily patch summary must contain date")
        summary = daily_summary.copy()
        summary["date"] = pd.to_datetime(summary.date, errors="raise").dt.normalize()
        if summary.date.duplicated().any() or not summary.date.is_monotonic_increasing:
            raise ValueError("Daily summary dates must be sorted and unique")
        if list(summary.date) != list(dates):
            raise ValueError("Daily summary dates must exactly match label-cube dates")
        if "patch_count" in summary:
            expected_counts = np.asarray([
                len(set(np.unique(patch_labels.isel(time=i).values).tolist()) - {0})
                for i in range(len(dates))
            ])
            if not np.array_equal(summary.patch_count.to_numpy(dtype=int), expected_counts):
                raise ValueError("Daily summary patch counts do not match label cube")

    modes = set(canonical.data_mode.dropna().astype(str))
    climates = set(canonical.climatology_method.dropna().astype(str))
    label_mode = labels.attrs.get("data_mode")
    label_climate = labels.attrs.get("climatology_method")
    if label_mode and modes and label_mode not in modes:
        raise ValueError("Patch-table data mode does not match label-cube metadata")
    if label_climate and climates and label_climate not in climates:
        raise ValueError("Patch-table climatology method does not match label-cube metadata")
    if any(_real_mode(mode) for mode in (*modes, label_mode)) and any(
        _synthetic(method) for method in (*climates, label_climate)
    ):
        raise ValueError("Real cached SST cannot be combined with synthetic climatology")
    return ValidatedTrackingInputs(
        patches=canonical,
        daily_summary=summary,
        labels=labels,
        geometry=geometry,
        dates=dates,
        warnings=tuple(warnings),
    )


def load_and_validate_tracking_inputs(
    daily_patches_path: str | Path,
    daily_labels_path: str | Path,
    daily_summary_path: str | Path,
) -> ValidatedTrackingInputs:
    """Read local cached products and apply complete structural validation."""
    patches_path = Path(daily_patches_path)
    labels_path = Path(daily_labels_path)
    summary_path = Path(daily_summary_path)
    for path, label in (
        (patches_path, "Daily patch table"),
        (labels_path, "Daily label cube"),
        (summary_path, "Daily patch summary"),
    ):
        if not path.exists():
            raise FileNotFoundError(f"{label} not found: {path}")
    try:
        patches = pd.read_parquet(patches_path)
    except Exception as exc:
        raise ValueError(f"Daily patch table is unreadable: {patches_path}") from exc
    try:
        summary = pd.read_parquet(summary_path)
    except Exception as exc:
        raise ValueError(f"Daily patch summary is unreadable: {summary_path}") from exc
    try:
        labels = xr.open_dataset(labels_path).load()
    except Exception as exc:
        raise ValueError(f"Daily label cube is unreadable: {labels_path}") from exc
    return validate_tracking_inputs(patches, labels, summary)
