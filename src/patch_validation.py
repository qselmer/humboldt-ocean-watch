"""Structural validation and lazy cached-climatology selection for patches."""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from src.daily_climatology import DAILY_METHOD, MONTHLY_METHOD
from src.data_loader import COPERNICUS_CACHED_MODE
from src.grid_geometry import GridGeometry, build_grid_geometry, unwrap_longitudes
from src.quality_control import coordinate_names, recognize_units


@dataclass(frozen=True)
class ValidatedPatchInput:
    """Canonical two-dimensional field, masks, threshold, and geometry."""

    field: xr.DataArray
    threshold: xr.DataArray
    valid_ocean_mask: xr.DataArray
    ocean_mask: xr.DataArray
    cell_area_km2: xr.DataArray
    geometry: GridGeometry
    warnings: tuple[str, ...] = ()


@dataclass
class CachedClimatologySelection:
    """A lazily opened daily primary or explicit monthly fallback."""

    dataset: xr.Dataset
    method: str
    path: Path
    fallback_used: bool = False
    warnings: list[str] = field(default_factory=list)

    def close(self) -> None:
        self.dataset.close()


def _select_exact_date(field: xr.DataArray, analysis_date: Any | None) -> xr.DataArray:
    if "time" not in field.dims:
        if analysis_date is not None and "time" in field.coords:
            raise ValueError("Selected field has a time coordinate but no time dimension")
        return field
    if analysis_date is None:
        raise ValueError("An analysis date is required for a field with a time dimension")
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(field.time.values, errors="raise")).normalize()
        requested = pd.Timestamp(analysis_date).normalize()
    except (TypeError, ValueError) as exc:
        raise ValueError("Patch analysis date or time coordinate is unreadable") from exc
    positions = np.flatnonzero(dates == requested)
    if positions.size == 0:
        raise ValueError(f"Analysis date {requested.date().isoformat()} is unavailable")
    if positions.size > 1:
        raise ValueError(f"Analysis date {requested.date().isoformat()} is duplicated")
    return field.isel(time=int(positions[0]), drop=True)


def prepare_spatial_field(
    field: xr.DataArray,
    *,
    analysis_date: Any | None = None,
) -> tuple[xr.DataArray, list[str]]:
    """Validate and canonicalize one numeric latitude-longitude field."""
    if not isinstance(field, xr.DataArray):
        raise TypeError("Patch field must be an xarray.DataArray")
    selected = _select_exact_date(field, analysis_date)
    latitude_name, longitude_name = coordinate_names(selected)
    if selected[latitude_name].dims != (latitude_name,):
        raise ValueError("Latitude coordinate must be one-dimensional")
    if selected[longitude_name].dims != (longitude_name,):
        raise ValueError("Longitude coordinate must be one-dimensional")
    if set(selected.dims) != {latitude_name, longitude_name}:
        raise ValueError(
            "Patch field must be two-dimensional after date selection; "
            f"got dimensions {selected.dims}"
        )
    if not np.issubdtype(selected.dtype, np.number):
        raise TypeError("Patch field must be numeric")
    rename: dict[str, str] = {}
    if latitude_name != "latitude":
        rename[latitude_name] = "latitude"
    if longitude_name != "longitude":
        rename[longitude_name] = "longitude"
    selected = selected.rename(rename).transpose("latitude", "longitude")
    latitude = np.asarray(selected.latitude.values, dtype=float)
    longitude = np.asarray(selected.longitude.values, dtype=float)
    if not np.isfinite(latitude).all() or not np.isfinite(longitude).all():
        raise ValueError("Latitude and longitude coordinates must be finite")
    if np.unique(latitude).size != latitude.size:
        raise ValueError("Latitude coordinate contains duplicate centres")
    if np.unique(np.mod(longitude, 360.0)).size != longitude.size:
        raise ValueError("Longitude coordinate contains duplicate physical centres")
    warnings: list[str] = []
    latitude_difference = np.diff(latitude)
    if not (
        bool((latitude_difference > 0).all())
        or bool((latitude_difference < 0).all())
    ):
        selected = selected.sortby("latitude")
        warnings.append("Latitude coordinate was safely sorted")
    unwrapped_longitude = unwrap_longitudes(longitude)
    longitude_difference = np.diff(unwrapped_longitude)
    if not (
        bool((longitude_difference > 0).all())
        or bool((longitude_difference < 0).all())
    ):
        normalized = (selected.longitude + 180.0) % 360.0 - 180.0
        selected = selected.assign_coords(longitude=normalized).sortby("longitude")
        warnings.append("Longitude coordinate was normalized and safely sorted")
    values = np.asarray(selected.values, dtype=float)
    if np.isinf(values).any():
        raise ValueError("Patch field contains infinite values")
    if not np.isfinite(values).any():
        raise ValueError("Patch field contains no finite ocean cells")
    return selected, warnings


def _aligned_auxiliary(
    value: xr.DataArray | np.ndarray | None,
    field: xr.DataArray,
    *,
    name: str,
    analysis_date: Any | None,
    default: float,
) -> xr.DataArray:
    if value is None:
        return xr.full_like(field, default, dtype=float).rename(name)
    if isinstance(value, np.ndarray):
        array = np.asarray(value)
        if array.shape != field.shape:
            raise ValueError(f"{name} shape is incompatible with the patch field")
        return xr.DataArray(
            array,
            dims=field.dims,
            coords=field.coords,
            name=name,
        )
    if not isinstance(value, xr.DataArray):
        raise TypeError(f"{name} must be an xarray.DataArray or NumPy array")
    selected = _select_exact_date(value, analysis_date) if "time" in value.dims else value
    latitude_name, longitude_name = coordinate_names(selected)
    rename = {
        key: destination
        for key, destination in (
            (latitude_name, "latitude"),
            (longitude_name, "longitude"),
        )
        if key != destination
    }
    selected = selected.rename(rename)
    if not set(selected.dims).issubset({"latitude", "longitude"}):
        raise ValueError(f"{name} has incompatible dimensions: {selected.dims}")
    if "longitude" in selected.dims:
        source_longitude = np.asarray(selected.longitude.values, dtype=float)
        target_longitude = np.asarray(field.longitude.values, dtype=float)
        if not np.all(np.isin(target_longitude, source_longitude)):
            normalized = (selected.longitude + 180.0) % 360.0 - 180.0
            selected = selected.assign_coords(longitude=normalized)
    try:
        indexers = {dimension: field[dimension] for dimension in selected.dims}
        selected = selected.sel(indexers).broadcast_like(field)
    except (KeyError, ValueError) as exc:
        raise ValueError(f"{name} coordinates are incompatible with the patch field") from exc
    return selected.transpose("latitude", "longitude").rename(name)


def validate_patch_inputs(
    field: xr.DataArray,
    threshold: float | xr.DataArray | np.ndarray,
    *,
    analysis_date: Any | None = None,
    ocean_mask: xr.DataArray | np.ndarray | None = None,
    cell_areas: xr.DataArray | np.ndarray | None = None,
) -> ValidatedPatchInput:
    """Validate a selected source field, threshold, ocean mask, and cell area."""
    prepared, warnings = prepare_spatial_field(field, analysis_date=analysis_date)
    if np.isscalar(threshold):
        scalar = float(threshold)
        if not np.isfinite(scalar):
            raise ValueError("Patch threshold must be finite")
        threshold_array = xr.full_like(prepared, scalar, dtype=float).rename("threshold")
    else:
        threshold_array = _aligned_auxiliary(
            threshold, prepared, name="threshold", analysis_date=analysis_date, default=np.nan
        ).astype(float)
        if np.isinf(np.asarray(threshold_array.values, dtype=float)).any():
            raise ValueError("Patch threshold contains infinite values")
    raw_mask_array = _aligned_auxiliary(
        ocean_mask, prepared, name="ocean_mask", analysis_date=analysis_date, default=1.0
    )
    raw_mask_values = np.asarray(raw_mask_array.values)
    if not np.isfinite(raw_mask_values.astype(float)).all():
        raise ValueError("Ocean mask must contain only finite values")
    mask_array = raw_mask_array.astype(bool)
    geometry = build_grid_geometry(prepared)
    if cell_areas is None:
        area_array = geometry.area_km2
    else:
        area_array = _aligned_auxiliary(
            cell_areas,
            prepared,
            name="cell_area",
            analysis_date=analysis_date,
            default=np.nan,
        ).astype(float)
        geometry = replace(geometry, area_km2=area_array)
    values = np.asarray(prepared.values, dtype=float)
    threshold_values = np.asarray(threshold_array.values, dtype=float)
    ocean_values = np.asarray(mask_array.values, dtype=bool)
    valid = ocean_values & np.isfinite(values) & np.isfinite(threshold_values)
    if not valid.any():
        raise ValueError("Patch field and threshold have no finite valid ocean cells")
    area_values = np.asarray(area_array.values, dtype=float)
    if not np.isfinite(area_values[valid]).all() or bool((area_values[valid] <= 0).any()):
        raise ValueError("Grid-cell areas must be finite and positive over valid ocean cells")
    valid_array = xr.DataArray(
        valid,
        dims=prepared.dims,
        coords=prepared.coords,
        name="valid_ocean_mask",
    )
    return ValidatedPatchInput(
        field=prepared,
        threshold=threshold_array,
        valid_ocean_mask=valid_array,
        ocean_mask=mask_array,
        cell_area_km2=area_array,
        geometry=geometry,
        warnings=tuple(warnings),
    )


def _looks_synthetic(climatology: xr.Dataset) -> bool:
    metadata = " ".join(
        str(climatology.attrs.get(name, ""))
        for name in ("climatology_method", "title", "source_dataset", "source")
    ).lower()
    return "synthetic" in metadata or "demo" in metadata


def validate_climatology_for_patches(
    climatology: xr.Dataset,
    *,
    data_mode: str,
    require_daily_threshold: bool,
    threshold_variable: str = "threshold_p90",
) -> str:
    """Validate the structural fields needed by patch definitions, without loading a cube."""
    if not isinstance(climatology, xr.Dataset):
        raise TypeError("Patch climatology must be an xarray.Dataset")
    if data_mode == COPERNICUS_CACHED_MODE and _looks_synthetic(climatology):
        raise ValueError("Real Copernicus SST cannot use a synthetic climatology")
    coordinate_names(climatology)
    if "climatological_day" in climatology.dims:
        method = str(climatology.attrs.get("climatology_method", DAILY_METHOD))
        if method != DAILY_METHOD:
            raise ValueError("Daily patch climatology must use method daily_smoothed")
        if climatology.sizes.get("climatological_day") != 366:
            raise ValueError("Daily patch climatology must contain 366 climatological days")
        required = {"climatology_mean", "climatology_std"}
        if require_daily_threshold:
            required.add(threshold_variable)
    elif "month" in climatology.dims:
        method = MONTHLY_METHOD
        required = {"climatological_mean", "climatological_standard_deviation"}
        if require_daily_threshold:
            raise ValueError(
                "Daily climatological patch thresholds require the smoothed daily climatology; "
                "monthly fallback is not permitted"
            )
    else:
        raise ValueError("Patch climatology has no supported calendar dimension")
    missing = required - set(climatology.data_vars)
    if missing:
        raise ValueError(f"Patch climatology is missing variables: {sorted(missing)}")
    for name in required:
        if not np.issubdtype(climatology[name].dtype, np.number):
            raise TypeError(f"Patch climatology variable {name!r} must be numeric")
        if recognize_units(climatology[name].attrs.get("units")) != "celsius":
            raise ValueError(f"Patch climatology variable {name!r} must use Celsius units")
    return method


def open_cached_climatology(
    daily_path: Path,
    monthly_path: Path,
    *,
    data_mode: str,
    require_daily_threshold: bool,
    allow_monthly_fallback: bool,
    threshold_variable: str = "threshold_p90",
) -> CachedClimatologySelection:
    """Apply the existing daily-primary/monthly-fallback policy lazily."""
    warnings: list[str] = []
    if daily_path.exists():
        daily = xr.open_dataset(daily_path)
        if data_mode == COPERNICUS_CACHED_MODE and _looks_synthetic(daily):
            daily.close()
            raise ValueError("Real Copernicus SST cannot use a synthetic climatology")
        try:
            method = validate_climatology_for_patches(
                daily,
                data_mode=data_mode,
                require_daily_threshold=require_daily_threshold,
                threshold_variable=threshold_variable,
            )
        except Exception as exc:
            daily.close()
            if require_daily_threshold:
                raise
            warnings.append(f"Daily climatology is invalid for patch detection: {exc}")
        else:
            return CachedClimatologySelection(daily, method, daily_path, False, warnings)
    elif require_daily_threshold:
        raise FileNotFoundError(
            f"Smoothed daily climatology required for percentile patches was not found: {daily_path}"
        )
    else:
        warnings.append(f"Smoothed daily climatology was not found: {daily_path}")

    if not allow_monthly_fallback:
        raise FileNotFoundError("No valid daily climatology is available and monthly fallback is disabled")
    if not monthly_path.exists():
        raise FileNotFoundError(f"Monthly fallback climatology was not found: {monthly_path}")
    monthly = xr.open_dataset(monthly_path)
    try:
        method = validate_climatology_for_patches(
            monthly,
            data_mode=data_mode,
            require_daily_threshold=False,
            threshold_variable=threshold_variable,
        )
    except Exception:
        monthly.close()
        raise
    warnings.append(
        "Daily smoothed climatology is unavailable or invalid; using the explicit real monthly fallback"
    )
    return CachedClimatologySelection(monthly, method, monthly_path, True, warnings)
