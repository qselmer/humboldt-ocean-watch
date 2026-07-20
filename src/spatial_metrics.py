"""Spatial thermal summary metrics."""

from __future__ import annotations

import numpy as np
import xarray as xr


def area_weighted_mean(data: xr.DataArray) -> xr.DataArray:
    """Calculate a latitude-weighted spatial mean."""
    required = {"latitude", "longitude"}
    missing = required - set(data.dims)
    if missing:
        raise ValueError(f"Data is missing spatial dimensions: {sorted(missing)}")
    weights = np.cos(np.deg2rad(data["latitude"]))
    return data.weighted(weights).mean(("latitude", "longitude"), skipna=True)


def fraction_above_threshold(data: xr.DataArray, threshold: float) -> xr.DataArray:
    """Return the fraction of finite grid cells strictly above a threshold."""
    required = {"latitude", "longitude"}
    missing = required - set(data.dims)
    if missing:
        raise ValueError(f"Data is missing spatial dimensions: {sorted(missing)}")
    valid = data.notnull()
    numerator = ((data > threshold) & valid).sum(("latitude", "longitude"))
    denominator = valid.sum(("latitude", "longitude"))
    return xr.where(denominator > 0, numerator / denominator, np.nan).rename("fraction_above_threshold")


def percentage_area_above_threshold(data: xr.DataArray, threshold: float) -> xr.DataArray:
    """Return cosine-weighted valid ocean area above a threshold, in percent."""
    required = {"latitude", "longitude"}
    missing = required - set(data.dims)
    if missing:
        raise ValueError(f"Data is missing spatial dimensions: {sorted(missing)}")
    weights = np.cos(np.deg2rad(data["latitude"])).broadcast_like(data)
    valid = data.notnull()
    denominator = weights.where(valid).sum(("latitude", "longitude"), skipna=True)
    numerator = weights.where(valid & (data > threshold)).sum(
        ("latitude", "longitude"), skipna=True
    )
    percentage = xr.where(denominator > 0, 100.0 * numerator / denominator, np.nan)
    percentage.name = "percentage_area_above_threshold"
    percentage.attrs["units"] = "%"
    return percentage


def percentage_area_at_or_above_threshold(
    data: xr.DataArray, threshold: float
) -> xr.DataArray:
    """Return cosine-weighted valid area at or above a threshold, in percent."""
    required = {"latitude", "longitude"}
    missing = required - set(data.dims)
    if missing:
        raise ValueError(f"Data is missing spatial dimensions: {sorted(missing)}")
    weights = np.cos(np.deg2rad(data["latitude"])).broadcast_like(data)
    valid = data.notnull()
    denominator = weights.where(valid).sum(("latitude", "longitude"), skipna=True)
    numerator = weights.where(valid & (data >= threshold)).sum(
        ("latitude", "longitude"), skipna=True
    )
    result = xr.where(denominator > 0, 100.0 * numerator / denominator, np.nan)
    result.name = "percentage_area_at_or_above_threshold"
    result.attrs["units"] = "%"
    return result


def valid_data_coverage(data: xr.DataArray) -> xr.DataArray:
    """Return cosine-weighted spatial coverage by finite data, in percent."""
    required = {"latitude", "longitude"}
    missing = required - set(data.dims)
    if missing:
        raise ValueError(f"Data is missing spatial dimensions: {sorted(missing)}")
    weights = np.cos(np.deg2rad(data["latitude"])).broadcast_like(data)
    numerator = weights.where(data.notnull()).sum(("latitude", "longitude"), skipna=True)
    denominator = weights.sum(("latitude", "longitude"), skipna=True)
    coverage = xr.where(denominator > 0, 100.0 * numerator / denominator, np.nan)
    coverage.name = "valid_data_coverage"
    coverage.attrs["units"] = "%"
    return coverage


def warm_anomaly_centroid(
    anomaly: xr.DataArray, threshold: float = 2.0
) -> tuple[xr.DataArray, xr.DataArray]:
    """Return anomaly- and cosine-weighted warm-patch centroid coordinates."""
    weights = (
        anomaly.where(anomaly >= threshold)
        * np.cos(np.deg2rad(anomaly["latitude"]))
    )
    total = weights.sum(("latitude", "longitude"), skipna=True)
    longitude = (weights * anomaly["longitude"]).sum(
        ("latitude", "longitude"), skipna=True
    ) / total
    latitude = (weights * anomaly["latitude"]).sum(
        ("latitude", "longitude"), skipna=True
    ) / total
    return longitude.where(total > 0).rename("warm_centroid_longitude"), latitude.where(
        total > 0
    ).rename("warm_centroid_latitude")


def anomaly_persistence(
    anomaly: xr.DataArray, analysis_date: str, window: int, threshold: float = 2.0
) -> xr.DataArray:
    """Return percentage of valid observations meeting threshold in trailing window."""
    if window not in {7, 15, 30}:
        raise ValueError("Persistence window must be 7, 15, or 30 days")
    selected = anomaly.sel(time=slice(None, analysis_date)).isel(time=slice(-window, None))
    if selected.sizes.get("time", 0) == 0:
        raise ValueError(f"No anomaly observations available on or before {analysis_date}")
    valid_count = selected.notnull().sum("time")
    warm_count = ((selected >= threshold) & selected.notnull()).sum("time")
    result = xr.where(valid_count > 0, 100.0 * warm_count / valid_count, np.nan)
    result.name = "anomaly_persistence"
    result.attrs["units"] = "%"
    return result


def anomaly_profiles(anomaly: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    """Return latitude and longitude mean anomaly profiles."""
    latitude_profile = anomaly.mean("longitude", skipna=True).rename("latitude_profile")
    longitude_profile = anomaly.weighted(
        np.cos(np.deg2rad(anomaly["latitude"]))
    ).mean("latitude", skipna=True).rename("longitude_profile")
    return latitude_profile, longitude_profile
