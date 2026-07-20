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
