"""Explicit threshold resolution for univariate thermal-event detection."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from src.daily_climatology import calendar_dayofyear, validate_daily_climatology
from src.quality_control import coordinate_names

ThresholdType = Literal[
    "fixed",
    "global_percentile",
    "daily_climatological",
    "custom_time_varying",
]


@dataclass(frozen=True)
class ThresholdResolution:
    """A threshold aligned exactly to the source-series calendar."""

    values: pd.Series
    threshold_type: ThresholdType
    description: str
    metadata: dict[str, Any] = field(default_factory=dict)


def _datetime_index(values: pd.Index | pd.Series | np.ndarray) -> pd.DatetimeIndex:
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(values, errors="raise"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Threshold dates are not readable datetimes") from exc
    if dates.tz is not None:
        dates = dates.tz_convert(None)
    dates = dates.normalize()
    if dates.has_duplicates:
        raise ValueError("Threshold resolution requires unique source dates")
    if not dates.is_monotonic_increasing:
        raise ValueError("Threshold resolution requires chronologically sorted source dates")
    return dates


def fixed_threshold(dates: pd.DatetimeIndex, value: float) -> ThresholdResolution:
    """Return one finite fixed threshold for every source date."""
    dates = _datetime_index(dates)
    if not np.isfinite(value):
        raise ValueError("Fixed threshold must be finite")
    return ThresholdResolution(
        values=pd.Series(float(value), index=dates, name="threshold"),
        threshold_type="fixed",
        description=f"fixed threshold {float(value):g}",
        metadata={"fixed_threshold": float(value)},
    )


def global_percentile_threshold(
    values: pd.Series,
    percentile: float,
    *,
    valid_mask: pd.Series | np.ndarray | None = None,
) -> ThresholdResolution:
    """Calculate one exact empirical percentile from finite eligible values."""
    dates = _datetime_index(values.index)
    if not 0.0 <= percentile <= 1.0:
        raise ValueError("Percentile must be between 0 and 1")
    numeric = pd.to_numeric(values, errors="coerce").astype(float)
    eligible = np.isfinite(numeric.to_numpy())
    if valid_mask is not None:
        mask = np.asarray(valid_mask, dtype=bool)
        if mask.shape != eligible.shape:
            raise ValueError("Percentile valid mask is incompatible with the source series")
        eligible &= mask
    if not eligible.any():
        raise ValueError("Global percentile threshold has no finite eligible observations")
    threshold = float(np.quantile(numeric.to_numpy()[eligible], percentile))
    return ThresholdResolution(
        values=pd.Series(threshold, index=dates, name="threshold"),
        threshold_type="global_percentile",
        description=f"global empirical percentile P{100.0 * percentile:g} ({threshold:g})",
        metadata={"percentile": percentile, "threshold": threshold},
    )


def _regional_climatological_threshold(
    climatology: xr.Dataset,
    variable: str,
) -> xr.DataArray:
    field = climatology[variable]
    spatial_dimensions = [name for name in ("latitude", "lat", "longitude", "lon") if name in field.dims]
    if not spatial_dimensions:
        return field
    latitude, longitude = coordinate_names(field)
    if set(field.dims) != {"climatological_day", latitude, longitude}:
        raise ValueError(
            f"Daily climatological threshold has unsupported dimensions: {field.dims}"
        )
    weights = np.cos(np.deg2rad(field[latitude])).broadcast_like(
        field.isel(climatological_day=0, drop=True)
    )
    valid = field.notnull()
    denominator = weights.where(valid).sum((latitude, longitude), skipna=True)
    numerator = (field * weights).sum((latitude, longitude), skipna=True)
    return xr.where(denominator > 0, numerator / denominator, np.nan)


def daily_climatological_threshold(
    dates: pd.DatetimeIndex,
    climatology: xr.Dataset,
    variable: str,
) -> ThresholdResolution:
    """Match a real smoothed daily threshold by stable month-day calendar.

    February 29 maps to canonical bin 60. Non-leap March 1 maps to bin 61,
    and December/January remain adjacent through the climatology's circular
    construction. No fixed or monthly substitute is permitted here.
    """
    dates = _datetime_index(dates)
    if variable not in {"threshold_p90", "threshold_p10"}:
        raise ValueError("Daily climatological event threshold must be threshold_p90 or threshold_p10")
    errors = validate_daily_climatology(climatology)
    if errors:
        raise ValueError("Daily climatology is incompatible: " + "; ".join(errors))
    if climatology.attrs.get("climatology_method") not in {None, "daily_smoothed"}:
        raise ValueError("Daily event thresholds require the smoothed daily climatology")
    regional = _regional_climatological_threshold(climatology, variable)
    date_array = xr.DataArray(dates.values, dims=("time",), coords={"time": dates.values})
    stable_days = calendar_dayofyear(date_array)
    matched = regional.sel(climatological_day=stable_days)
    threshold_values = np.asarray(matched.values, dtype=float).reshape(-1)
    if threshold_values.size != dates.size:
        raise ValueError("Daily climatological threshold did not align to every source date")
    if not np.isfinite(threshold_values).any():
        raise ValueError(f"Daily climatological variable {variable!r} has no finite matched values")
    reference = str(climatology.attrs.get("reference_period", "1991-2020"))
    return ThresholdResolution(
        values=pd.Series(threshold_values, index=dates, name="threshold"),
        threshold_type="daily_climatological",
        description=(
            f"smoothed daily climatological {variable} matched by stable month-day "
            f"calendar ({reference})"
        ),
        metadata={
            "climatological_threshold_variable": variable,
            "climatology_method": "daily_smoothed",
            "reference_period": reference,
            "calendar_mapping": climatology.attrs.get("calendar_mapping"),
            "leap_day_method": climatology.attrs.get("leap_day_method"),
        },
    )


def custom_time_varying_threshold(
    dates: pd.DatetimeIndex,
    threshold: pd.Series | pd.DataFrame,
    *,
    date_column: str = "date",
    threshold_column: str = "threshold",
) -> ThresholdResolution:
    """Align an explicit custom threshold series; unmatched dates remain missing."""
    dates = _datetime_index(dates)
    if isinstance(threshold, pd.Series):
        if not isinstance(threshold.index, pd.DatetimeIndex):
            raise ValueError("Custom threshold Series requires an explicit DatetimeIndex")
        custom_dates = pd.DatetimeIndex(pd.to_datetime(threshold.index, errors="raise"))
        if custom_dates.tz is not None:
            custom_dates = custom_dates.tz_convert(None)
        custom = pd.Series(
            pd.to_numeric(threshold, errors="coerce").to_numpy(dtype=float),
            index=custom_dates.normalize(),
        )
    elif isinstance(threshold, pd.DataFrame):
        if date_column in threshold.columns:
            custom_dates = pd.DatetimeIndex(pd.to_datetime(threshold[date_column], errors="raise"))
        elif isinstance(threshold.index, pd.DatetimeIndex):
            custom_dates = pd.DatetimeIndex(threshold.index)
        else:
            raise ValueError("Custom threshold DataFrame requires a date column or DatetimeIndex")
        if custom_dates.tz is not None:
            custom_dates = custom_dates.tz_convert(None)
        if threshold_column not in threshold.columns:
            raise ValueError(f"Custom threshold DataFrame is missing {threshold_column!r}")
        custom = pd.Series(
            pd.to_numeric(threshold[threshold_column], errors="coerce").to_numpy(dtype=float),
            index=custom_dates.normalize(),
        )
    else:
        raise TypeError("Custom threshold must be a pandas Series or DataFrame")
    if custom.index.has_duplicates:
        raise ValueError("Custom threshold contains duplicate dates")
    if np.isinf(custom.to_numpy()).any():
        raise ValueError("Custom threshold contains infinite values")
    aligned = custom.sort_index().reindex(dates)
    aligned.name = "threshold"
    return ThresholdResolution(
        values=aligned,
        threshold_type="custom_time_varying",
        description="custom time-varying threshold aligned by exact date",
        metadata={"matched_dates": int(aligned.notna().sum()), "requested_dates": len(dates)},
    )


def resolve_threshold(
    values: pd.Series,
    threshold_type: ThresholdType,
    *,
    fixed_value: float | None = None,
    percentile: float | None = None,
    climatology: xr.Dataset | None = None,
    climatological_variable: str | None = None,
    custom_threshold: pd.Series | pd.DataFrame | None = None,
    valid_mask: pd.Series | np.ndarray | None = None,
) -> ThresholdResolution:
    """Resolve exactly one requested threshold type without substitution."""
    dates = _datetime_index(values.index)
    if threshold_type == "fixed":
        if fixed_value is None:
            raise ValueError("Fixed threshold type requires fixed_value")
        return fixed_threshold(dates, fixed_value)
    if threshold_type == "global_percentile":
        if percentile is None:
            raise ValueError("Global percentile threshold requires percentile")
        return global_percentile_threshold(values, percentile, valid_mask=valid_mask)
    if threshold_type == "daily_climatological":
        if climatology is None or climatological_variable is None:
            raise ValueError(
                "Daily climatological threshold requires a compatible climatology and threshold variable"
            )
        return daily_climatological_threshold(dates, climatology, climatological_variable)
    if threshold_type == "custom_time_varying":
        if custom_threshold is None:
            raise ValueError("Custom time-varying threshold requires an explicit threshold series")
        return custom_time_varying_threshold(dates, custom_threshold)
    raise ValueError(f"Unsupported threshold type: {threshold_type!r}")
