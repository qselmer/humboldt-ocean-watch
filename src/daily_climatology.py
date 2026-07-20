"""Stable-calendar SST climatology calculation, validation, and selection."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

CANONICAL_YEAR = 2000
DAILY_METHOD = "daily_smoothed"
MONTHLY_METHOD = "monthly"
DAILY_VARIABLES = (
    "climatology_mean", "climatology_median", "climatology_std",
    "threshold_p10", "threshold_p90", "observation_count",
)


@dataclass(frozen=True)
class ClimatologySelection:
    dataset: xr.Dataset | None
    method: str | None
    path: str | None = None
    fallback_used: bool = False
    warning: str | None = None


def calendar_dayofyear(values: xr.DataArray | np.ndarray) -> xr.DataArray:
    """Map timestamps to stable month-day bins on a canonical leap year.

    February 29 is always bin 60. Non-leap years simply contribute no direct
    February 29 sample; their March 1 remains bin 61.
    """
    raw = values.values if isinstance(values, xr.DataArray) else values
    dates = pd.DatetimeIndex(raw)
    mapped = np.asarray(
        [pd.Timestamp(CANONICAL_YEAR, value.month, value.day).dayofyear for value in dates],
        dtype=np.int16,
    )
    return xr.DataArray(
        mapped,
        dims=values.dims if isinstance(values, xr.DataArray) else ("time",),
        coords=values.coords if isinstance(values, xr.DataArray) else None,
        name="climatological_day",
    )


def calendar_coordinates() -> dict[str, tuple[str, np.ndarray] | np.ndarray]:
    dates = pd.date_range(f"{CANONICAL_YEAR}-01-01", f"{CANONICAL_YEAR}-12-31")
    return {
        "climatological_day": np.arange(1, 367, dtype=np.int16),
        "month": ("climatological_day", dates.month.to_numpy(dtype=np.int8)),
        "day": ("climatological_day", dates.day.to_numpy(dtype=np.int8)),
        "month_day": ("climatological_day", dates.strftime("%m-%d").to_numpy()),
    }


def circular_day_distance(day: xr.DataArray, target: int, period: int = 366) -> xr.DataArray:
    delta = abs(day - target)
    return xr.apply_ufunc(np.minimum, delta, period - delta)


def circular_rolling_mean(field: xr.DataArray, window: int) -> xr.DataArray:
    """Return a centered NaN-aware circular mean; ``window`` must be odd."""
    dimension = "climatological_day" if "climatological_day" in field.dims else "dayofyear"
    if window < 1 or window % 2 == 0:
        raise ValueError("Smoothing window must be a positive odd number")
    if dimension not in field.dims:
        raise ValueError("Field must include a climatological calendar dimension")
    half = window // 2
    extended = xr.concat(
        [field.isel({dimension: slice(-half, None)}), field, field.isel({dimension: slice(0, half)})],
        dim=dimension,
    )
    smoothed = extended.rolling({dimension: window}, center=True, min_periods=1).mean(skipna=True)
    result = smoothed.isel({dimension: slice(half, half + field.sizes[dimension])})
    return result.assign_coords({dimension: field[dimension]})


def _quantile(pooled: xr.DataArray, probability: float, template: xr.DataArray) -> xr.DataArray:
    if pooled.sizes["time"] == 0:
        return xr.full_like(template, np.nan)
    return pooled.quantile(probability, dim="time", skipna=True).drop_vars("quantile")


def build_daily_statistics(
    sst: xr.DataArray,
    *,
    sampling_half_window_days: int = 5,
    smoothing_window_days: int = 31,
    calculate_percentiles: bool = True,
) -> xr.Dataset:
    """Calculate exact pooled daily statistics and circular smoothing.

    Mean, P10, and P90 are smoothed. Median and population standard deviation
    are deliberately unsmoothed. Exact percentiles require raw observations;
    when disabled their state is explicit and no approximation is substituted.
    """
    if "time" not in sst.dims or sst.sizes["time"] == 0:
        raise ValueError("SST must include a non-empty 'time' dimension")
    if sampling_half_window_days < 0:
        raise ValueError("Sampling half-window must be non-negative")
    days = calendar_dayofyear(sst.time)
    template = sst.isel(time=0, drop=True).astype(float)
    pieces: list[xr.Dataset] = []
    for target in range(1, 367):
        pooled = sst.where(circular_day_distance(days, target) <= sampling_half_window_days, drop=True)
        if pooled.sizes["time"]:
            count = pooled.count("time").astype("int64")
            mean = pooled.mean("time", skipna=True).where(count > 0)
            std = pooled.std("time", skipna=True, ddof=0).where(count > 0)
            median = _quantile(pooled, 0.5, template) if calculate_percentiles else xr.full_like(template, np.nan)
            p10 = _quantile(pooled, 0.1, template) if calculate_percentiles else xr.full_like(template, np.nan)
            p90 = _quantile(pooled, 0.9, template) if calculate_percentiles else xr.full_like(template, np.nan)
        else:
            count = xr.zeros_like(template, dtype="int64")
            mean = std = median = p10 = p90 = xr.full_like(template, np.nan)
        pieces.append(xr.Dataset({
            "climatology_mean_unsmoothed": mean,
            "climatology_median": median,
            "climatology_std": std,
            "threshold_p10_unsmoothed": p10,
            "threshold_p90_unsmoothed": p90,
            "observation_count": count,
        }).expand_dims(climatological_day=[target]))
    raw = xr.concat(pieces, dim="climatological_day").assign_coords(calendar_coordinates())
    result = xr.Dataset({
        "climatology_mean": circular_rolling_mean(raw.climatology_mean_unsmoothed, smoothing_window_days),
        "climatology_median": raw.climatology_median,
        "climatology_std": raw.climatology_std,
        "threshold_p10": circular_rolling_mean(raw.threshold_p10_unsmoothed, smoothing_window_days),
        "threshold_p90": circular_rolling_mean(raw.threshold_p90_unsmoothed, smoothing_window_days),
        "observation_count": raw.observation_count,
        "climatology_mean_unsmoothed": raw.climatology_mean_unsmoothed,
    }, attrs={
        "climatology_method": DAILY_METHOD,
        "calendar_mapping": "stable month-day mapping on canonical leap year 2000",
        "leap_day_method": "February 29 is bin 60; non-leap years have no direct bin-60 sample but may contribute through the circular ±window",
        "sampling_half_window_days": sampling_half_window_days,
        "smoothing_window_days": smoothing_window_days,
        "percentile_calculation_state": "calculated" if calculate_percentiles else "not_calculated",
    })
    statistics = {
        "climatology_mean": "smoothed pooled mean", "climatology_median": "unsmoothed pooled median",
        "climatology_std": "unsmoothed pooled population standard deviation",
        "threshold_p10": "smoothed exact pooled 10th percentile",
        "threshold_p90": "smoothed exact pooled 90th percentile",
        "observation_count": "unsmoothed valid observation count",
        "climatology_mean_unsmoothed": "unsmoothed pooled mean",
    }
    for name, statistic in statistics.items():
        result[name].attrs.update(long_name=statistic, statistic=statistic)
        if name != "observation_count":
            result[name].attrs["units"] = "degrees_Celsius"
        else:
            result[name].attrs["units"] = "1"
    return result


def _variable_names(climatology: xr.Dataset) -> tuple[str, str, str | None]:
    if "climatological_day" in climatology.dims:
        return "climatology_mean", "climatology_std", "threshold_p90"
    if "dayofyear" in climatology.dims:  # compatibility with early daily test files
        return "climatological_mean", "climatological_standard_deviation", None
    return "climatological_mean", "climatological_standard_deviation", None


def match_climatology(
    climatology: xr.Dataset, dates: xr.DataArray
) -> tuple[xr.DataArray, xr.DataArray, xr.DataArray | None]:
    """Match one daily or monthly baseline without blending methods."""
    mean_name, std_name, p90_name = _variable_names(climatology)
    if "climatological_day" in climatology.dims:
        indexer = calendar_dayofyear(dates)
        return (
            climatology[mean_name].sel(climatological_day=indexer),
            climatology[std_name].sel(climatological_day=indexer),
            climatology[p90_name].sel(climatological_day=indexer) if p90_name else None,
        )
    if "dayofyear" in climatology.dims:
        indexer = calendar_dayofyear(dates).rename("dayofyear")
        return climatology[mean_name].sel(dayofyear=indexer), climatology[std_name].sel(dayofyear=indexer), None
    if "month" in climatology.dims:
        return climatology[mean_name].sel(month=dates.dt.month), climatology[std_name].sel(month=dates.dt.month), None
    raise ValueError("Climatology must have 'climatological_day', 'dayofyear', or 'month'")


def validate_daily_climatology(dataset: xr.Dataset) -> list[str]:
    """Return structural/scientific validation errors for a daily climatology."""
    errors: list[str] = []
    if dataset.sizes.get("climatological_day") != 366:
        errors.append("climatological_day must contain exactly 366 values")
    for coord in ("latitude", "longitude", "month", "day", "month_day"):
        if coord not in dataset.coords:
            errors.append(f"missing coordinate: {coord}")
    for name in DAILY_VARIABLES:
        if name not in dataset.data_vars:
            errors.append(f"missing variable: {name}")
    if errors:
        return errors
    expected = pd.date_range("2000-01-01", "2000-12-31").strftime("%m-%d").tolist()
    if dataset.month_day.astype(str).values.tolist() != expected:
        errors.append("month_day does not contain the complete ordered 366-day calendar")
    for name in DAILY_VARIABLES[:-1]:
        values = np.asarray(dataset[name].values)
        if np.isinf(values).any():
            errors.append(f"{name} contains infinite values")
        if not np.isfinite(values).any():
            errors.append(f"{name} contains no finite values")
        if str(dataset[name].attrs.get("units")) != "degrees_Celsius":
            errors.append(f"{name} units must be degrees_Celsius")
    if bool((dataset.climatology_std < 0).any()):
        errors.append("climatology_std contains negative values")
    if bool(((dataset.threshold_p10 > dataset.threshold_p90) & dataset.threshold_p10.notnull() & dataset.threshold_p90.notnull()).any()):
        errors.append("threshold_p10 exceeds threshold_p90")
    if bool((dataset.observation_count < 0).any()):
        errors.append("observation_count contains negative values")
    spatial = (dataset.sizes.get("latitude"), dataset.sizes.get("longitude"))
    for name in DAILY_VARIABLES:
        if (dataset[name].sizes.get("latitude"), dataset[name].sizes.get("longitude")) != spatial:
            errors.append(f"{name} has incompatible spatial dimensions")
    return errors


def select_climatology(
    daily_path: str | Path,
    monthly_path: str | Path,
    *, primary_method: str = DAILY_METHOD, fallback_method: str = MONTHLY_METHOD,
    allow_monthly_fallback: bool = True, data_mode: str | None = None,
    synthetic_climatology: xr.Dataset | None = None,
) -> ClimatologySelection:
    """Select daily, explicit monthly fallback, then synthetic-for-demo only."""
    if primary_method != DAILY_METHOD or fallback_method != MONTHLY_METHOD:
        raise ValueError("Supported climatology order is daily_smoothed then monthly")
    daily, monthly = Path(daily_path), Path(monthly_path)
    if daily.exists():
        dataset = xr.load_dataset(daily)
        errors = validate_daily_climatology(dataset)
        if not errors:
            dataset.attrs["climatology_method"] = DAILY_METHOD
            dataset.attrs["climatology_file"] = str(daily)
            dataset.attrs["climatology_fallback_used"] = False
            return ClimatologySelection(dataset, DAILY_METHOD, str(daily))
    if allow_monthly_fallback and monthly.exists():
        dataset = xr.load_dataset(monthly)
        dataset.attrs.update(
            climatology_method=MONTHLY_METHOD,
            climatology_file=str(monthly),
            climatology_fallback_used=True,
        )
        return ClimatologySelection(dataset, MONTHLY_METHOD, str(monthly), True,
            "Daily smoothed climatology is unavailable or invalid; using the real monthly sensitivity baseline.")
    if synthetic_climatology is not None:
        if data_mode != "Synthetic demonstration data":
            raise ValueError("Synthetic climatology cannot be used with real Copernicus SST")
        synthetic_climatology.attrs.update(climatology_method="synthetic", climatology_file="generated")
        return ClimatologySelection(synthetic_climatology, "synthetic", "generated", True,
            "Using a synthetic climatology with synthetic demonstration SST.")
    return ClimatologySelection(None, None, None, False, "No compatible real climatology is available.")
