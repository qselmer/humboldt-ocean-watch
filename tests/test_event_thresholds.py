"""Deterministic tests for explicit univariate event thresholds."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.daily_climatology import calendar_coordinates
from src.event_thresholds import (
    custom_time_varying_threshold,
    daily_climatological_threshold,
    fixed_threshold,
    global_percentile_threshold,
    resolve_threshold,
)


def _daily_climatology() -> xr.Dataset:
    days = np.arange(1, 367, dtype=float)
    shape = (366, 1, 1)
    p90 = days[:, None, None]
    fields = {
        "climatology_mean": np.broadcast_to(p90 - 1.0, shape).copy(),
        "climatology_median": np.broadcast_to(p90 - 1.0, shape).copy(),
        "climatology_std": np.ones(shape),
        "threshold_p10": np.broadcast_to(p90 - 2.0, shape).copy(),
        "threshold_p90": np.broadcast_to(p90, shape).copy(),
        "observation_count": np.full(shape, 30, dtype=np.int64),
    }
    dataset = xr.Dataset(
        {
            name: (("climatological_day", "latitude", "longitude"), values)
            for name, values in fields.items()
        },
        coords={**calendar_coordinates(), "latitude": [-5.0], "longitude": [-85.0]},
        attrs={
            "climatology_method": "daily_smoothed",
            "reference_period": "1991-2020",
            "calendar_mapping": "stable month-day mapping on canonical leap year 2000",
            "leap_day_method": "February 29 is stable bin 60",
        },
    )
    for name in fields:
        dataset[name].attrs["units"] = (
            "1" if name == "observation_count" else "degrees_Celsius"
        )
    return dataset


def test_fixed_threshold_is_constant_and_finite() -> None:
    dates = pd.date_range("2024-01-01", periods=3)
    result = fixed_threshold(dates, 2.0)
    np.testing.assert_allclose(result.values, 2.0)
    assert result.threshold_type == "fixed"
    with pytest.raises(ValueError, match="finite"):
        fixed_threshold(dates, np.inf)


def test_global_percentile_uses_only_eligible_finite_values() -> None:
    dates = pd.date_range("2024-01-01", periods=5)
    values = pd.Series([0.0, 1.0, 2.0, np.nan, 100.0], index=dates)
    result = global_percentile_threshold(
        values, 0.5, valid_mask=np.array([True, True, True, True, False])
    )
    np.testing.assert_allclose(result.values, 1.0)
    assert result.metadata["percentile"] == 0.5


def test_custom_time_varying_threshold_aligns_exact_dates_without_substitution() -> None:
    dates = pd.date_range("2024-01-01", periods=3)
    custom = pd.Series([10.0, 12.0], index=[dates[0], dates[2]])
    result = custom_time_varying_threshold(dates, custom)
    assert result.values.iloc[0] == 10.0
    assert np.isnan(result.values.iloc[1])
    assert result.values.iloc[2] == 12.0
    assert result.metadata["matched_dates"] == 2


def test_daily_climatological_threshold_uses_stable_month_day_calendar() -> None:
    dates = pd.DatetimeIndex(
        ["2022-12-31", "2023-01-01", "2023-02-28", "2023-03-01", "2024-02-29"]
    )
    result = daily_climatological_threshold(dates, _daily_climatology(), "threshold_p90")
    np.testing.assert_allclose(result.values, [366.0, 1.0, 59.0, 61.0, 60.0])
    assert "stable month-day" in result.description
    assert result.metadata["leap_day_method"] == "February 29 is stable bin 60"


def test_daily_p10_is_supported_and_missing_climatology_is_not_substituted() -> None:
    dates = pd.date_range("2024-01-01", periods=2)
    result = daily_climatological_threshold(dates, _daily_climatology(), "threshold_p10")
    np.testing.assert_allclose(result.values, [-1.0, 0.0])
    values = pd.Series([20.0, 21.0], index=dates)
    with pytest.raises(ValueError, match="requires a compatible climatology"):
        resolve_threshold(
            values,
            "daily_climatological",
            climatological_variable="threshold_p90",
        )


def test_threshold_dates_must_be_sorted_and_unique() -> None:
    with pytest.raises(ValueError, match="sorted"):
        fixed_threshold(pd.DatetimeIndex(["2024-01-02", "2024-01-01"]), 1.0)
    with pytest.raises(ValueError, match="unique"):
        fixed_threshold(pd.DatetimeIndex(["2024-01-01", "2024-01-01"]), 1.0)
    with pytest.raises(ValueError, match="unique"):
        fixed_threshold(
            pd.DatetimeIndex(["2024-01-01T00:00:00", "2024-01-01T12:00:00"]), 1.0
        )
