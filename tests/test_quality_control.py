"""Deterministic tests for structural and recoverable quality control."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.quality_control import (
    recognize_units,
    temporal_observation_gate,
    validate_input,
)


def cube(times=None, values=None, units="degrees_Celsius") -> xr.DataArray:
    times = times if times is not None else pd.date_range("2026-01-01", periods=3, freq="D")
    values = values if values is not None else np.arange(len(times) * 4, dtype=float).reshape(len(times), 2, 2)
    data = xr.DataArray(
        values,
        dims=("time", "latitude", "longitude"),
        coords={"time": times, "latitude": [-5.0, 0.0], "longitude": [-85.0, -84.0]},
        name="sst",
        attrs={"units": units},
    )
    return data


def validate(data, **kwargs):
    return validate_input(data, minimum_temporal_observations=1, **kwargs)


def test_sorted_dates_and_regular_intervals_are_valid() -> None:
    cleaned, _, report = validate(cube())
    assert pd.DatetimeIndex(cleaned.time.values).is_monotonic_increasing
    assert report.temporal_interval_summary["regular"] is True
    assert report.overall_status == "valid"


def test_unsorted_dates_are_sorted_with_warning() -> None:
    data = cube(times=pd.to_datetime(["2026-01-03", "2026-01-01", "2026-01-02"]))
    cleaned, _, report = validate(data)
    assert list(pd.DatetimeIndex(cleaned.time.values).day) == [1, 2, 3]
    assert any("unsorted" in warning for warning in report.warnings)


def test_duplicate_date_policies() -> None:
    times = pd.to_datetime(["2026-01-01", "2026-01-01", "2026-01-02"])
    values = np.stack([np.zeros((2, 2)), np.ones((2, 2)) * 2, np.ones((2, 2)) * 3])
    with pytest.raises(ValueError, match="Duplicate"):
        validate(cube(times=times, values=values), duplicate_policy="error")
    first, _, _ = validate(cube(times=times, values=values), duplicate_policy="first")
    last, _, _ = validate(cube(times=times, values=values), duplicate_policy="last")
    mean, _, _ = validate(cube(times=times, values=values), duplicate_policy="mean")
    assert float(first.isel(time=0).mean()) == 0.0
    assert float(last.isel(time=0).mean()) == 2.0
    assert float(mean.isel(time=0).mean()) == 1.0


def test_irregular_intervals_warn_or_invalidate() -> None:
    data = cube(times=pd.to_datetime(["2026-01-01", "2026-01-02", "2026-01-05"]))
    _, _, warning = validate(data, allow_irregular_intervals=True)
    _, _, invalid = validate(data, allow_irregular_intervals=False)
    assert warning.temporal_interval_summary["regular"] is False
    assert any("irregular" in value for value in warning.warnings)
    assert invalid.overall_status == "invalid"


def test_missing_coordinates_and_unknown_units_are_fatal() -> None:
    with pytest.raises(ValueError, match="latitude"):
        validate(cube().drop_vars("latitude"))
    with pytest.raises(ValueError, match="Unknown"):
        validate(cube(units="mystery"))
    assert recognize_units("kelvin") == "kelvin"
    assert recognize_units("degrees_Celsius") == "celsius"


def test_incompatible_mask_and_negative_weights_are_fatal() -> None:
    bad_mask = xr.DataArray(np.ones(3), dims="latitude", coords={"latitude": [-5.0, 0.0, 5.0]})
    with pytest.raises(ValueError, match="mask"):
        validate(cube(), mask=bad_mask)
    negative = xr.DataArray([-1.0, 1.0], dims="latitude", coords={"latitude": [-5.0, 0.0]})
    with pytest.raises(ValueError, match="negative"):
        validate(cube(), weights=negative)


def test_valid_coverage_insufficient_coverage_and_all_nan_detection() -> None:
    values = np.arange(12, dtype=float).reshape(3, 2, 2)
    values[:, 0, :] = np.nan
    _, _, report = validate(cube(values=values), minimum_valid_coverage=0.8)
    assert report.valid_data_fraction == 0.5
    assert report.weighted_spatial_coverage < 0.8
    assert any("below" in warning for warning in report.warnings)

    all_nan = np.full((3, 2, 2), np.nan)
    _, _, invalid = validate(cube(values=all_nan))
    assert len(invalid.all_nan_dates) == 3
    assert invalid.overall_status == "invalid"


def test_constant_dates_and_insufficient_temporal_observations() -> None:
    values = np.stack([np.ones((2, 2)), np.arange(4).reshape(2, 2), np.ones((2, 2)) * 3])
    _, _, report = validate(cube(values=values))
    assert len(report.constant_field_dates) == 2
    result = temporal_observation_gate(
        "series.trend", 3, 10, unit="degrees_Celsius per day", family="trend"
    )
    assert result is not None and result.status == "not_calculated"
