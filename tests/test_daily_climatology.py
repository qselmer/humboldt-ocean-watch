"""Tests for the daily smoothed climatology and explicit fallback policy."""

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

import scripts.build_daily_climatology as builder
from src.daily_climatology import (
    DAILY_METHOD,
    MONTHLY_METHOD,
    build_daily_statistics,
    calendar_coordinates,
    calendar_dayofyear,
    circular_day_distance,
    circular_rolling_mean,
    match_climatology,
    select_climatology,
)


def _sst(dates: list[str], values: list[float]) -> xr.DataArray:
    return xr.DataArray(
        np.asarray(values, dtype=float)[:, None, None],
        dims=("time", "latitude", "longitude"),
        coords={
            "time": pd.to_datetime(dates),
            "latitude": [-5.0],
            "longitude": [-85.0],
        },
    )


def test_december_january_sampling_window_is_circular() -> None:
    days = calendar_dayofyear(xr.DataArray(pd.to_datetime(["2001-12-31", "2001-01-02"]), dims="time"))
    distance = circular_day_distance(days, 1)
    np.testing.assert_array_equal(distance, [1, 1])


def test_leap_day_has_explicit_stable_calendar_bin() -> None:
    days = calendar_dayofyear(
        xr.DataArray(pd.to_datetime(["2000-02-29", "2000-03-01", "2001-03-01"]), dims="time")
    )
    np.testing.assert_array_equal(days, [60, 61, 61])


def test_month_boundary_continuity_uses_calendar_distance() -> None:
    days = calendar_dayofyear(
        xr.DataArray(pd.to_datetime(["2001-01-31", "2001-02-01"]), dims="time")
    )
    assert int(circular_day_distance(days, int(days[0])).isel(time=1)) == 1


def test_daily_climatology_matches_analysis_month_and_day() -> None:
    day = np.arange(1, 367)
    climatology = xr.Dataset(
        {
            "climatological_mean": ("dayofyear", day.astype(float)),
            "climatological_standard_deviation": ("dayofyear", np.ones(366)),
        },
        coords={"dayofyear": day},
        attrs={"climatology_method": DAILY_METHOD},
    )
    dates = xr.DataArray(pd.to_datetime(["2023-03-01", "2024-02-29"]), dims="time")
    mean, _, _ = match_climatology(climatology, dates)
    np.testing.assert_array_equal(mean, [61.0, 60.0])


def test_31_day_smoothing_wraps_and_handles_constant_fields() -> None:
    field = xr.DataArray(
        np.arange(1, 367, dtype=float), dims="dayofyear", coords={"dayofyear": np.arange(1, 367)}
    )
    smoothed = circular_rolling_mean(field, 31)
    expected_first = np.mean(np.r_[np.arange(352, 367), np.arange(1, 17)])
    assert float(smoothed.sel(dayofyear=1)) == expected_first
    constant = xr.ones_like(field) * 4.0
    np.testing.assert_allclose(circular_rolling_mean(constant, 31), 4.0)


def test_missing_observations_are_ignored_and_counted() -> None:
    result = build_daily_statistics(
        _sst(["2000-01-01", "2001-01-01", "2002-01-01"], [20.0, np.nan, 24.0]),
        sampling_half_window_days=0,
        smoothing_window_days=1,
    )
    assert int(result.observation_count.sel(climatological_day=1).item()) == 2
    assert float(result.climatology_mean.sel(climatological_day=1).item()) == 22.0
    assert float(result.climatology_median.sel(climatological_day=1).item()) == 22.0


def _write_climatology(path: Path, dimension: str) -> None:
    coordinate = [1]
    shape = (1, 1, 1)
    xr.Dataset(
        {
            "climatological_mean": ((dimension, "latitude", "longitude"), np.ones(shape)),
            "climatological_standard_deviation": ((dimension, "latitude", "longitude"), np.ones(shape)),
            "valid_observation_count": ((dimension, "latitude", "longitude"), np.ones(shape, dtype=int)),
        },
        coords={dimension: coordinate, "latitude": [-5.0], "longitude": [-85.0]},
    ).to_netcdf(path)


def _valid_daily_climatology() -> xr.Dataset:
    shape = (366, 1, 1)
    fields = {
        "climatology_mean": np.full(shape, 20.0),
        "climatology_median": np.full(shape, 20.0),
        "climatology_std": np.full(shape, 1.0),
        "threshold_p10": np.full(shape, 19.0),
        "threshold_p90": np.full(shape, 21.0),
        "observation_count": np.full(shape, 10, dtype=np.int64),
    }
    dataset = xr.Dataset(
        {name: (("climatological_day", "latitude", "longitude"), values) for name, values in fields.items()},
        coords={**calendar_coordinates(), "latitude": [-5.0], "longitude": [-85.0]},
    )
    for name in fields:
        dataset[name].attrs["units"] = "1" if name == "observation_count" else "degrees_Celsius"
    return dataset


def test_primary_daily_then_explicit_monthly_fallback(tmp_path: Path) -> None:
    daily = tmp_path / "daily.nc"
    monthly = tmp_path / "monthly.nc"
    _write_climatology(monthly, "month")
    fallback = select_climatology(daily, monthly)
    assert fallback.method == MONTHLY_METHOD
    assert fallback.warning and "monthly sensitivity" in fallback.warning

    _valid_daily_climatology().to_netcdf(daily)
    primary = select_climatology(daily, monthly)
    assert primary.method == DAILY_METHOD
    assert primary.warning is None


def test_copernicus_open_dataset_uses_supported_lazy_chunking(
    tmp_path: Path, monkeypatch
) -> None:
    captured: dict[str, object] = {}
    source = xr.Dataset(
        {
            "analysed_sst": (
                ("time", "latitude", "longitude"),
                np.array([[[293.15]], [[294.15]]]),
                {"units": "kelvin"},
            )
        },
        coords={
            "time": pd.to_datetime(["2001-01-01", "2001-12-31"]),
            "latitude": [-5.0],
            "longitude": [-85.0],
        },
    )

    def fake_open_dataset(
        dataset_id=None, variables=None, minimum_longitude=None, maximum_longitude=None,
        minimum_latitude=None, maximum_latitude=None, start_datetime=None,
        end_datetime=None, chunk_size_limit=-1,
    ):
        captured.update(locals())
        return source

    result = _valid_daily_climatology()
    monkeypatch.setattr("copernicusmarine.open_dataset", fake_open_dataset)
    monkeypatch.setattr(builder, "build_daily_statistics", lambda *args, **kwargs: result)

    output = tmp_path / "daily.nc"
    builder.build_daily_climatology(2001, 2001, output, checkpoint_dir=tmp_path / "checkpoints")

    assert "chunks" not in captured
    assert captured["chunk_size_limit"] == -1
    assert captured["dataset_id"] == "METOFFICE-GLO-SST-L4-REP-OBS-SST"
    assert captured["variables"] == ["analysed_sst"]
    assert captured["minimum_longitude"] == -90.0
    assert captured["maximum_longitude"] == -80.0
    assert captured["minimum_latitude"] == -10.0
    assert captured["maximum_latitude"] == 0.0
    assert captured["start_datetime"] == "2001-01-01T00:00:00"
    assert captured["end_datetime"] == "2001-12-31T23:59:59"
