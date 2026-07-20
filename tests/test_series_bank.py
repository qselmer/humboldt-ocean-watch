"""Tests for weighted statistics and long-format series-bank construction."""

from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.daily_climatology import select_climatology
from src.data_loader import COPERNICUS_CACHED_MODE
from src.feature_metadata import REQUIRED_FAMILIES
from src.series_bank import (
    SERIES_BANK_COLUMNS,
    build_series_bank,
    calculate_daily_features,
    weighted_mean,
    weighted_mean_result,
    weighted_median,
    weighted_quantile,
)


def test_weighted_mean_median_and_quantiles() -> None:
    values = np.array([1.0, 2.0, 3.0, 4.0])
    equal = np.ones(4)
    assert weighted_mean(values, equal) == 2.5
    assert weighted_median(values, equal) == 2.5
    assert weighted_quantile(values, equal, 0.10) == 1.0
    assert weighted_quantile(values, equal, 0.90) == 4.0
    assert weighted_mean(values, [1, 1, 1, 7]) == 3.4
    assert weighted_median(values, [1, 1, 1, 7]) > 3.0


def test_negative_weights_raise_and_zero_denominator_is_not_calculated() -> None:
    with pytest.raises(ValueError, match="negative"):
        weighted_mean([1, 2], [1, -1])
    result = weighted_mean_result(
        "sst.weighted_mean", [np.nan, np.nan], [1, 1],
        unit="degrees_Celsius",
    )
    assert result.status == "not_calculated"
    assert np.isnan(result.value)


def daily_field(values, name="anomaly", units="degrees_Celsius") -> xr.DataArray:
    return xr.DataArray(
        np.asarray(values, dtype=float),
        dims=("latitude", "longitude"),
        coords={"latitude": [-5.0, 0.0], "longitude": [-85.0, -84.0]},
        name=name,
        attrs={"units": units},
    )


def test_insufficient_coverage_and_no_threshold_centroid_are_not_calculated() -> None:
    low = daily_field([[3.0, np.nan], [np.nan, np.nan]])
    results = calculate_daily_features(low, variable="anomaly", minimum_valid_coverage=0.8)
    assert next(item for item in results if item.metric == "anomaly.weighted_mean").status == "not_calculated"
    cold = daily_field([[0.0, 1.0], [1.5, -1.0]])
    results = calculate_daily_features(cold, variable="anomaly", minimum_valid_coverage=0.0)
    centroids = [item for item in results if item.family == "centroid"]
    assert len(centroids) == 2
    assert all(item.status == "not_calculated" and "No valid cells" in (item.reason or "") for item in centroids)


def test_long_format_schema_and_all_required_families_for_one_date() -> None:
    times = pd.to_datetime(["2026-01-01"])
    base = np.array([[[22.0, 23.0], [24.0, 25.0]]])
    fields = xr.Dataset(
        {
            "sst": (("time", "latitude", "longitude"), base),
            "anomaly": (("time", "latitude", "longitude"), base - 20.0),
            "zscore": (("time", "latitude", "longitude"), (base - 20.0) / 2.0),
        },
        coords={"time": times, "latitude": [-5.0, 0.0], "longitude": [-85.0, -84.0]},
    )
    fields.sst.attrs["units"] = "degrees_Celsius"
    fields.anomaly.attrs["units"] = "degrees_Celsius"
    fields.zscore.attrs["units"] = "1"
    bank = build_series_bank(
        fields, minimum_valid_coverage=0.0,
        climatology_method="daily_smoothed", data_mode=COPERNICUS_CACHED_MODE,
    )
    assert list(bank.columns) == SERIES_BANK_COLUMNS
    assert set(bank.family) == REQUIRED_FAMILIES
    assert bank.date.nunique() == 1
    assert bank.metric.nunique() == len(bank)
    assert all(metric.split(".")[0] in {"sst", "anomaly", "zscore"} for metric in bank.metric)


def test_real_sst_cannot_select_synthetic_climatology(tmp_path: Path) -> None:
    synthetic = xr.Dataset()
    with pytest.raises(ValueError, match="cannot be used with real"):
        select_climatology(
            tmp_path / "missing_daily.nc", tmp_path / "missing_monthly.nc",
            allow_monthly_fallback=False,
            data_mode=COPERNICUS_CACHED_MODE,
            synthetic_climatology=synthetic,
        )
