"""Tests for date-specific diagnosis, spatial behaviour, and metrics."""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.daily_diagnosis import diagnose_date
from src.spatial_metrics import anomaly_persistence, warm_anomaly_centroid
from src.temporal_metrics import METRIC_COLUMNS, build_metrics_table


def daily_dataset(days: int = 10) -> xr.Dataset:
    values = np.arange(days, dtype=float)[:, None, None] + np.array([[20.0, 21.0], [22.0, 23.0]])
    return xr.Dataset(
        {"sst": (("time", "latitude", "longitude"), values)},
        coords={
            "time": pd.date_range("2026-01-01", periods=days, freq="D"),
            "latitude": [-10.0, 0.0],
            "longitude": [-90.0, -80.0],
        },
    )


def climatology() -> xr.Dataset:
    mean = xr.DataArray(
        np.stack([np.ones((2, 2)) * 20.0, np.ones((2, 2)) * 30.0]),
        dims=("month", "latitude", "longitude"),
        coords={"month": [1, 2], "latitude": [-10.0, 0.0], "longitude": [-90.0, -80.0]},
    )
    return xr.Dataset({
        "climatological_mean": mean,
        "climatological_standard_deviation": xr.ones_like(mean) * 2.0,
        "valid_observation_count": xr.ones_like(mean, dtype=int) * 30,
    })


def test_exact_date_selection_and_changes() -> None:
    fields, report = diagnose_date(daily_dataset(), "2026-01-10", climatology=climatology())
    np.testing.assert_allclose(fields.sst_current, [[29.0, 30.0], [31.0, 32.0]])
    np.testing.assert_allclose(fields.sst_daily_change, 1.0)
    np.testing.assert_allclose(fields.sst_seven_day_change, 7.0)
    assert report["date"] == "2026-01-10"


def test_unavailable_date_has_clear_error() -> None:
    with pytest.raises(ValueError, match="unavailable"):
        diagnose_date(daily_dataset(), "2025-12-31", climatology=climatology())


def test_climatology_is_matched_by_calendar_month() -> None:
    dataset = daily_dataset(2).assign_coords(time=pd.to_datetime(["2026-01-31", "2026-02-01"]))
    fields, _ = diagnose_date(dataset, "2026-02-01", climatology=climatology())
    np.testing.assert_allclose(fields.climatological_mean, 30.0)


def test_real_data_without_climatology_disables_anomalies() -> None:
    fields, report = diagnose_date(daily_dataset(), "2026-01-10", climatology=None)
    assert "sst_anomaly" not in fields
    assert "sst_z_score" not in fields
    assert report["warning"] is not None


def test_warm_anomaly_centroid_and_no_threshold_case() -> None:
    anomaly = xr.DataArray(
        [[0.0, 2.0], [0.0, 4.0]],
        dims=("latitude", "longitude"),
        coords={"latitude": [-10.0, 0.0], "longitude": [-90.0, -80.0]},
    )
    longitude, latitude = warm_anomaly_centroid(anomaly, 2.0)
    assert float(longitude) == -80.0
    assert -10.0 < float(latitude) < 0.0
    empty_lon, empty_lat = warm_anomaly_centroid(anomaly, 5.0)
    assert bool(empty_lon.isnull()) and bool(empty_lat.isnull())


def test_persistence_uses_requested_trailing_window() -> None:
    anomaly = daily_dataset(10).sst - 25.0
    persistence = anomaly_persistence(anomaly, "2026-01-10", 7, threshold=2.0)
    expected = ((anomaly.isel(time=slice(-7, None)) >= 2.0).sum("time") / 7 * 100)
    xr.testing.assert_allclose(persistence, expected.rename("anomaly_persistence"))


def test_metrics_table_has_one_row_per_date_and_required_structure() -> None:
    table = build_metrics_table(daily_dataset(), climatology())
    assert list(table.columns) == METRIC_COLUMNS
    assert len(table) == 10
    assert table.date.iloc[-1] == pd.Timestamp("2026-01-10")
    assert table.valid_data_coverage_percent.eq(100.0).all()
