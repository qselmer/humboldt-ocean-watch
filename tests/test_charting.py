"""Tests for stable profile tables and robust chart domains."""

import numpy as np
import pandas as pd
import xarray as xr

from src.charting import (
    PROFILE_VALUE_COLUMN,
    finite_numeric_domain,
    profile_chart,
    profile_dataframe,
    temporal_chart,
)


def test_profile_dataframe_uses_stable_value_column_name() -> None:
    profile = xr.DataArray(
        [0.5, 1.0], dims="latitude", coords={"latitude": [-10.0, 0.0]}, name="any_name"
    )
    frame = profile_dataframe(profile, "latitude")
    assert list(frame.columns) == ["latitude", PROFILE_VALUE_COLUMN]
    np.testing.assert_allclose(frame[PROFILE_VALUE_COLUMN], [0.5, 1.0])


def test_missing_and_infinite_profile_values_are_removed() -> None:
    profile = xr.DataArray(
        [np.nan, np.inf, -np.inf],
        dims="longitude",
        coords={"longitude": [-90.0, -85.0, -80.0]},
    )
    assert profile_dataframe(profile, "longitude").empty
    assert profile_chart(profile, coordinate="longitude", title="Empty") is None


def test_constant_profile_gets_nonzero_padded_y_domain() -> None:
    domain = finite_numeric_domain(pd.Series([2.0, 2.0]), padding_fraction=0.05)
    assert domain is not None
    assert domain[0] < 2.0 < domain[1]
    chart = profile_chart(
        xr.DataArray([2.0, 2.0], dims="latitude", coords={"latitude": [-10.0, 0.0]}),
        coordinate="latitude",
        title="Constant",
    )
    assert chart is not None


def test_temporal_chart_ignores_nonfinite_values_and_handles_constant_series() -> None:
    frame = pd.DataFrame(
        {"date": pd.date_range("2026-01-01", periods=3), "value": [1.0, np.nan, 1.0]}
    )
    assert temporal_chart(frame, x="date", columns=["value"], y_title="Value", title="Test") is not None
