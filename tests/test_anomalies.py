"""Scientific unit tests for SST anomalies and standardized anomalies."""

import numpy as np
import xarray as xr

from src.anomalies import calculate_anomaly, calculate_z_score


def test_anomaly_is_current_sst_minus_climatology() -> None:
    current = xr.DataArray([25.0, 27.5, np.nan], dims="point")
    climatology = xr.DataArray([24.0, 26.0, 25.0], dims="point")

    anomaly = calculate_anomaly(current, climatology)

    np.testing.assert_allclose(anomaly.values, [1.0, 1.5, np.nan], equal_nan=True)
    assert anomaly.attrs["units"] == "degrees_Celsius"


def test_z_score_is_anomaly_divided_by_climatological_std() -> None:
    anomaly = xr.DataArray([1.0, -2.0, np.nan], dims="point")
    climatological_std = xr.DataArray([0.5, 2.0, 1.0], dims="point")

    z_score = calculate_z_score(anomaly, climatological_std)

    np.testing.assert_allclose(z_score.values, [2.0, -1.0, np.nan], equal_nan=True)
    assert z_score.attrs["units"] == "1"


def test_zero_climatological_std_returns_nan_not_infinity() -> None:
    anomaly = xr.DataArray([1.0, 0.0, 3.0], dims="point")
    climatological_std = xr.DataArray([0.0, 0.0, 1.5], dims="point")

    z_score = calculate_z_score(anomaly, climatological_std)

    np.testing.assert_allclose(z_score.values, [np.nan, np.nan, 2.0], equal_nan=True)
    assert not bool(np.isinf(z_score).any())
