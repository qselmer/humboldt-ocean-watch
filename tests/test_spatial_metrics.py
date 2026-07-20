"""Scientific unit tests for latitude-weighted spatial metrics."""

import numpy as np
import xarray as xr

from src.spatial_metrics import (
    area_weighted_mean,
    percentage_area_above_threshold,
    percentage_area_at_or_above_threshold,
)


def test_cosine_latitude_weighted_spatial_mean() -> None:
    field = xr.DataArray(
        [[10.0], [20.0]],
        dims=("latitude", "longitude"),
        coords={"latitude": [0.0, 60.0], "longitude": [-85.0]},
    )

    mean = area_weighted_mean(field)

    # cos(0°)=1 and cos(60°)=0.5: (10*1 + 20*0.5) / 1.5
    np.testing.assert_allclose(float(mean), 40.0 / 3.0)


def test_weighted_mean_ignores_missing_values() -> None:
    field = xr.DataArray(
        [[10.0, np.nan], [20.0, 30.0]],
        dims=("latitude", "longitude"),
        coords={"latitude": [0.0, 60.0], "longitude": [-90.0, -80.0]},
    )

    mean = area_weighted_mean(field)

    # Valid weights are 1, 0.5, and 0.5.
    np.testing.assert_allclose(float(mean), 17.5)


def test_percentage_of_valid_ocean_area_above_threshold_ignores_nan() -> None:
    field = xr.DataArray(
        [[2.0, np.nan], [4.0, 1.0]],
        dims=("latitude", "longitude"),
        coords={"latitude": [0.0, 60.0], "longitude": [-90.0, -80.0]},
    )

    percentage = percentage_area_above_threshold(field, threshold=3.0)

    # Above-threshold weight is 0.5; total valid weight is 1 + 0.5 + 0.5 = 2.
    np.testing.assert_allclose(float(percentage), 25.0)
    assert percentage.attrs["units"] == "%"


def test_area_weighted_threshold_is_inclusive() -> None:
    field = xr.DataArray(
        [[2.0], [1.0]],
        dims=("latitude", "longitude"),
        coords={"latitude": [60.0, 0.0], "longitude": [-85.0]},
    )
    result = percentage_area_at_or_above_threshold(field, 2.0)
    np.testing.assert_allclose(float(result), 100.0 / 3.0)
