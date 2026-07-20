"""Offline tests for incremental climatology accumulation."""

import numpy as np
import pandas as pd
import xarray as xr

from scripts.build_climatology import finalize_accumulator, yearly_accumulator


def test_yearly_accumulator_converts_kelvin_and_ignores_missing_values() -> None:
    values = np.array([[[273.15]], [[275.15]], [[np.nan]]])
    dataset = xr.Dataset(
        {"analysed_sst": (("time", "latitude", "longitude"), values)},
        coords={
            "time": pd.to_datetime(["2000-01-01", "2000-01-02", "2000-01-03"]),
            "latitude": [-5.0],
            "longitude": [-85.0],
        },
    )
    dataset.analysed_sst.attrs["units"] = "kelvin"
    accumulator = yearly_accumulator(dataset)
    result = finalize_accumulator(accumulator)
    np.testing.assert_allclose(result.climatological_mean.sel(month=1), [[1.0]])
    np.testing.assert_allclose(result.climatological_standard_deviation.sel(month=1), [[1.0]])
    assert int(result.valid_observation_count.sel(month=1).item()) == 2
