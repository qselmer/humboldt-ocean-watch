"""Simple SST climatology helpers."""

from __future__ import annotations

import xarray as xr


def daily_climatology(sst: xr.DataArray) -> xr.DataArray:
    """Compute a day-of-year climatology from a time-indexed SST array."""
    if "time" not in sst.dims:
        raise ValueError("SST must include a 'time' dimension")
    return sst.groupby("time.dayofyear").mean("time", skipna=True).rename("sst_climatology")
