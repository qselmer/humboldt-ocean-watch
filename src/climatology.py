"""Simple SST climatology helpers."""

from __future__ import annotations

import xarray as xr


def daily_climatology(sst: xr.DataArray) -> xr.DataArray:
    """Compute a day-of-year climatology from a time-indexed SST array."""
    if "time" not in sst.dims:
        raise ValueError("SST must include a 'time' dimension")
    return sst.groupby("time.dayofyear").mean("time", skipna=True).rename("sst_climatology")


def monthly_climatology(sst: xr.DataArray) -> xr.Dataset:
    """Calculate monthly mean and standard deviation from daily SST."""
    if "time" not in sst.dims:
        raise ValueError("SST must include a 'time' dimension")
    if sst.sizes.get("time", 0) < 2:
        raise ValueError("Monthly climatology requires at least two daily SST fields")
    grouped = sst.groupby("time.month")
    mean = grouped.mean("time", skipna=True).rename("climatological_mean")
    standard_deviation = grouped.std("time", skipna=True, ddof=0).rename(
        "climatological_standard_deviation"
    )
    mean.attrs.update(units="degrees_Celsius", climatology_type="monthly")
    standard_deviation.attrs.update(units="degrees_Celsius", climatology_type="monthly")
    return xr.Dataset(
        {
            "climatological_mean": mean,
            "climatological_standard_deviation": standard_deviation,
        },
        attrs={"climatology_type": "synthetic demonstration climatology"},
    )
