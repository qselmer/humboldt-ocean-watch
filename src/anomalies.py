"""Thermal anomaly calculations."""

from __future__ import annotations

import xarray as xr


def calculate_anomaly(observed: xr.DataArray, reference: xr.DataArray) -> xr.DataArray:
    """Calculate observed-minus-reference SST anomaly in degrees Celsius."""
    if observed.size == 0 or reference.size == 0:
        raise ValueError("Observed and reference SST arrays must not be empty")
    anomaly = observed - reference
    anomaly.name = "sst_anomaly"
    anomaly.attrs.update(units="degrees_Celsius", long_name="sea surface temperature anomaly")
    return anomaly


def calculate_z_score(
    anomaly: xr.DataArray, climatological_std: xr.DataArray
) -> xr.DataArray:
    """Standardize an anomaly, returning NaN where variability is zero."""
    if anomaly.size == 0 or climatological_std.size == 0:
        raise ValueError("Anomaly and climatological standard deviation must not be empty")
    z_score = xr.where(climatological_std != 0, anomaly / climatological_std, float("nan"))
    z_score.name = "sst_z_score"
    z_score.attrs.update(units="1", long_name="standardized sea surface temperature anomaly")
    return z_score


def anomalies_from_time_mean(sst: xr.DataArray) -> xr.DataArray:
    """Calculate anomalies relative to the supplied period mean."""
    if "time" not in sst.dims:
        raise ValueError("SST must include a 'time' dimension")
    return calculate_anomaly(sst, sst.mean("time", skipna=True))
