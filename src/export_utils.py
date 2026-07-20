"""Output helpers."""

from __future__ import annotations

from pathlib import Path

import xarray as xr


def export_netcdf(dataset: xr.Dataset, path: str | Path) -> Path:
    """Write a dataset to NetCDF, creating its parent directory."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output)
    return output
