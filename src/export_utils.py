"""Output helpers."""

from __future__ import annotations

from pathlib import Path
import json
from typing import Any

import xarray as xr


def export_netcdf(dataset: xr.Dataset, path: str | Path) -> Path:
    """Write a dataset to NetCDF, creating its parent directory."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output)
    return output


def export_json(payload: dict[str, Any], path: str | Path) -> Path:
    """Write a JSON report with stable, human-readable formatting."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as stream:
        json.dump(payload, stream, indent=2, ensure_ascii=False, allow_nan=False)
        stream.write("\n")
    return output
