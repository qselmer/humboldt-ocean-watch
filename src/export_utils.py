"""Output helpers."""

from __future__ import annotations

import json
import math
from datetime import date, datetime
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr


def to_json_compatible(value: Any) -> Any:
    """Recursively convert scientific Python values into strict JSON values.

    Numeric values remain numbers, dates become ISO-8601 strings, and missing
    or non-finite values become ``None``. Dictionary keys are always strings.
    """
    if value is None or value is pd.NA or value is pd.NaT:
        return None
    if isinstance(value, xr.DataArray):
        if value.ndim == 0:
            return to_json_compatible(value.item())
        return to_json_compatible(value.values)
    if isinstance(value, dict):
        return {str(key): to_json_compatible(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [to_json_compatible(item) for item in value]
    if isinstance(value, np.ndarray):
        if value.ndim == 0:
            return to_json_compatible(value[()])
        return [to_json_compatible(item) for item in value]
    if isinstance(value, np.datetime64):
        return None if np.isnat(value) else pd.Timestamp(value).isoformat()
    if isinstance(value, pd.Timestamp):
        return None if pd.isna(value) else value.isoformat()
    if isinstance(value, datetime):
        return value.isoformat()
    if isinstance(value, date):
        return value.isoformat()
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, np.bool_):
        return bool(value)
    if isinstance(value, np.integer):
        return int(value)
    if isinstance(value, np.floating):
        number = float(value)
        return number if math.isfinite(number) else None
    if isinstance(value, float):
        return value if math.isfinite(value) else None
    if isinstance(value, (str, int, bool)):
        return value
    if isinstance(value, np.generic):
        return to_json_compatible(value.item())
    raise TypeError(f"Unsupported value for JSON serialization: {type(value).__name__}")


def dumps_json_safe(
    value: Any,
    *,
    indent: int | None = 2,
    ensure_ascii: bool = False,
) -> str:
    """Serialize a value after strict recursive scientific-type conversion."""
    return json.dumps(
        to_json_compatible(value),
        indent=indent,
        ensure_ascii=ensure_ascii,
        allow_nan=False,
    )


def export_netcdf(dataset: xr.Dataset, path: str | Path) -> Path:
    """Write a dataset to NetCDF, creating its parent directory."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset.to_netcdf(output)
    return output


def export_json(payload: Any, path: str | Path) -> Path:
    """Write a JSON report with stable, human-readable formatting."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(dumps_json_safe(payload) + "\n", encoding="utf-8")
    return output
