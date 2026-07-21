"""Pure UI preparation helpers for the cached thermal-event dashboard."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd

from src.export_utils import dumps_json_safe


def finite_domain(
    values: Iterable[Any], *, padding_fraction: float = 0.08, minimum_padding: float = 0.10
) -> tuple[float, float] | None:
    """Return finite dynamic bounds with robust constant-series padding."""
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").to_numpy(dtype=float)
    finite = numeric[np.isfinite(numeric)]
    if not finite.size:
        return None
    lower, upper = float(finite.min()), float(finite.max())
    padding = max((upper - lower) * padding_fraction, minimum_padding)
    return lower - padding, upper + padding


def add_temporal_segments(
    data: pd.DataFrame,
    *,
    date_column: str = "date",
    value_columns: tuple[str, ...] = (),
    maximum_gap_days: int = 1,
) -> pd.DataFrame:
    """Add a segment identifier so charts never bridge missing dates or values."""
    if date_column not in data.columns:
        raise ValueError(f"Missing date column {date_column!r}")
    frame = data.copy()
    frame[date_column] = pd.to_datetime(frame[date_column], errors="coerce")
    for column in value_columns:
        if column in frame.columns:
            frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame = frame[frame[date_column].notna()].sort_values(date_column, kind="mergesort").reset_index(drop=True)
    date_break = frame[date_column].diff().dt.total_seconds().div(86400).gt(maximum_gap_days)
    missing = frame[list(value_columns)].isna().any(axis=1) if value_columns else pd.Series(False, index=frame.index)
    previous_missing = missing.shift(fill_value=False)
    frame["segment_id"] = (date_break | missing | previous_missing).cumsum().astype(int)
    return frame


def normalize_identifier(value: Any, options: Iterable[Any]) -> str | None:
    """Return a valid string ID, never silently preserving a stale selection."""
    available = {str(option) for option in options if pd.notna(option)}
    if value is None or str(value) not in available:
        return None
    return str(value)


def format_scalar(value: Any, unit: str = "", *, precision: int = 2) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number) or not np.isfinite(float(number)):
        return "Unavailable"
    suffix = f" {unit}" if unit else ""
    return f"{float(number):.{precision}f}{suffix}"


def format_fraction(value: Any) -> str:
    number = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    if pd.isna(number) or not np.isfinite(float(number)):
        return "Unavailable"
    return f"{100.0 * float(number):.1f}%"


def dataframe_to_iso_csv(data: pd.DataFrame) -> str:
    """Serialize a table to CSV with ISO dates and blank missing values."""
    frame = data.copy()
    for column in frame.columns:
        if pd.api.types.is_datetime64_any_dtype(frame[column]):
            frame[column] = frame[column].dt.strftime("%Y-%m-%d")
        elif column == "date" or column.endswith("_date"):
            converted = pd.to_datetime(frame[column], errors="coerce")
            if converted.notna().any():
                frame[column] = converted.dt.strftime("%Y-%m-%d")
    return frame.to_csv(index=False, na_rep="")


def record_to_json(data: pd.DataFrame | pd.Series | dict[str, Any] | None) -> str:
    """Build a strict JSON download from an empty or selected record set."""
    if data is None:
        payload: Any = None
    elif isinstance(data, pd.DataFrame):
        payload = data.to_dict(orient="records")
    elif isinstance(data, pd.Series):
        payload = data.to_dict()
    else:
        payload = data
    return dumps_json_safe(payload)

