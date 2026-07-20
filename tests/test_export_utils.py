"""Tests for strict reusable JSON export conversion."""

from __future__ import annotations

import json
from datetime import date, datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.export_utils import dumps_json_safe, export_json, to_json_compatible


def test_numpy_scalar_types_are_native_json_types() -> None:
    converted = to_json_compatible(
        {
            "integer": np.int64(7),
            "float32": np.float32(1.25),
            "float64": np.float64(2.5),
            "boolean": np.bool_(True),
        }
    )
    assert converted == {
        "integer": 7,
        "float32": 1.25,
        "float64": 2.5,
        "boolean": True,
    }
    assert type(converted["integer"]) is int
    assert type(converted["float32"]) is float
    assert type(converted["boolean"]) is bool


def test_arrays_nested_containers_and_string_dictionary_keys() -> None:
    converted = to_json_compatible(
        {
            np.int64(5): np.array([[1, 2], [3, 4]], dtype=np.int64),
            "nested": (np.float32(3.5), {np.int64(8), np.int64(9)}),
        }
    )
    assert converted["5"] == [[1, 2], [3, 4]]
    assert converted["nested"][0] == 3.5
    assert sorted(converted["nested"][1]) == [8, 9]
    assert to_json_compatible(
        np.array(["2026-07-20T12:30:00"], dtype="datetime64[ns]")
    ) == ["2026-07-20T12:30:00"]


def test_dates_paths_and_missing_values_are_converted() -> None:
    converted = to_json_compatible(
        {
            "numpy_date": np.datetime64("2026-07-20T12:30:00"),
            "timestamp": pd.Timestamp("2026-07-20T12:30:00-05:00"),
            "datetime": datetime(2026, 7, 20, 12, 30, tzinfo=timezone.utc),
            "date": date(2026, 7, 20),
            "path": Path("outputs/reports/example.json"),
            "pandas_na": pd.NA,
        }
    )
    assert converted["numpy_date"] == "2026-07-20T12:30:00"
    assert converted["timestamp"] == "2026-07-20T12:30:00-05:00"
    assert converted["datetime"] == "2026-07-20T12:30:00+00:00"
    assert converted["date"] == "2026-07-20"
    assert converted["path"] == str(Path("outputs/reports/example.json"))
    assert converted["pandas_na"] is None


def test_nonfinite_numbers_become_null_and_never_invalid_json_tokens() -> None:
    text = dumps_json_safe(
        {
            "nan": float("nan"),
            "positive": float("inf"),
            "negative": float("-inf"),
            "numpy_nan": np.float64(np.nan),
        }
    )
    assert json.loads(text) == {
        "nan": None,
        "positive": None,
        "negative": None,
        "numpy_nan": None,
    }
    assert "NaN" not in text
    assert "Infinity" not in text


def test_scalar_and_non_scalar_xarray_dataarrays() -> None:
    scalar = xr.DataArray(np.int64(12))
    field = xr.DataArray(np.array([[1.0, np.nan], [np.inf, 4.0]]))
    assert to_json_compatible(scalar) == 12
    assert to_json_compatible(field) == [[1.0, None], [None, 4.0]]


def test_representative_diagnosis_with_numpy_metrics_is_valid_json() -> None:
    diagnosis = {
        "date": pd.Timestamp("2026-07-19"),
        "experimental_product": np.bool_(True),
        "persistence_window_days": np.int64(7),
        "metrics": {
            "mean_sst_c": np.float32(24.75),
            "observation_count": np.int64(38120),
            "unavailable": np.float64(np.nan),
            "centroid": [np.float64(-84.5), np.float64(-4.25)],
        },
    }
    text = dumps_json_safe(diagnosis)
    parsed = json.loads(text)
    assert parsed["date"] == "2026-07-19T00:00:00"
    assert parsed["experimental_product"] is True
    assert parsed["persistence_window_days"] == 7
    assert parsed["metrics"]["mean_sst_c"] == 24.75
    assert parsed["metrics"]["unavailable"] is None


def test_file_export_uses_the_same_safe_pathway(tmp_path: Path) -> None:
    output = export_json(
        {"count": np.int64(2), "invalid": np.float64(np.inf)},
        tmp_path / "report.json",
    )
    assert json.loads(output.read_text(encoding="utf-8")) == {
        "count": 2,
        "invalid": None,
    }
