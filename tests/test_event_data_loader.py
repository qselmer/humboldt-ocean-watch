from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.event_data_loader import load_json_product, load_netcdf_date, load_table_product


def test_valid_empty_and_missing_table_products(tmp_path: Path) -> None:
    valid_path = tmp_path / "valid.parquet"
    pd.DataFrame({"date": ["2026-01-01"], "value": [1.0]}).to_parquet(valid_path)
    valid = load_table_product(valid_path, required_columns={"date", "value"}, date_columns=("date",))
    assert valid.available and valid.status == "available"
    assert pd.api.types.is_datetime64_any_dtype(valid.value.date)

    empty_path = tmp_path / "empty.parquet"
    pd.DataFrame(columns=["date", "value"]).to_parquet(empty_path)
    empty = load_table_product(empty_path, required_columns={"date", "value"}, date_columns=("date",))
    assert empty.available and empty.empty

    missing = load_table_product(tmp_path / "missing.parquet", required_columns={"value"})
    assert not missing.available and missing.status == "unavailable"


def test_missing_required_columns_are_structurally_invalid(tmp_path: Path) -> None:
    path = tmp_path / "wrong.parquet"
    pd.DataFrame({"other": [1]}).to_parquet(path)
    state = load_table_product(path, required_columns={"value"})
    assert state.status == "invalid"
    assert "value" in state.reason


def test_json_product_is_failure_tolerant(tmp_path: Path) -> None:
    path = tmp_path / "summary.json"
    path.write_text('{"overall_status": "warning"}', encoding="utf-8")
    state = load_json_product(path)
    assert state.available
    assert state.value == {"overall_status": "warning"}
    invalid = tmp_path / "invalid.json"
    invalid.write_text("[1, 2]", encoding="utf-8")
    assert load_json_product(invalid).status == "invalid"


def test_selected_netcdf_date_is_eager_and_source_handle_is_closed(tmp_path: Path) -> None:
    path = tmp_path / "labels.nc"
    dataset = xr.Dataset(
        {
            "patch_id": (("time", "latitude", "longitude"), np.arange(8).reshape(2, 2, 2)),
            "valid_ocean_mask": (("time", "latitude", "longitude"), np.ones((2, 2, 2), dtype=np.uint8)),
        },
        coords={"time": pd.date_range("2026-01-01", periods=2), "latitude": [-1.0, 0.0], "longitude": [-90.0, -89.0]},
    )
    dataset.to_netcdf(path)
    state = load_netcdf_date(path, "2026-01-02", required_variables=("patch_id", "valid_ocean_mask"))
    assert state.available
    assert state.value.patch_id.dims == ("latitude", "longitude")
    assert state.value.patch_id.values.tolist() == [[4, 5], [6, 7]]
    renamed = tmp_path / "renamed.nc"
    path.rename(renamed)  # Windows permits this only when the source handle is closed.
    assert renamed.exists()


def test_unavailable_netcdf_date_and_missing_variable_are_isolated(tmp_path: Path) -> None:
    path = tmp_path / "labels.nc"
    xr.Dataset(
        {"patch_id": (("time", "latitude", "longitude"), np.ones((1, 1, 1), dtype=int))},
        coords={"time": [np.datetime64("2026-01-01")], "latitude": [0.0], "longitude": [-90.0]},
    ).to_netcdf(path)
    unavailable = load_netcdf_date(path, "2026-01-02", required_variables=("patch_id",))
    assert unavailable.status == "unavailable"
    invalid = load_netcdf_date(path, "2026-01-01", required_variables=("track_numeric_id",))
    assert invalid.status == "invalid"

