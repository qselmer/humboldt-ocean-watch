"""Offline integration tests for the cached daily patch builder."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

import scripts.build_daily_patches as builder
from src.daily_climatology import calendar_coordinates
from src.patch_metrics import DAILY_PATCH_SUMMARY_COLUMNS, PATCH_COLUMNS
from src.utils import load_config


def _write_sst(path: Path) -> None:
    values = np.full((3, 4, 4), 20.0)
    values[1, 1:3, 1:3] = 23.0
    values[2, 0, 0] = 23.0
    values[2, 3, 3] = 24.0
    dataset = xr.Dataset(
        {
            "analysed_sst": (
                ("time", "latitude", "longitude"),
                values + 273.15,
                {"units": "kelvin"},
            )
        },
        coords={
            "time": pd.date_range("2024-01-01", periods=3),
            "latitude": [-3.5, -2.5, -1.5, -0.5],
            "longitude": [-89.5, -88.5, -87.5, -86.5],
        },
    )
    dataset.to_netcdf(path)


def _write_climatology(path: Path, *, synthetic: bool = False) -> None:
    shape = (366, 4, 4)
    fields = {
        "climatology_mean": np.full(shape, 20.0),
        "climatology_median": np.full(shape, 20.0),
        "climatology_std": np.ones(shape),
        "threshold_p10": np.full(shape, 19.0),
        "threshold_p90": np.full(shape, 21.0),
        "observation_count": np.full(shape, 30, dtype=np.int64),
    }
    dataset = xr.Dataset(
        {
            name: (("climatological_day", "latitude", "longitude"), values)
            for name, values in fields.items()
        },
        coords={
            **calendar_coordinates(),
            "latitude": [-3.5, -2.5, -1.5, -0.5],
            "longitude": [-89.5, -88.5, -87.5, -86.5],
        },
        attrs={
            "climatology_method": "synthetic" if synthetic else "daily_smoothed",
            "reference_period": "1991-01-01 to 2020-12-31",
            "source_dataset": "synthetic demonstration" if synthetic else "METOFFICE-GLO-SST-L4-REP-OBS-SST",
        },
    )
    for name in fields:
        dataset[name].attrs["units"] = "1" if name == "observation_count" else "degrees_Celsius"
    dataset.to_netcdf(path)


def _write_config(tmp_path: Path) -> Path:
    config = deepcopy(load_config())
    config["patch_detection"].update(
        {
            "minimum_patch_cells": 1,
            "patches_output": str(tmp_path / "patches.parquet"),
            "daily_summary_output": str(tmp_path / "daily.parquet"),
            "labels_output": str(tmp_path / "labels.nc"),
            "json_summary_output": str(tmp_path / "summary.json"),
        }
    )
    config["climatology"]["monthly_path"] = str(tmp_path / "missing-monthly.nc")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _run(tmp_path: Path, **overrides):
    arguments = {
        "config_path": _write_config(tmp_path),
        "sst_input": tmp_path / "sst.nc",
        "climatology_input": tmp_path / "climatology.nc",
        "source_variable": None,
        "threshold_type": None,
        "fixed_threshold": None,
        "direction": None,
        "connectivity": None,
        "minimum_patch_cells": None,
        "minimum_patch_area_km2": None,
        "patches_output": None,
        "daily_summary_output": None,
        "labels_output": None,
        "json_summary_output": None,
        "overwrite": False,
    }
    arguments.update(overrides)
    return builder.build(**arguments)


def test_builder_outputs_schemas_all_dates_and_local_labels(tmp_path: Path, monkeypatch) -> None:
    _write_sst(tmp_path / "sst.nc")
    _write_climatology(tmp_path / "climatology.nc")
    monkeypatch.setattr(
        "copernicusmarine.open_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network API called")),
    )
    artifacts = _run(tmp_path)
    assert artifacts.patches.columns.tolist() == PATCH_COLUMNS
    assert artifacts.daily_summary.columns.tolist() == DAILY_PATCH_SUMMARY_COLUMNS
    assert artifacts.daily_summary.shape == (3, len(DAILY_PATCH_SUMMARY_COLUMNS))
    assert artifacts.patches.shape[0] == 3
    assert artifacts.patches_path.exists()
    assert artifacts.daily_summary_path.exists()
    assert artifacts.labels_path and artifacts.labels_path.exists()
    assert artifacts.json_summary_path.exists()
    assert dict(artifacts.labels.sizes) == {"time": 3, "latitude": 4, "longitude": 4}
    assert set(artifacts.labels.data_vars) == {"patch_id", "threshold_mask", "valid_ocean_mask"}
    assert artifacts.labels.patch_id.attrs["patch_id_scope"].startswith("local to each date")
    assert int(artifacts.labels.patch_id.isel(time=1).max()) == 1
    assert set(np.unique(artifacts.labels.patch_id.isel(time=2))) == {0, 1, 2}


def test_builder_json_is_safe_and_complete(tmp_path: Path) -> None:
    _write_sst(tmp_path / "sst.nc")
    _write_climatology(tmp_path / "climatology.nc")
    artifacts = _run(tmp_path)
    parsed = json.loads(artifacts.json_summary_path.read_text(encoding="utf-8"))
    assert parsed["number_of_dates"] == 3
    assert parsed["number_of_dates_with_patches"] == 2
    assert parsed["total_patches_detected"] == 3
    assert parsed["maximum_daily_patch_count"] == 2
    assert parsed["connectivity"] == "queen"
    assert parsed["official_enso_classification"] == "not provided"
    assert parsed["spatiotemporal_tracking"] == "not performed"
    assert parsed["largest_patch_observed"]["area_km2"] > 0
    assert "approximate_output_sizes_bytes" in parsed


def test_daily_climatological_p90_definition_uses_sst_field(tmp_path: Path) -> None:
    _write_sst(tmp_path / "sst.nc")
    _write_climatology(tmp_path / "climatology.nc")
    artifacts = _run(
        tmp_path,
        source_variable="sst",
        threshold_type="daily_climatological",
    )
    assert set(artifacts.patches.source_variable) == {"sst"}
    assert set(artifacts.patches.threshold_type) == {"daily_climatological"}
    assert "threshold_p90" in artifacts.summary["threshold_definition"]


def test_standardized_patch_definition_uses_zscore(tmp_path: Path) -> None:
    _write_sst(tmp_path / "sst.nc")
    _write_climatology(tmp_path / "climatology.nc")
    artifacts = _run(
        tmp_path,
        source_variable="zscore",
        threshold_type="standardized",
    )
    assert set(artifacts.patches.source_variable) == {"zscore"}
    assert set(artifacts.patches.threshold_type) == {"standardized"}
    assert set(artifacts.patches.threshold_value) == {2.0}


def test_real_sst_rejects_synthetic_climatology(tmp_path: Path) -> None:
    _write_sst(tmp_path / "sst.nc")
    _write_climatology(tmp_path / "climatology.nc", synthetic=True)
    with pytest.raises(ValueError, match="synthetic|daily_smoothed"):
        _run(tmp_path)


def test_existing_outputs_survive_processing_failure(tmp_path: Path, monkeypatch) -> None:
    _write_sst(tmp_path / "sst.nc")
    _write_climatology(tmp_path / "climatology.nc")
    destinations = [
        tmp_path / "patches.parquet",
        tmp_path / "daily.parquet",
        tmp_path / "labels.nc",
        tmp_path / "summary.json",
    ]
    for path in destinations:
        path.write_bytes(b"existing-output")

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate date-processing failure")

    monkeypatch.setattr(builder, "_match_date_fields", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        _run(tmp_path, overwrite=True)
    for path in destinations:
        assert path.read_bytes() == b"existing-output"


def test_builder_requires_overwrite_for_existing_outputs(tmp_path: Path) -> None:
    _write_sst(tmp_path / "sst.nc")
    _write_climatology(tmp_path / "climatology.nc")
    _run(tmp_path)
    with pytest.raises(FileExistsError, match="--overwrite"):
        _run(tmp_path)
