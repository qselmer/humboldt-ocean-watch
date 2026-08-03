from dataclasses import replace
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

from src.climatology_checkpoints import (
    CheckpointContract, atomic_write_checkpoint, checkpoint_metadata,
    checksum_path, validate_checkpoint,
)
from src.climatology_build_plan import build_climatology_plan
from src.real_climatology_builder import prepare_daily_checkpoint
from src.utils import load_config


def _daily_dataset(year=1991, step=0.05, size=5):
    data = xr.Dataset(
        {"sst": (("time", "latitude", "longitude"), np.full((2, size, size), 20.0))},
        coords={
            "time": pd.to_datetime([f"{year}-01-01", f"{year}-01-02"]),
            "latitude": np.arange(size) * step - 5.0,
            "longitude": np.arange(size) * step - 90.0,
        },
    )
    data.sst.attrs["units"] = "degrees_Celsius"
    return data


def _contract(year=1991):
    return CheckpointContract(
        domain_id="humboldt_coastal", tile_id="r000_c000",
        processing_stage="daily_target_grid",
        source_dataset="METOFFICE-GLO-SST-L4-REP-OBS-SST",
        source_variable="analysed_sst", target_resolution=0.05,
        checkpoint_year=year, source_version="test-v1",
    )


def test_atomic_checkpoint_metadata_checksum_and_resume_validation(tmp_path: Path) -> None:
    dataset = _daily_dataset()
    dataset.attrs["partial_edge_blocks"] = True
    contract = _contract()
    dataset.attrs.update(checkpoint_metadata(
        contract, requested_bounds=(-90, -89.8, -5, -4.8),
        effective_bounds=(-90, -89.8, -5, -4.8), source_resolution=0.05,
        valid_coverage=1.0, time_range=("1991-01-01", "1991-12-31"),
        aggregation_applied=False, created_at="2026-01-01T00:00:00Z",
        software_revision="test",
    ))
    path = tmp_path / "checkpoint.nc"
    written = atomic_write_checkpoint(dataset, path, contract)
    assert written.valid
    assert checksum_path(path).exists()
    assert validate_checkpoint(path, contract).valid
    with xr.open_dataset(path) as opened:
        assert opened.attrs["partial_edge_blocks"] == "true"
    assert not list(tmp_path.glob(".checkpoint-*.nc"))


def test_checkpoint_rejects_corruption_year_tile_units_and_version(tmp_path: Path) -> None:
    dataset = _daily_dataset()
    contract = _contract()
    dataset.attrs.update(checkpoint_metadata(
        contract, requested_bounds=(-90, -89.8, -5, -4.8), effective_bounds=(-90, -89.8, -5, -4.8),
        source_resolution=0.05, valid_coverage=1.0, time_range=("1991-01-01", "1991-12-31"),
        aggregation_applied=False, created_at="now", software_revision="test",
    ))
    path = tmp_path / "checkpoint.nc"
    atomic_write_checkpoint(dataset, path, contract)
    assert not validate_checkpoint(path, replace(contract, tile_id="other")).valid
    assert not validate_checkpoint(path, replace(contract, checkpoint_year=1992)).valid
    assert not validate_checkpoint(path, replace(contract, source_version="test-v2")).valid
    path.write_bytes(b"corrupt")
    assert validate_checkpoint(path, contract).reason == "checksum mismatch"


def test_prepare_daily_checkpoint_aggregates_pacific_and_preserves_humboldt() -> None:
    config = load_config()
    plans = {item.domain_id: item for item in build_climatology_plan(config, ("pacific_context", "humboldt_coastal"), pilot=True).domains}
    source = _daily_dataset(size=10).rename({"sst": "analysed_sst"})
    source.analysed_sst.attrs["units"] = "degrees_Celsius"
    pacific, resolution, aggregated = prepare_daily_checkpoint(source, plans["pacific_context"], config)
    assert np.isclose(resolution, 0.05) and aggregated is True
    assert pacific.sizes["latitude"] == pacific.sizes["longitude"] == 2
    coastal, _, aggregated = prepare_daily_checkpoint(source, plans["humboldt_coastal"], config)
    assert aggregated is False
    assert coastal.sizes["latitude"] == coastal.sizes["longitude"] == 10
