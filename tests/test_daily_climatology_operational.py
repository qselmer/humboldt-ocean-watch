"""Offline operational tests for checkpoints, schema, integration, and validation."""

from __future__ import annotations

import inspect
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import scripts.build_daily_climatology as builder
import scripts.validate_daily_climatology as validator
from src.daily_climatology import (
    DAILY_VARIABLES,
    calendar_coordinates,
    select_climatology,
    validate_daily_climatology,
)
from src.daily_diagnosis import diagnose_date
from src.data_loader import COPERNICUS_CACHED_MODE
from src.temporal_metrics import build_metrics_table


def valid_daily(std: float = 1.0) -> xr.Dataset:
    shape = (366, 1, 1)
    values = {
        "climatology_mean": np.full(shape, 20.0),
        "climatology_median": np.full(shape, 20.0),
        "climatology_std": np.full(shape, std),
        "threshold_p10": np.full(shape, 19.0),
        "threshold_p90": np.full(shape, 21.0),
        "observation_count": np.full(shape, 30, dtype=np.int64),
    }
    dataset = xr.Dataset(
        {name: (("climatological_day", "latitude", "longitude"), data) for name, data in values.items()},
        coords={**calendar_coordinates(), "latitude": [-5.0], "longitude": [-85.0]},
        attrs={
            "climatology_method": "daily_smoothed",
            "reference_period": "1991-01-01 to 2020-12-31",
            "sampling_half_window_days": 5,
            "smoothing_window_days": 31,
            "leap_day_method": "explicit bin 60",
        },
    )
    for name in values:
        dataset[name].attrs["units"] = "1" if name == "observation_count" else "degrees_Celsius"
    return dataset


def yearly_source(year: int = 2001) -> xr.Dataset:
    return xr.Dataset(
        {"analysed_sst": (("time", "latitude", "longitude"), np.array([[[293.15]], [[294.15]]]))},
        coords={"time": pd.to_datetime([f"{year}-01-01", f"{year}-12-31"]), "latitude": [-5.0], "longitude": [-85.0]},
    ).assign(analysed_sst=lambda data: data.analysed_sst.assign_attrs(units="kelvin"))


def write_checkpoint(path: Path, year: int = 2001) -> None:
    checkpoint = builder.prepare_year(yearly_source(year), year)
    checkpoint.to_netcdf(path)


def supported_fake(captured: dict[str, object], source: xr.Dataset):
    def fake(
        dataset_id=None, variables=None, minimum_longitude=None, maximum_longitude=None,
        minimum_latitude=None, maximum_latitude=None, start_datetime=None,
        end_datetime=None, chunk_size_limit=-1,
    ):
        captured.update(locals())
        return source
    return fake


def test_installed_copernicus_signature_is_compatible() -> None:
    import copernicusmarine

    signature = inspect.signature(copernicusmarine.open_dataset)
    assert "chunks" not in signature.parameters
    assert "chunk_size_limit" in signature.parameters
    assert builder.copernicus_open_kwargs(copernicusmarine.open_dataset) == {"chunk_size_limit": -1}


def test_preflight_rejects_incompatible_api() -> None:
    def incompatible(dataset_id: str):
        return dataset_id

    with pytest.raises(RuntimeError, match="incompatible"):
        builder.copernicus_open_kwargs(incompatible)


def test_required_schema_celsius_and_ordered_percentiles() -> None:
    dataset = valid_daily()
    assert set(DAILY_VARIABLES).issubset(dataset.data_vars)
    assert dataset.sizes["climatological_day"] == 366
    assert all(dataset[name].attrs["units"] == "degrees_Celsius" for name in DAILY_VARIABLES[:-1])
    assert bool((dataset.threshold_p10 <= dataset.threshold_p90).all())
    assert validate_daily_climatology(dataset) == []


def test_kelvin_is_converted_before_checkpoint_statistics() -> None:
    prepared = builder.prepare_year(yearly_source(), 2001)
    np.testing.assert_allclose(prepared.sst.values[:, 0, 0], [20.0, 21.0])
    assert prepared.sst.attrs["units"] == "degrees_Celsius"


def test_invalid_standard_deviation_is_rejected() -> None:
    errors = validate_daily_climatology(valid_daily(std=-1.0))
    assert "climatology_std contains negative values" in errors


def test_daily_diagnosis_matches_month_day_and_masks_invalid_std() -> None:
    dataset = xr.Dataset(
        {"sst": (("time", "latitude", "longitude"), np.array([[[22.0]], [[23.0]]]))},
        coords={"time": pd.to_datetime(["2023-03-01", "2024-02-29"]), "latitude": [-5.0], "longitude": [-85.0]},
        attrs={"data_mode": COPERNICUS_CACHED_MODE},
    )
    climatology = valid_daily(std=0.0)
    climatology.climatology_mean.loc[{"climatological_day": 61}] = 21.0
    fields, report = diagnose_date(dataset, "2023-03-01", climatology=climatology, data_mode=COPERNICUS_CACHED_MODE)
    assert float(fields.sst_anomaly.item()) == 1.0
    assert bool(fields.sst_z_score.isnull())
    assert bool(fields.p90_exceedance)
    assert report["metrics"]["area_above_climatological_p90_percent"] == 100.0
    assert report["metrics"]["percentile_warm_state"] == "widespread_above_p90"
    assert report["climatology_reference_period"] == "1991-01-01 to 2020-12-31"


def test_synthetic_climatology_is_rejected_for_real_sst(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="cannot be used with real"):
        select_climatology(
            tmp_path / "daily.nc", tmp_path / "monthly.nc",
            allow_monthly_fallback=False, data_mode=COPERNICUS_CACHED_MODE,
            synthetic_climatology=valid_daily(),
        )


def test_metrics_include_daily_climatology_metadata() -> None:
    dataset = xr.Dataset(
        {"sst": (("time", "latitude", "longitude"), np.array([[[22.0]], [[23.0]]]))},
        coords={"time": pd.to_datetime(["2024-01-01", "2024-01-02"]), "latitude": [-5.0], "longitude": [-85.0]},
    )
    table = build_metrics_table(dataset, valid_daily())
    assert table.climatology_method.eq("daily_smoothed").all()
    assert table.climatology_reference_period.eq("1991-01-01 to 2020-12-31").all()
    assert table.area_above_climatological_p90_percent.eq(100.0).all()


def test_invalid_checkpoint_is_replaced(tmp_path: Path, monkeypatch) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    (checkpoints / "ostia_nino12_2001.nc").write_text("broken", encoding="utf-8")
    captured: dict[str, object] = {}
    monkeypatch.setattr("copernicusmarine.open_dataset", supported_fake(captured, yearly_source()))
    monkeypatch.setattr(builder, "build_daily_statistics", lambda *args, **kwargs: valid_daily())
    output = tmp_path / "daily.nc"
    builder.build_daily_climatology(2001, 2001, output, checkpoint_dir=checkpoints, resume=True)
    assert captured["variables"] == ["analysed_sst"]
    assert builder.validate_year_checkpoint(checkpoints / "ostia_nino12_2001.nc", 2001)[0]


def test_valid_checkpoint_is_reused(tmp_path: Path, monkeypatch) -> None:
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    write_checkpoint(checkpoints / "ostia_nino12_2001.nc")
    monkeypatch.setattr("copernicusmarine.open_dataset", supported_fake({}, yearly_source()))
    monkeypatch.setattr(builder, "build_daily_statistics", lambda *args, **kwargs: valid_daily())
    output = tmp_path / "daily.nc"
    builder.build_daily_climatology(2001, 2001, output, checkpoint_dir=checkpoints, resume=True)
    assert output.exists()


def test_existing_output_survives_processing_failure(tmp_path: Path, monkeypatch) -> None:
    output = tmp_path / "daily.nc"
    original = b"existing-valid-output-placeholder"
    output.write_bytes(original)
    checkpoints = tmp_path / "checkpoints"
    checkpoints.mkdir()
    write_checkpoint(checkpoints / "ostia_nino12_2001.nc")
    monkeypatch.setattr("copernicusmarine.open_dataset", supported_fake({}, yearly_source()))
    monkeypatch.setattr(builder, "build_daily_statistics", lambda *args, **kwargs: (_ for _ in ()).throw(ValueError("failure")))
    with pytest.raises(RuntimeError, match="not_calculated"):
        builder.build_daily_climatology(
            2001, 2001, output, checkpoint_dir=checkpoints, resume=True, overwrite=True
        )
    assert output.read_bytes() == original


def test_validation_script_returns_success_and_failure(tmp_path: Path, monkeypatch) -> None:
    monkeypatch.setattr(validator, "REPORT_PATH", tmp_path / "report.json")
    monkeypatch.setattr(validator, "FIGURE_PATH", tmp_path / "continuity.png")
    valid_path = tmp_path / "valid.nc"
    valid_daily().to_netcdf(valid_path)
    assert validator.main(["--input", str(valid_path)]) == 0
    assert (tmp_path / "report.json").exists()
    assert (tmp_path / "continuity.png").exists()
    assert validator.main(["--input", str(tmp_path / "missing.nc")]) == 1
