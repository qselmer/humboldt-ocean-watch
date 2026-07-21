"""Offline integration tests for the Increment 4C tracking builder."""

from copy import deepcopy
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest
import xarray as xr
import yaml

import scripts.build_spatiotemporal_tracks as builder
from src.grid_geometry import build_grid_geometry, weighted_geographic_centroid
from src.patch_tracking import EDGE_COLUMNS, OBSERVATION_COLUMNS
from src.track_metrics import FAMILY_COLUMNS, TRACK_COLUMNS
from src.utils import load_config


def _write_inputs(tmp_path: Path, *, synthetic_climatology=False, mismatch=False):
    dates = pd.date_range("2024-01-01", periods=4)
    latitude = np.asarray([-3.5, -2.5, -1.5, -0.5])
    longitude = np.asarray([-89.5, -88.5, -87.5, -86.5])
    labels = np.zeros((4, 4, 4), dtype=np.int32)
    labels[0, 1, 1:3] = 1
    labels[1, 1, 2:4] = 1
    labels[3, 2, 2] = 1
    valid = np.ones_like(labels, dtype=np.uint8)
    dataset = xr.Dataset(
        {
            "patch_id": (("time", "latitude", "longitude"), labels),
            "threshold_mask": (("time", "latitude", "longitude"), (labels > 0).astype(np.uint8)),
            "valid_ocean_mask": (("time", "latitude", "longitude"), valid),
        },
        coords={"time": dates, "latitude": latitude, "longitude": longitude},
        attrs={
            "data_mode": "Copernicus cached data",
            "climatology_method": "synthetic" if synthetic_climatology else "daily_smoothed",
            "patch_id_scope": "local to each date; not persistent",
        },
    )
    dataset.to_netcdf(tmp_path / "labels.nc")
    geometry = build_grid_geometry(dataset.patch_id.isel(time=0, drop=True))
    areas = np.asarray(geometry.area_km2.values)
    rows = []
    for position in (0, 1, 3):
        cells = labels[position] == 1
        centroid_latitude, centroid_longitude = weighted_geographic_centroid(cells, geometry)
        rows.append({
            "date": dates[position], "patch_id": 1,
            "source_variable": "anomaly", "threshold_type": "fixed",
            "direction": "above", "area_km2": float(areas[cells].sum()),
            "centroid_latitude": centroid_latitude,
            "centroid_longitude": centroid_longitude,
            "mean_source_value": 3.0, "maximum_source_value": 4.0,
            "mean_exceedance": 1.0, "maximum_exceedance": 2.0,
            "valid_coverage": 1.0,
            "climatology_method": "synthetic" if synthetic_climatology else "daily_smoothed",
            "data_mode": "Copernicus cached data", "status": "valid", "reason": None,
            "touches_north_boundary": False, "touches_south_boundary": False,
            "touches_east_boundary": position == 1, "touches_west_boundary": False,
        })
    patches = pd.DataFrame(rows)
    if mismatch:
        patches.loc[0, "patch_id"] = 2
    patches.to_parquet(tmp_path / "patches.parquet", index=False)
    pd.DataFrame({
        "date": dates,
        "patch_count": [1, 1, 0, 1],
    }).to_parquet(tmp_path / "summary.parquet", index=False)


def _write_config(tmp_path: Path) -> Path:
    config = deepcopy(load_config())
    outputs = {
        "observations_output": tmp_path / "observations.parquet",
        "edges_output": tmp_path / "edges.parquet",
        "tracks_output": tmp_path / "tracks.parquet",
        "families_output": tmp_path / "families.parquet",
        "track_labels_output": tmp_path / "track_labels.nc",
        "track_mapping_output": tmp_path / "track_mapping.parquet",
        "family_mapping_output": tmp_path / "family_mapping.parquet",
        "json_summary_output": tmp_path / "tracking_summary.json",
    }
    config["patch_tracking"].update({name: str(path) for name, path in outputs.items()})
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _run(tmp_path: Path, **overrides):
    arguments = {
        "config_path": _write_config(tmp_path),
        "daily_patches": tmp_path / "patches.parquet",
        "daily_labels": tmp_path / "labels.nc",
        "daily_summary": tmp_path / "summary.parquet",
        "observations_output": None, "edges_output": None,
        "tracks_output": None, "families_output": None,
        "track_labels_output": None, "track_mapping_output": None,
        "family_mapping_output": None, "json_summary_output": None,
        "overwrite": False,
    }
    arguments.update(overrides)
    return builder.build(**arguments)


def test_builder_outputs_schemas_assignments_mappings_and_no_patch_dates(
    tmp_path: Path, monkeypatch
) -> None:
    _write_inputs(tmp_path)
    monkeypatch.setattr(
        "copernicusmarine.open_dataset",
        lambda *args, **kwargs: (_ for _ in ()).throw(AssertionError("network API called")),
    )
    artifacts = _run(tmp_path)
    assert artifacts.observations.columns.tolist() == OBSERVATION_COLUMNS
    assert artifacts.edges.columns.tolist() == EDGE_COLUMNS
    assert artifacts.tracks.columns.tolist() == TRACK_COLUMNS
    assert artifacts.families.columns.tolist() == FAMILY_COLUMNS
    assert artifacts.observations.shape[0] == 3
    assert artifacts.edges.shape[0] == 1
    assert artifacts.tracks.shape[0] == 2
    assert artifacts.families.shape[0] == 2
    assert artifacts.observations.track_id.notna().all()
    assert artifacts.observations.event_family_id.notna().all()
    assert dict(artifacts.track_labels.sizes) == {"time": 4, "latitude": 4, "longitude": 4}
    assert int(artifacts.track_labels.track_numeric_id.isel(time=2).max()) == 0
    assert set(artifacts.track_mapping.track_numeric_id) == {1, 2}
    assert set(artifacts.family_mapping.event_family_numeric_id) == {1, 2}
    first_track = artifacts.observations.loc[artifacts.observations.date < pd.Timestamp("2024-01-03"), "track_id"]
    last_track = artifacts.observations.loc[artifacts.observations.date == pd.Timestamp("2024-01-04"), "track_id"]
    assert first_track.nunique() == 1
    assert first_track.iloc[0] != last_track.iloc[0]
    persisted = pd.read_parquet(artifacts.observations_path)
    assert all(isinstance(value, (list, np.ndarray)) for value in persisted.parent_node_ids)
    assert all(isinstance(value, (list, np.ndarray)) for value in persisted.child_node_ids)
    for path in (
        artifacts.observations_path, artifacts.edges_path, artifacts.tracks_path,
        artifacts.families_path, artifacts.track_labels_path,
        artifacts.track_mapping_path, artifacts.family_mapping_path,
        artifacts.json_summary_path,
    ):
        assert path.exists()


def test_builder_summary_is_strict_json_and_reports_lineage(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    artifacts = _run(tmp_path)
    parsed = json.loads(artifacts.json_summary_path.read_text(encoding="utf-8"))
    assert parsed["number_of_dates"] == 4
    assert parsed["total_patch_observations"] == 3
    assert parsed["accepted_lineage_edges"] == 1
    assert parsed["total_tracks"] == 2
    assert parsed["total_event_families"] == 2
    assert parsed["official_enso_classification"] == "not provided"
    assert "NaN" not in artifacts.json_summary_path.read_text(encoding="utf-8")


def test_table_label_mismatch_and_real_synthetic_mix_are_rejected(tmp_path: Path) -> None:
    _write_inputs(tmp_path, mismatch=True)
    with pytest.raises(ValueError, match="do not match"):
        _run(tmp_path)
    other = tmp_path / "synthetic"
    other.mkdir()
    _write_inputs(other, synthetic_climatology=True)
    with pytest.raises(ValueError, match="synthetic"):
        _run(other)


def test_existing_outputs_survive_tracking_failure(tmp_path: Path, monkeypatch) -> None:
    _write_inputs(tmp_path)
    _write_config(tmp_path)
    destinations = [
        tmp_path / "observations.parquet", tmp_path / "edges.parquet",
        tmp_path / "tracks.parquet", tmp_path / "families.parquet",
        tmp_path / "track_labels.nc", tmp_path / "track_mapping.parquet",
        tmp_path / "family_mapping.parquet", tmp_path / "tracking_summary.json",
    ]
    for path in destinations:
        path.write_bytes(b"existing-valid-output")

    def fail(*args, **kwargs):
        raise RuntimeError("deliberate tracking failure")

    monkeypatch.setattr(builder, "build_lineage", fail)
    with pytest.raises(RuntimeError, match="deliberate"):
        _run(tmp_path, overwrite=True)
    for path in destinations:
        assert path.read_bytes() == b"existing-valid-output"


def test_builder_requires_overwrite_for_existing_products(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    _run(tmp_path)
    with pytest.raises(FileExistsError, match="--overwrite"):
        _run(tmp_path)


def test_all_zero_patch_dates_are_valid_and_preserved_in_label_cube(tmp_path: Path) -> None:
    _write_inputs(tmp_path)
    patches = pd.read_parquet(tmp_path / "patches.parquet")
    patches.iloc[0:0].to_parquet(tmp_path / "patches.parquet", index=False)
    with xr.open_dataset(tmp_path / "labels.nc") as source:
        labels = source.load()
    labels["patch_id"][:] = 0
    labels["threshold_mask"][:] = 0
    labels.to_netcdf(tmp_path / "labels.nc", mode="w")
    summary = pd.read_parquet(tmp_path / "summary.parquet")
    summary["patch_count"] = 0
    summary.to_parquet(tmp_path / "summary.parquet", index=False)
    artifacts = _run(tmp_path)
    assert artifacts.observations.empty
    assert artifacts.edges.empty
    assert artifacts.tracks.empty
    assert artifacts.families.empty
    assert artifacts.summary["overall_status"] == "not_calculated"
    assert artifacts.track_labels.sizes["time"] == 4
    assert int(artifacts.track_labels.track_numeric_id.max()) == 0


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        ("negative_area", "negative|positive area"),
        ("outside_centroid", "outside"),
        ("duplicate_local_id", "unique"),
    ],
)
def test_fatal_patch_input_inconsistencies_are_rejected(tmp_path: Path, mutation: str, message: str) -> None:
    _write_inputs(tmp_path)
    patches = pd.read_parquet(tmp_path / "patches.parquet")
    if mutation == "negative_area":
        patches.loc[0, "area_km2"] = -1.0
    elif mutation == "outside_centroid":
        patches.loc[0, "centroid_latitude"] = 25.0
    else:
        patches = pd.concat([patches, patches.iloc[[0]]], ignore_index=True).sort_values("date")
    patches.to_parquet(tmp_path / "patches.parquet", index=False)
    with pytest.raises(ValueError, match=message):
        _run(tmp_path)
