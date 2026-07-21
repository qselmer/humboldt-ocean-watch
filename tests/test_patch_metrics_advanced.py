"""Advanced kilometre-area patch metrics and daily summary tests."""

import numpy as np
import pytest
import xarray as xr

from src.patch_detection import PatchDetectionConfig, identify_patches
from src.patch_metrics import (
    DAILY_PATCH_SUMMARY_COLUMNS,
    PATCH_COLUMNS,
    characterize_daily_patches,
    compactness_result,
)


def _field(values: np.ndarray) -> xr.DataArray:
    return xr.DataArray(
        np.asarray(values, dtype=float),
        dims=("latitude", "longitude"),
        coords={
            "latitude": -3.5 + np.arange(values.shape[0]),
            "longitude": -89.5 + np.arange(values.shape[1]),
        },
        attrs={"units": "degrees_Celsius"},
    )


def _config(**changes) -> PatchDetectionConfig:
    values = dict(
        source_variable="anomaly",
        direction="above",
        threshold_type="fixed",
        comparison="inclusive",
        connectivity="rook",
        minimum_patch_cells=1,
        minimum_patch_area_km2=0.0,
    )
    values.update(changes)
    return PatchDetectionConfig(**values)


def _characterize(values: np.ndarray, *, config=None):
    config = config or _config()
    field = _field(values)
    detection = identify_patches(field, 2.0, config=config)
    patches, summary = characterize_daily_patches(
        detection,
        date="2024-01-01",
        config=config,
        climatology_method="daily_smoothed",
        data_mode="Copernicus cached data",
        ocean_domain_mask=xr.ones_like(field, dtype=bool),
    )
    return detection, patches, summary


def test_patch_size_location_intensity_and_schema() -> None:
    values = np.zeros((4, 4))
    values[1:3, 1:3] = [[3.0, 4.0], [5.0, 6.0]]
    detection, patches, summary = _characterize(values)
    assert patches.columns.tolist() == PATCH_COLUMNS
    assert list(summary) == DAILY_PATCH_SUMMARY_COLUMNS
    row = patches.iloc[0]
    cells = detection.patch_id.values == 1
    expected_area = float(detection.validated.cell_area_km2.values[cells].sum())
    assert row.cell_count == 4
    assert row.area_km2 == pytest.approx(expected_area)
    assert row.centroid_latitude == pytest.approx(-2.0, abs=0.01)
    assert row.centroid_longitude == pytest.approx(-88.0, abs=0.01)
    assert row.minimum_latitude == -2.5
    assert row.maximum_latitude == -1.5
    assert row.minimum_longitude == -88.5
    assert row.maximum_longitude == -87.5
    assert row.mean_source_value == pytest.approx(4.5, rel=1e-3)
    assert row.maximum_source_value == 6.0
    assert row.mean_exceedance == pytest.approx(2.5, rel=1e-3)
    assert row.maximum_exceedance == 4.0
    expected_cumulative = float(
        np.sum((values[cells] - 2.0) * detection.validated.cell_area_km2.values[cells])
    )
    assert row.cumulative_exceedance_km2_units == pytest.approx(expected_cumulative)


def test_perimeter_compactness_dispersion_orientation_and_elongation() -> None:
    values = np.zeros((5, 6))
    values[1:3, 1:5] = 3.0
    detection, patches, _ = _characterize(values)
    row = patches.iloc[0]
    expected_compactness = 4.0 * np.pi * row.area_km2 / row.perimeter_km**2
    assert row.perimeter_km > 0
    assert row.compactness == pytest.approx(expected_compactness)
    assert row.perimeter_area_ratio == pytest.approx(row.perimeter_km / row.area_km2)
    assert row.centroid_dispersion_km > 0
    assert abs(row.orientation_degrees) < 5.0
    assert row.elongation > 1.0
    assert row.major_axis_length_km > row.minor_axis_length_km > 0
    assert 0 < row.edge_cell_fraction <= 1


def test_boundary_touching_flags() -> None:
    values = np.zeros((4, 4))
    values[0, 0] = values[0, 1] = 3.0
    _, patches, _ = _characterize(values)
    row = patches.iloc[0]
    assert row.touches_south_boundary
    assert row.touches_west_boundary
    assert not row.touches_north_boundary
    assert not row.touches_east_boundary


def test_zero_perimeter_safeguard_uses_metric_result() -> None:
    result = compactness_result(10.0, 0.0, cell_count=1, valid_coverage=1.0)
    assert result.status == "not_calculated"
    assert np.isnan(result.value)
    assert "perimeter" in (result.reason or "").lower()


def test_zero_patch_daily_summary_is_explicit() -> None:
    _, patches, summary = _characterize(np.zeros((3, 3)))
    assert patches.empty
    assert summary["patch_count"] == 0
    assert summary["total_patch_area_km2"] == 0
    assert summary["threshold_area_fraction"] == 0
    assert np.isnan(summary["area_weighted_centroid_latitude"])
    assert summary["status"] == "not_calculated"


def test_multiple_patch_summary_fragmentation_density_and_centroid() -> None:
    values = np.zeros((5, 5))
    values[0:2, 0:2] = 3.0
    values[4, 4] = 4.0
    _, patches, summary = _characterize(values)
    assert len(patches) == 2
    areas = patches.area_km2.to_numpy()
    expected_dominant = areas.max() / areas.sum()
    assert summary["patch_count"] == 2
    assert summary["dominant_patch_fraction"] == pytest.approx(expected_dominant)
    assert summary["fragmentation_index"] == pytest.approx(1 - expected_dominant)
    assert summary["largest_patch_fraction"] < summary["dominant_patch_fraction"]
    assert summary["patch_density_per_10000_km2"] > 0
    assert np.isfinite(summary["area_weighted_centroid_latitude"])
    assert np.isfinite(summary["area_weighted_centroid_longitude"])
    assert summary["patch_centroid_dispersion_km"] > 0


def test_single_patch_daily_summary_has_zero_fragmentation() -> None:
    values = np.zeros((3, 3))
    values[1, 1] = 3.0
    _, _, summary = _characterize(values)
    assert summary["largest_patch_area_km2"] == summary["total_patch_area_km2"]
    assert summary["dominant_patch_fraction"] == 1.0
    assert summary["fragmentation_index"] == 0.0
    assert summary["patch_centroid_dispersion_km"] == pytest.approx(0.0)
