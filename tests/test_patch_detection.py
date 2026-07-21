"""Tests for deterministic daily patch masks, connectivity, and filtering."""

import numpy as np
import pytest
import xarray as xr

from src.grid_geometry import spherical_cell_areas
from src.patch_detection import PatchDetectionConfig, identify_patches


def _field(values: np.ndarray, *, dims=("latitude", "longitude")) -> xr.DataArray:
    coordinates = {
        "latitude": -4.5 + np.arange(values.shape[0], dtype=float),
        "longitude": -89.5 + np.arange(values.shape[1], dtype=float),
    }
    data = xr.DataArray(values, dims=("latitude", "longitude"), coords=coordinates)
    return data.transpose(*dims)


def _config(**changes) -> PatchDetectionConfig:
    values = {
        "source_variable": "anomaly",
        "direction": "above",
        "threshold_type": "fixed",
        "comparison": "inclusive",
        "connectivity": "queen",
        "minimum_patch_cells": 1,
        "minimum_patch_area_km2": 0.0,
    }
    values.update(changes)
    return PatchDetectionConfig(**values)


def test_one_isolated_patch_and_two_separate_patches() -> None:
    one = np.zeros((4, 4))
    one[1, 1] = 3.0
    result = identify_patches(_field(one), 2.0, config=_config())
    assert result.retained_patch_count == 1
    assert result.patch_id.values[1, 1] == 1

    two = one.copy()
    two[3, 3] = 3.0
    result = identify_patches(_field(two), 2.0, config=_config(connectivity="rook"))
    assert result.retained_patch_count == 2


def test_diagonal_cells_depend_on_rook_or_queen_connectivity() -> None:
    values = np.zeros((3, 3))
    values[0, 0] = values[1, 1] = 3.0
    rook = identify_patches(_field(values), 2.0, config=_config(connectivity="rook"))
    queen = identify_patches(_field(values), 2.0, config=_config(connectivity="queen"))
    assert rook.retained_patch_count == 2
    assert queen.retained_patch_count == 1


def test_filtering_by_cell_count_and_area_relabels_consecutively() -> None:
    values = np.zeros((5, 5))
    values[0, 0] = 3.0
    values[3:5, 3:5] = 3.0
    count_filtered = identify_patches(
        _field(values), 2.0, config=_config(connectivity="rook", minimum_patch_cells=2)
    )
    assert count_filtered.removed_by_cell_count == 1
    assert set(np.unique(count_filtered.patch_id)) == {0, 1}
    assert count_filtered.patch_id.values[3, 3] == 1

    cell_area = float(spherical_cell_areas(_field(values)).values[0, 0])
    area_filtered = identify_patches(
        _field(values),
        2.0,
        config=_config(
            connectivity="rook",
            minimum_patch_area_km2=cell_area * 1.5,
        ),
    )
    assert area_filtered.removed_by_area == 1
    assert area_filtered.retained_patch_count == 1


def test_boundary_only_artifact_filter_is_explicit() -> None:
    values = np.zeros((4, 4))
    values[0, 0:2] = 3.0
    retained = identify_patches(_field(values), 2.0, config=_config())
    removed = identify_patches(
        _field(values),
        2.0,
        config=_config(remove_boundary_only_artifacts=True),
    )
    assert retained.retained_patch_count == 1
    assert removed.retained_patch_count == 0
    assert removed.removed_boundary_only == 1


def test_no_patch_is_explicit_and_all_nan_field_is_fatal() -> None:
    result = identify_patches(_field(np.zeros((3, 3))), 2.0, config=_config())
    assert result.retained_patch_count == 0
    assert result.status == "not_calculated"
    assert "No valid ocean cells" in (result.reason or "")
    with pytest.raises(ValueError, match="no finite ocean cells"):
        identify_patches(_field(np.full((3, 3), np.nan)), 2.0, config=_config())


def test_land_mask_excludes_threshold_cells() -> None:
    values = np.zeros((3, 3))
    values[1, 1] = 3.0
    ocean = xr.ones_like(_field(values), dtype=bool)
    ocean.values[1, 1] = False
    result = identify_patches(_field(values), 2.0, config=_config(), ocean_mask=ocean)
    assert result.retained_patch_count == 0
    assert not bool(result.valid_ocean_mask.values[1, 1])


def test_inclusive_and_exclusive_comparisons() -> None:
    values = np.zeros((2, 2))
    values[0, 0] = 2.0
    inclusive = identify_patches(
        _field(values), 2.0, config=_config(comparison="inclusive")
    )
    exclusive = identify_patches(
        _field(values), 2.0, config=_config(comparison="exclusive")
    )
    assert inclusive.retained_patch_count == 1
    assert exclusive.retained_patch_count == 0


def test_above_and_below_directions_produce_nonnegative_exceedance() -> None:
    values = np.array([[3.0, 0.0], [0.0, -2.0]])
    above = identify_patches(_field(values), 2.0, config=_config(direction="above"))
    below = identify_patches(
        _field(values), -1.0, config=_config(direction="below")
    )
    assert above.retained_patch_count == 1
    assert below.retained_patch_count == 1
    assert above.exceedance.values[0, 0] == 1.0
    assert below.exceedance.values[1, 1] == 1.0


def test_date_selection_transpose_and_coordinate_aliases() -> None:
    values = np.zeros((2, 3, 3))
    values[1, 1, 1] = 3.0
    cube = xr.DataArray(
        values.transpose(0, 2, 1),
        dims=("time", "lon", "lat"),
        coords={"time": ["2024-01-01", "2024-01-02"], "lat": [-2, -1, 0], "lon": [-90, -89, -88]},
    )
    result = identify_patches(
        cube, 2.0, analysis_date="2024-01-02", config=_config()
    )
    assert result.patch_id.dims == ("latitude", "longitude")
    assert result.retained_patch_count == 1
    with pytest.raises(ValueError, match="unavailable"):
        identify_patches(cube, 2.0, analysis_date="2024-01-03", config=_config())


def test_invalid_threshold_and_mask_are_rejected() -> None:
    field = _field(np.ones((3, 3)))
    with pytest.raises(ValueError, match="threshold must be finite"):
        identify_patches(field, np.inf, config=_config())
    with pytest.raises(ValueError, match="shape is incompatible"):
        identify_patches(field, 0.0, config=_config(), ocean_mask=np.ones((2, 2)))
