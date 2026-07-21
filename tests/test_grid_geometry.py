"""Deterministic spherical latitude-longitude grid geometry tests."""

import numpy as np
import pytest
import xarray as xr

from src.grid_geometry import (
    EARTH_RADIUS_KM,
    build_grid_geometry,
    infer_cell_edges,
    spherical_cell_areas,
)


def _field(latitude, longitude) -> xr.DataArray:
    return xr.DataArray(
        np.ones((len(latitude), len(longitude))),
        dims=("latitude", "longitude"),
        coords={"latitude": latitude, "longitude": longitude},
    )


def test_regular_grid_cell_area_matches_spherical_quadrilateral() -> None:
    field = _field([-0.5, 0.5], [0.5, 1.5])
    area = spherical_cell_areas(field)
    expected = (
        EARTH_RADIUS_KM**2
        * (np.sin(np.deg2rad(0.0)) - np.sin(np.deg2rad(-1.0)))
        * np.deg2rad(1.0)
    )
    assert float(area.isel(latitude=0, longitude=0)) == pytest.approx(expected)
    assert area.attrs["units"] == "km2"


def test_cell_area_decreases_with_absolute_latitude() -> None:
    area = spherical_cell_areas(_field([0.5, 1.5, 2.5], [0.5, 1.5]))
    assert float(area.isel(latitude=0, longitude=0)) > float(
        area.isel(latitude=2, longitude=0)
    )


def test_ascending_and_descending_latitudes_remain_aligned() -> None:
    ascending = spherical_cell_areas(_field([-2.5, -1.5, -0.5], [0.5, 1.5]))
    descending = spherical_cell_areas(_field([-0.5, -1.5, -2.5], [0.5, 1.5]))
    np.testing.assert_allclose(ascending.values, descending.values[::-1])
    np.testing.assert_array_equal(descending.latitude, [-0.5, -1.5, -2.5])


def test_longitude_conventions_and_seam_crossing_have_equal_area() -> None:
    negative = spherical_cell_areas(_field([-1.5, -0.5], [-89.5, -88.5]))
    positive = spherical_cell_areas(_field([-1.5, -0.5], [270.5, 271.5]))
    seam = spherical_cell_areas(_field([-1.5, -0.5], [359.5, 0.5]))
    reference = spherical_cell_areas(_field([-1.5, -0.5], [-0.5, 0.5]))
    np.testing.assert_allclose(negative, positive)
    np.testing.assert_allclose(seam, reference)


def test_irregular_edge_inference_uses_midpoints_and_extrapolation() -> None:
    edges = infer_cell_edges(np.array([0.0, 1.0, 3.0]), coordinate="longitude")
    np.testing.assert_allclose(edges, [-0.5, 0.5, 2.0, 4.0])
    polar = infer_cell_edges(np.array([89.0, 89.8]), coordinate="latitude")
    assert polar[-1] == 90.0


def test_invalid_geometry_rejects_duplicates_and_nonpositive_cells() -> None:
    with pytest.raises(ValueError, match="monotonic"):
        infer_cell_edges(np.array([0.0, 0.0]), coordinate="latitude")
    with pytest.raises(ValueError, match="between -90 and 90"):
        build_grid_geometry(_field([90.1, 89.0], [0.0, 1.0]))


def test_geometry_returns_positive_edge_lengths() -> None:
    geometry = build_grid_geometry(_field([-2.0, -0.5, 1.5], [270.0, 271.0, 273.0]))
    assert bool((geometry.area_km2.values > 0).all())
    assert bool((geometry.north_edge_km > 0).all())
    assert bool((geometry.east_edge_km > 0).all())
