"""Offline-only tests for local coastline discovery and loading."""

from __future__ import annotations

from pathlib import Path

import pytest
import shapefile
from shapely import get_coordinates
from shapely.geometry import MultiPolygon, box
from shapely.ops import unary_union

from src.coastline_source import (
    LocalCoastlineSource,
    coastline_source_status,
    ensure_no_network_download,
    find_local_coastline_source,
    extract_mainland_source_coastline,
    identify_south_america_mainland,
    load_local_coastline,
)
from src.geography import (
    GeographicBounds,
    build_geodesic_coastal_corridor,
    geometry_is_valid,
    load_geography_registry,
)
from src.utils import load_config


def _write_shape(path: Path, shape_type: int, coordinates) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    writer = shapefile.Writer(str(path), shapeType=shape_type)
    writer.field("name", "C")
    if shape_type == shapefile.POLYGON:
        writer.poly([coordinates])
    else:
        writer.line([coordinates])
    writer.record("test fixture")
    writer.close()


def synthetic_source(tmp_path: Path) -> LocalCoastlineSource:
    physical = tmp_path / "shapefiles/natural_earth/physical"
    cultural = tmp_path / "shapefiles/natural_earth/cultural"
    coast = physical / "ne_10m_coastline.shp"
    land = physical / "ne_10m_land.shp"
    borders = cultural / "ne_10m_admin_0_boundary_lines_land.shp"
    _write_shape(coast, shapefile.POLYLINE, [(-80.0, -44.0), (-80.0, 1.0)])
    _write_shape(
        land,
        shapefile.POLYGON,
        [
            (-80.0, -55.0),
            (-30.0, -55.0),
            (-30.0, 15.0),
            (-80.0, 15.0),
            (-80.0, -55.0),
        ],
    )
    _write_shape(borders, shapefile.POLYLINE, [(-75.0, -8.0), (-70.0, -8.0)])
    return LocalCoastlineSource("10m", tmp_path, coast, land, borders)


def test_missing_local_source_is_controlled(tmp_path) -> None:
    assert find_local_coastline_source(data_directories=[tmp_path]) is None
    status = coastline_source_status(None)
    assert status.available is False
    assert "no download" in status.message.lower()


def test_complete_local_source_is_discovered_without_private_status_path(tmp_path) -> None:
    source = synthetic_source(tmp_path)
    found = find_local_coastline_source(data_directories=[tmp_path])
    assert found == source
    status = found.status.to_dict()
    assert status["available"] is True
    assert status["path"] == "shapefiles/natural_earth/physical/ne_10m_coastline.shp"
    assert str(tmp_path) not in str(status)


def test_local_source_loads_and_clips_synthetic_fixture(tmp_path) -> None:
    source = synthetic_source(tmp_path)
    geometries = load_local_coastline(
        source,
        GeographicBounds(-85.0, -70.0, -45.0, 2.0),
        mainland_coastline_bounds=GeographicBounds(-85.0, -70.0, -45.0, 2.0),
    )
    assert geometry_is_valid(geometries.display_coastline)
    assert geometry_is_valid(geometries.all_land)
    assert geometry_is_valid(geometries.borders)
    assert geometry_is_valid(geometries.mainland_land)
    assert geometry_is_valid(geometries.mainland_source_coastline)
    assert geometries.island_land is None


def synthetic_mainland_and_islands():
    mainland = box(-75.0, -20.0, -68.0, 5.0)
    near_island = box(-75.80, -10.20, -75.60, -10.00)
    distant_island = box(-77.30, -8.20, -77.10, -8.00)
    controls = ((-74.0, -15.0), (-73.0, -8.0), (-72.0, 0.0))
    bounds = GeographicBounds(-80.0, -70.0, -15.0, 0.0)
    return mainland, near_island, distant_island, controls, bounds


def test_mainland_selection_excludes_disconnected_island_components() -> None:
    mainland, near_island, distant_island, controls, bounds = (
        synthetic_mainland_and_islands()
    )
    all_land = MultiPolygon([mainland, near_island, distant_island])
    selected = identify_south_america_mainland(all_land, control_points=controls)
    source = extract_mainland_source_coastline(selected, bounds)
    assert selected.equals(mainland)
    assert source.disjoint(near_island.boundary)
    assert source.disjoint(distant_island.boundary)
    coordinates = get_coordinates(source)
    assert all(longitude == pytest.approx(-75.0) for longitude in coordinates[:, 0])
    assert source.bounds == pytest.approx((-75.0, -15.0, -75.0, 0.0))


def test_islands_never_create_or_extend_the_geodesic_corridor() -> None:
    mainland, near_island, distant_island, controls, bounds = (
        synthetic_mainland_and_islands()
    )
    selected = identify_south_america_mainland(
        MultiPolygon([mainland, near_island, distant_island]),
        control_points=controls,
    )
    source = extract_mainland_source_coastline(selected, bounds)
    spec = load_geography_registry(load_config()).coastal_corridors["humboldt_60nm"]
    mainland_only = build_geodesic_coastal_corridor(
        source,
        spec,
        clip_bounds=bounds,
        land_geometry=mainland,
    )
    with_all_land = build_geodesic_coastal_corridor(
        source,
        spec,
        clip_bounds=bounds,
        land_geometry=unary_union([mainland, near_island, distant_island]),
    )
    assert geometry_is_valid(with_all_land)
    assert with_all_land.difference(mainland_only).is_empty
    assert with_all_land.intersection(near_island).area == pytest.approx(0.0)
    assert with_all_land.intersection(distant_island).area == pytest.approx(0.0)
    assert not with_all_land.contains(near_island.centroid)
    assert with_all_land.bounds == pytest.approx(mainland_only.bounds, abs=1.0e-9)
    assert not with_all_land.buffer(1.0e-12).contains(distant_island.centroid)


def test_network_downloader_is_explicitly_blocked() -> None:
    cartopy_io = pytest.importorskip("cartopy.io")
    with ensure_no_network_download(), pytest.raises(RuntimeError, match="disabled"):
        cartopy_io.Downloader.acquire_resource(None, None, None)
