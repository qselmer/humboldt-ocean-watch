"""Deterministic tests for the Increment 6A geographic registry."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import replace

import pytest
from shapely.geometry import LineString, Polygon, box

from src.geography import (
    GeographicBounds,
    build_geodesic_coastal_corridor,
    geometry_is_valid,
    kilometres_to_nautical_miles,
    load_geography_registry,
    nautical_miles_to_kilometres,
    rectangle_polygon,
    rectangle_vertices,
    resolve_standard_region_alias,
    validate_latitude_bounds,
    validate_longitude_bounds,
)
from src.utils import load_config


def config():
    return deepcopy(load_config())


def test_geography_loads_without_changing_legacy_region() -> None:
    values = config()
    registry = load_geography_registry(values)
    assert values["region"] == {
        "longitude": [-90.0, -80.0],
        "latitude": [-10.0, 0.0],
    }
    assert registry.display_domain.id == "pacific_context"
    assert set(registry.analysis_domains) == {
        "pacific_context",
        "humboldt_coastal",
        "south_pacific_high",
    }


@pytest.mark.parametrize(
    "bounds",
    [(-170.0, -70.0), (-90.0, -80.0), (-1.0, 1.0)],
)
def test_valid_longitude_bounds(bounds) -> None:
    assert validate_longitude_bounds(bounds) == bounds


@pytest.mark.parametrize(
    "bounds",
    [(-181.0, -70.0), (-90.0, 181.0), (-80.0, -90.0), (0.0, 0.0)],
)
def test_invalid_longitude_bounds(bounds) -> None:
    with pytest.raises(ValueError):
        validate_longitude_bounds(bounds)


@pytest.mark.parametrize("bounds", [(-91.0, 0.0), (-10.0, 91.0), (2.0, -45.0)])
def test_invalid_latitude_bounds(bounds) -> None:
    with pytest.raises(ValueError):
        validate_latitude_bounds(bounds)


def test_alias_resolution_and_ambiguous_alias_rejection() -> None:
    values = config()
    registry = load_geography_registry(values)
    assert resolve_standard_region_alias(registry, "nino1.2") == "nino12"
    assert resolve_standard_region_alias(registry, "NINO_3_4") == "nino34"
    values["geography"]["standard_regions"]["nino3"]["aliases"].append("nino34")
    with pytest.raises(ValueError, match="Ambiguous"):
        load_geography_registry(values)


@pytest.mark.parametrize(
    ("identifier", "vertices"),
    [
        (
            "nino34",
            ((-170.0, -5.0), (-120.0, -5.0), (-120.0, 5.0), (-170.0, 5.0), (-170.0, -5.0)),
        ),
        (
            "nino3",
            ((-150.0, -5.0), (-90.0, -5.0), (-90.0, 5.0), (-150.0, 5.0), (-150.0, -5.0)),
        ),
        (
            "nino12",
            ((-90.0, -10.0), (-80.0, -10.0), (-80.0, 0.0), (-90.0, 0.0), (-90.0, -10.0)),
        ),
    ],
)
def test_nino_rectangle_vertices_are_closed_and_deterministic(identifier, vertices) -> None:
    region = load_geography_registry(config()).standard_regions[identifier]
    assert rectangle_vertices(region) == vertices
    assert rectangle_vertices(region)[0] == rectangle_vertices(region)[-1]
    assert rectangle_polygon(region).equals(Polygon(vertices))


def test_rectangle_centres_dimensions_and_display_containment() -> None:
    registry = load_geography_registry(config())
    display = registry.display_domain.bounds
    nino12 = registry.standard_regions["nino12"]
    assert (nino12.central_longitude, nino12.central_latitude) == (-85.0, -5.0)
    assert (nino12.width, nino12.height) == (10.0, 10.0)
    assert all(display.contains(region.bounds) for region in registry.standard_regions.values())
    assert all(display.contains(domain.bounds) for domain in registry.analysis_domains.values())


def test_sixty_nautical_miles_conversion_is_exactly_configured() -> None:
    assert nautical_miles_to_kilometres(60.0) == pytest.approx(111.12)
    assert kilometres_to_nautical_miles(111.12) == pytest.approx(60.0)
    spec = load_geography_registry(config()).coastal_corridors["humboldt_60nm"]
    assert spec.source_coastline == "south_america_pacific_mainland"
    assert spec.coastline_type == "continental_mainland_only"
    assert spec.include_islands is False
    assert spec.latitude_range == (-45.0, 2.0)
    assert spec.buffer_distance_nm == 60.0
    assert spec.buffer_distance_km == 111.12


def test_island_buffer_sources_are_rejected_by_configuration() -> None:
    values = config()
    values["geography"]["coastal_corridors"]["humboldt_60nm"][
        "include_islands"
    ] = True
    with pytest.raises(ValueError, match="Islands cannot be buffer sources"):
        load_geography_registry(values)


def synthetic_corridor_inputs():
    registry = load_geography_registry(config())
    spec = registry.coastal_corridors["humboldt_60nm"]
    bounds = GeographicBounds(-80.0, -70.0, -15.0, 0.0)
    coastline = LineString([(-75.0, -14.0), (-75.0, -1.0)])
    land = box(-75.0, -15.0, -70.0, 0.0)
    return spec, bounds, coastline, land


def test_geodesic_corridor_is_ocean_only_valid_and_deterministic() -> None:
    spec, bounds, coastline, land = synthetic_corridor_inputs()
    first = build_geodesic_coastal_corridor(
        coastline, spec, clip_bounds=bounds, land_geometry=land
    )
    second = build_geodesic_coastal_corridor(
        coastline, spec, clip_bounds=bounds, land_geometry=land
    )
    assert geometry_is_valid(first)
    assert first.geom_type in {"Polygon", "MultiPolygon"}
    assert first.wkb == second.wkb
    assert first.bounds[0] < -75.5
    assert first.bounds[2] <= -75.0 + 1.0e-8
    assert -180.0 <= first.bounds[0] <= first.bounds[2] <= 180.0
    assert -45.0 <= first.bounds[1] <= first.bounds[3] <= 2.0


def test_geodesic_corridor_rejects_empty_geometry_and_zero_distance() -> None:
    spec, bounds, _, land = synthetic_corridor_inputs()
    with pytest.raises(ValueError, match="Coastline"):
        build_geodesic_coastal_corridor(
            LineString(), spec, clip_bounds=bounds, land_geometry=land
        )
    with pytest.raises(ValueError, match="greater than zero"):
        build_geodesic_coastal_corridor(
            LineString([(-75.0, -14.0), (-75.0, -1.0)]),
            replace(spec, buffer_distance_nm=0.0),
            clip_bounds=bounds,
            land_geometry=land,
        )
