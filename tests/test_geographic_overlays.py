"""Tests for non-scientific geographic foundation overlays."""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
from shapely.geometry import LineString, box

import src.geographic_overlays as overlays
from src.coastline_source import CoastlineSourceStatus, LocalGeometries
from src.geography import GeographicBounds, build_geodesic_coastal_corridor, load_geography_registry
from src.utils import load_config


def inputs():
    registry = load_geography_registry(load_config())
    coastline = LineString([(-75.0, -44.0), (-75.0, 1.0)])
    land = box(-75.0, -45.0, -70.0, 2.0)
    geometries = LocalGeometries(
        display_coastline=coastline,
        all_land=land,
        borders=None,
        mainland_land=land,
        mainland_source_coastline=coastline,
        island_land=None,
    )
    corridor = build_geodesic_coastal_corridor(
        coastline,
        registry.coastal_corridors["humboldt_60nm"],
        clip_bounds=registry.analysis_domains["humboldt_coastal"].bounds,
        land_geometry=land,
    )
    status = CoastlineSourceStatus(True, "test fixture", "synthetic", None, True, False, "available")
    return registry, geometries, corridor, status


def test_foundation_fallback_returns_figure_with_fixed_pacific_extent(monkeypatch) -> None:
    registry, geometries, corridor, status = inputs()
    monkeypatch.setattr(overlays, "ccrs", None)
    figure = overlays.plot_geographic_foundation(
        registry,
        local_geometries=geometries,
        corridor=corridor,
        source_status=status,
    )
    assert isinstance(figure, matplotlib.figure.Figure)
    axis = figure.axes[0]
    assert axis.get_xlim() == (-170.0, -70.0)
    assert axis.get_ylim() == (-45.0, 10.0)
    assert figure.get_facecolor()[:3] == (1.0, 1.0, 1.0)
    labels = {text.get_text() for text in axis.texts}
    assert {"Niño 3.4", "Niño 3", "Niño 1+2"}.issubset(labels)
    plt.close(figure)


def test_foundation_handles_missing_coast_and_corridor(monkeypatch) -> None:
    registry, _, _, _ = inputs()
    monkeypatch.setattr(overlays, "ccrs", None)
    status = CoastlineSourceStatus(
        False, "local cache", None, None, False, False, "Local coastline unavailable"
    )
    figure = overlays.plot_geographic_foundation(
        registry, local_geometries=None, corridor=None, source_status=status
    )
    assert isinstance(figure, matplotlib.figure.Figure)
    assert any("unavailable" in text.get_text().lower() for text in figure.axes[0].texts)
    plt.close(figure)


def test_rectangle_vertices_stay_within_configured_extent() -> None:
    registry, _, _, _ = inputs()
    for region in registry.standard_regions.values():
        assert registry.display_domain.bounds.contains(region.bounds)
        assert GeographicBounds(*(
            region.bounds.west,
            region.bounds.east,
            region.bounds.south,
            region.bounds.north,
        )) == region.bounds
