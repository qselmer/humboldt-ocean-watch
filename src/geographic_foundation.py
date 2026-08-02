"""Offline orchestration for geographic-foundation geometry and status."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping

from shapely.geometry.base import BaseGeometry

from src.coastline_source import (
    CoastlineSourceStatus,
    LocalCoastlineSource,
    LocalGeometries,
    coastline_source_status,
    ensure_no_network_download,
    find_local_coastline_source,
    load_local_coastline,
)
from src.geography import (
    GeographyRegistry,
    build_geodesic_coastal_corridor,
    geometry_bounds,
    load_geography_registry,
)


@dataclass(frozen=True)
class GeographicFoundation:
    registry: GeographyRegistry
    source_status: CoastlineSourceStatus
    local_geometries: LocalGeometries | None
    corridor: BaseGeometry | None
    warnings: tuple[str, ...]

    @property
    def corridor_status(self) -> dict[str, Any]:
        spec = self.registry.coastal_corridors["humboldt_60nm"]
        source_coastline = (
            None
            if self.local_geometries is None
            else self.local_geometries.mainland_source_coastline
        )
        source_parts = (
            0
            if source_coastline is None
            else 1
            if source_coastline.geom_type == "LineString"
            else len(getattr(source_coastline, "geoms", ()))
        )
        definition = {
            "source_coastline": spec.source_coastline,
            "coastline_type": spec.coastline_type,
            "include_islands": spec.include_islands,
            "islands_excluded_as_buffer_sources": True,
            "land_exclusion_mask": "all_land_including_islands",
            "all_land_retained_as_exclusion_mask": True,
            "buffer_distance_nm": spec.buffer_distance_nm,
            "buffer_distance_km": spec.buffer_distance_km,
            "distance_conversion": "60 nautical miles × 1.852 = 111.12 kilometres",
            "source_geometry_type": (
                None if source_coastline is None else source_coastline.geom_type
            ),
            "source_component_count": source_parts,
        }
        if self.corridor is None:
            status = (
                "geometry_source_unavailable"
                if self.local_geometries is None
                else "mainland_source_or_land_mask_unavailable"
            )
            return {
                "available": False,
                "status": status,
                "geometry_type": None,
                "bounds": None,
                **definition,
            }
        return {
            "available": True,
            "status": "valid",
            "geometry_type": self.corridor.geom_type,
            "bounds": geometry_bounds(self.corridor).to_dict(),
            **definition,
        }


def prepare_geographic_foundation(
    config: Mapping[str, Any],
    *,
    source: LocalCoastlineSource | None = None,
    local_geometries: LocalGeometries | None = None,
) -> GeographicFoundation:
    """Validate config, load only local resources, and build the 60 nm corridor."""
    registry = load_geography_registry(config)
    selected_source = source if source is not None else find_local_coastline_source()
    status = coastline_source_status(selected_source)
    warnings: list[str] = []
    geometries = local_geometries
    corridor = None
    with ensure_no_network_download():
        if geometries is None and selected_source is not None:
            geometries = load_local_coastline(
                selected_source,
                registry.display_domain.bounds,
                mainland_coastline_bounds=registry.analysis_domains[
                    "humboldt_coastal"
                ].bounds,
            )
        corridor_spec = registry.coastal_corridors["humboldt_60nm"]
        if geometries is None:
            warnings.append(status.message)
        elif corridor_spec.ocean_only and geometries.all_land is None:
            warnings.append(
                "A local coastline exists, but the complete local land mask is unavailable; "
                "the ocean-only 60 nm corridor was not created."
            )
        elif geometries.mainland_source_coastline is None:
            warnings.append(
                "The South America mainland Pacific coastline could not be extracted; "
                "no corridor was created and islands were not used as a fallback."
            )
        else:
            corridor = build_geodesic_coastal_corridor(
                geometries.mainland_source_coastline,
                corridor_spec,
                clip_bounds=registry.analysis_domains["humboldt_coastal"].bounds,
                land_geometry=geometries.all_land,
            )
    return GeographicFoundation(
        registry=registry,
        source_status=status,
        local_geometries=geometries,
        corridor=corridor,
        warnings=tuple(warnings),
    )
