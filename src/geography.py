"""Typed geographic registry and offline geodesic geometry helpers.

Increment 6A deliberately keeps this registry separate from the legacy
``config["region"]`` calculation domain.  All configured longitudes use the
``-180_to_180`` convention, and antimeridian-crossing domains are rejected.

The coastal corridor is built in latitude bands.  Each band is buffered in a
local azimuthal-equidistant projection, transformed back to WGS84, clipped to
its band, and unioned.  This avoids treating nautical miles as angular degrees
while keeping projection distortion bounded across the long Humboldt coast.
"""

from __future__ import annotations

from dataclasses import dataclass
import math
import re
from typing import Any, Iterable, Iterator, Mapping, Sequence

import numpy as np

try:
    from pyproj import CRS, Transformer
    from shapely import get_coordinates, make_valid, normalize
    from shapely.geometry import LineString, MultiLineString, MultiPolygon, Polygon, box
    from shapely.geometry.base import BaseGeometry
    from shapely.ops import transform, unary_union
except ImportError:  # pragma: no cover - dependencies are pinned for this project.
    CRS = Transformer = None
    BaseGeometry = object  # type: ignore[assignment,misc]
    LineString = MultiLineString = MultiPolygon = Polygon = None
    box = get_coordinates = make_valid = normalize = transform = unary_union = None


NAUTICAL_MILE_KM = 1.852
COORDINATE_CONVENTION = "-180_to_180"
_IDENTIFIER = re.compile(r"^[a-z][a-z0-9_]*$")


def _finite_float(value: Any, name: str) -> float:
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be a finite number")
    return result


def validate_longitude_bounds(values: Sequence[Any]) -> tuple[float, float]:
    """Validate a non-antimeridian WGS84 longitude range."""
    if len(values) != 2:
        raise ValueError("Longitude bounds must contain exactly two values")
    west = _finite_float(values[0], "minimum longitude")
    east = _finite_float(values[1], "maximum longitude")
    if west < -180.0 or east > 180.0:
        raise ValueError("Longitude bounds must remain within [-180, 180]")
    if west >= east:
        raise ValueError(
            "Minimum longitude must be less than maximum longitude; "
            "antimeridian-crossing domains are not supported"
        )
    return west, east


def validate_latitude_bounds(values: Sequence[Any]) -> tuple[float, float]:
    """Validate an ordered WGS84 latitude range."""
    if len(values) != 2:
        raise ValueError("Latitude bounds must contain exactly two values")
    south = _finite_float(values[0], "minimum latitude")
    north = _finite_float(values[1], "maximum latitude")
    if south < -90.0 or north > 90.0:
        raise ValueError("Latitude bounds must remain within [-90, 90]")
    if south >= north:
        raise ValueError("Minimum latitude must be less than maximum latitude")
    return south, north


@dataclass(frozen=True)
class GeographicBounds:
    """Ordered WGS84 rectangle bounds."""

    west: float
    east: float
    south: float
    north: float

    def __post_init__(self) -> None:
        validate_longitude_bounds((self.west, self.east))
        validate_latitude_bounds((self.south, self.north))

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "GeographicBounds":
        west, east = validate_longitude_bounds(values.get("longitude", ()))
        south, north = validate_latitude_bounds(values.get("latitude", ()))
        return cls(west, east, south, north)

    @property
    def longitude(self) -> tuple[float, float]:
        return self.west, self.east

    @property
    def latitude(self) -> tuple[float, float]:
        return self.south, self.north

    @property
    def central_longitude(self) -> float:
        return (self.west + self.east) / 2.0

    @property
    def central_latitude(self) -> float:
        return (self.south + self.north) / 2.0

    @property
    def width(self) -> float:
        return self.east - self.west

    @property
    def height(self) -> float:
        return self.north - self.south

    def contains(self, other: "GeographicBounds", *, tolerance: float = 1.0e-10) -> bool:
        return (
            other.west >= self.west - tolerance
            and other.east <= self.east + tolerance
            and other.south >= self.south - tolerance
            and other.north <= self.north + tolerance
        )

    def to_dict(self) -> dict[str, list[float]]:
        return {
            "longitude": [self.west, self.east],
            "latitude": [self.south, self.north],
        }


@dataclass(frozen=True)
class GeographicDomain:
    """Named display or future analysis domain."""

    id: str
    label: str
    bounds: GeographicBounds
    target_resolution_degrees: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            **self.bounds.to_dict(),
            "target_resolution_degrees": self.target_resolution_degrees,
        }


@dataclass(frozen=True)
class StandardRegion:
    """Canonical rectangular climate-index region."""

    id: str
    label: str
    aliases: tuple[str, ...]
    geometry_type: str
    bounds: GeographicBounds

    @property
    def central_longitude(self) -> float:
        return self.bounds.central_longitude

    @property
    def central_latitude(self) -> float:
        return self.bounds.central_latitude

    @property
    def width(self) -> float:
        return self.bounds.width

    @property
    def height(self) -> float:
        return self.bounds.height

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "aliases": list(self.aliases),
            "geometry_type": self.geometry_type,
            **self.bounds.to_dict(),
            "central_longitude": self.central_longitude,
            "central_latitude": self.central_latitude,
            "width_degrees": self.width,
            "height_degrees": self.height,
            "vertices": [list(vertex) for vertex in rectangle_vertices(self)],
        }


@dataclass(frozen=True)
class CoastalCorridorSpec:
    """Configuration for a geodesic ocean-side coastal corridor."""

    id: str
    label: str
    geometry_type: str
    source_coastline: str
    coastline_type: str
    include_islands: bool
    latitude_range: tuple[float, float]
    buffer_distance_nm: float
    buffer_distance_km: float
    ocean_only: bool

    @property
    def latitude_bounds(self) -> tuple[float, float]:
        return self.latitude_range

    def to_dict(self) -> dict[str, Any]:
        return {
            "id": self.id,
            "label": self.label,
            "geometry_type": self.geometry_type,
            "source_coastline": self.source_coastline,
            "coastline_type": self.coastline_type,
            "include_islands": self.include_islands,
            "latitude_range": list(self.latitude_range),
            "buffer_distance_nm": self.buffer_distance_nm,
            "buffer_distance_km": self.buffer_distance_km,
            "ocean_only": self.ocean_only,
            "land_exclusion_mask": "all_land_including_islands",
        }


@dataclass(frozen=True)
class GeographyRegistry:
    """Validated immutable registry for the future multidomain scope."""

    coordinate_convention: str
    display_domain: GeographicDomain
    analysis_domains: Mapping[str, GeographicDomain]
    standard_regions: Mapping[str, StandardRegion]
    coastal_corridors: Mapping[str, CoastalCorridorSpec]
    map_overlays: Mapping[str, bool]
    alias_index: Mapping[str, str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "coordinate_convention": self.coordinate_convention,
            "display_domain": self.display_domain.to_dict(),
            "analysis_domains": {
                key: value.to_dict() for key, value in self.analysis_domains.items()
            },
            "standard_regions": {
                key: value.to_dict() for key, value in self.standard_regions.items()
            },
            "coastal_corridors": {
                key: value.to_dict() for key, value in self.coastal_corridors.items()
            },
            "map_overlays": dict(self.map_overlays),
        }


def _validate_identifier(identifier: str, context: str) -> str:
    value = str(identifier).strip()
    if not _IDENTIFIER.fullmatch(value):
        raise ValueError(f"{context} ID must use lowercase English ASCII: {value!r}")
    return value


def validate_domain(
    domain: GeographicDomain, *, within: GeographicBounds | None = None
) -> None:
    """Validate one domain and optional containment."""
    _validate_identifier(domain.id, "Domain")
    if not domain.label.strip():
        raise ValueError(f"Domain {domain.id!r} requires a visible label")
    if domain.target_resolution_degrees is not None:
        resolution = _finite_float(
            domain.target_resolution_degrees, f"{domain.id} target resolution"
        )
        if resolution <= 0:
            raise ValueError(f"{domain.id} target resolution must be positive")
    if within is not None and not within.contains(domain.bounds):
        raise ValueError(f"Domain {domain.id!r} falls outside the display domain")


def nautical_miles_to_kilometres(value: float) -> float:
    distance = _finite_float(value, "nautical-mile distance")
    return distance * NAUTICAL_MILE_KM


def kilometres_to_nautical_miles(value: float) -> float:
    distance = _finite_float(value, "kilometre distance")
    return distance / NAUTICAL_MILE_KM


def _domain(identifier: str, values: Mapping[str, Any]) -> GeographicDomain:
    resolution = values.get("target_resolution_degrees")
    return GeographicDomain(
        id=_validate_identifier(identifier, "Domain"),
        label=str(values.get("label", "")).strip(),
        bounds=GeographicBounds.from_mapping(values),
        target_resolution_degrees=(
            None if resolution is None else _finite_float(resolution, "target resolution")
        ),
    )


def _standard_region(identifier: str, values: Mapping[str, Any]) -> StandardRegion:
    aliases = values.get("aliases", ())
    if not isinstance(aliases, Sequence) or isinstance(aliases, (str, bytes)):
        raise ValueError(f"Aliases for {identifier!r} must be a sequence")
    return StandardRegion(
        id=_validate_identifier(identifier, "Standard-region"),
        label=str(values.get("label", "")).strip(),
        aliases=tuple(str(alias).strip().casefold() for alias in aliases),
        geometry_type=str(values.get("geometry_type", "")).strip(),
        bounds=GeographicBounds.from_mapping(values),
    )


def _corridor(identifier: str, values: Mapping[str, Any]) -> CoastalCorridorSpec:
    latitude_range = validate_latitude_bounds(values.get("latitude_range", ()))
    return CoastalCorridorSpec(
        id=_validate_identifier(identifier, "Coastal-corridor"),
        label=str(values.get("label", "")).strip(),
        geometry_type=str(values.get("geometry_type", "")).strip(),
        source_coastline=str(values.get("source_coastline", "")).strip(),
        coastline_type=str(values.get("coastline_type", "")).strip(),
        include_islands=bool(values.get("include_islands", False)),
        latitude_range=latitude_range,
        buffer_distance_nm=_finite_float(
            values.get("buffer_distance_nm"), "coastal buffer distance in nm"
        ),
        buffer_distance_km=_finite_float(
            values.get("buffer_distance_km"), "coastal buffer distance in km"
        ),
        ocean_only=bool(values.get("ocean_only", True)),
    )


def load_geography_registry(config: Mapping[str, Any]) -> GeographyRegistry:
    """Load and fully validate ``config['geography']``."""
    values = config.get("geography")
    if not isinstance(values, Mapping):
        raise ValueError("Configuration must contain a 'geography' mapping")
    convention = str(values.get("coordinate_convention", "")).strip()
    if convention != COORDINATE_CONVENTION:
        raise ValueError(
            f"Unsupported longitude convention {convention!r}; expected {COORDINATE_CONVENTION!r}"
        )
    display_values = values.get("display_domain")
    if not isinstance(display_values, Mapping):
        raise ValueError("geography.display_domain must be a mapping")
    display_id = _validate_identifier(str(display_values.get("id", "")), "Display-domain")
    display = _domain(display_id, display_values)
    validate_domain(display)

    analysis_values = values.get("analysis_domains")
    region_values = values.get("standard_regions")
    corridor_values = values.get("coastal_corridors")
    if not all(isinstance(item, Mapping) for item in (analysis_values, region_values, corridor_values)):
        raise ValueError(
            "analysis_domains, standard_regions, and coastal_corridors must be mappings"
        )
    analysis = {str(key): _domain(str(key), item) for key, item in analysis_values.items()}
    regions = {
        str(key): _standard_region(str(key), item) for key, item in region_values.items()
    }
    corridors = {
        str(key): _corridor(str(key), item) for key, item in corridor_values.items()
    }
    if not analysis or not regions or not corridors:
        raise ValueError("Geography registry requires domains, standard regions, and corridors")

    for domain in analysis.values():
        validate_domain(domain, within=display.bounds)
    for region in regions.values():
        if region.geometry_type != "rectangle":
            raise ValueError(f"Standard region {region.id!r} must use rectangle geometry")
        if not region.label:
            raise ValueError(f"Standard region {region.id!r} requires a label")
        if not display.bounds.contains(region.bounds):
            raise ValueError(f"Standard region {region.id!r} falls outside the display domain")

    required_domains = {"pacific_context", "humboldt_coastal", "south_pacific_high"}
    required_regions = {"nino34", "nino3", "nino12"}
    if not required_domains.issubset(analysis):
        raise ValueError(f"Missing analysis domains: {sorted(required_domains - set(analysis))}")
    if not required_regions.issubset(regions):
        raise ValueError(f"Missing standard regions: {sorted(required_regions - set(regions))}")
    if analysis["south_pacific_high"].label != "South Pacific High diagnostic domain":
        raise ValueError("The South Pacific High domain must be labelled as a diagnostic domain")

    alias_index: dict[str, str] = {}
    for region in regions.values():
        names = (region.id, *region.aliases)
        if any(not name for name in names):
            raise ValueError(f"Standard region {region.id!r} contains an empty alias")
        for name in names:
            normalized_name = name.casefold()
            existing = alias_index.get(normalized_name)
            if existing is not None and existing != region.id:
                raise ValueError(
                    f"Ambiguous standard-region alias {name!r}: {existing!r} and {region.id!r}"
                )
            alias_index[normalized_name] = region.id

    for corridor in corridors.values():
        if corridor.geometry_type != "geodesic_coastal_buffer":
            raise ValueError(f"Coastal corridor {corridor.id!r} requires geodesic buffering")
        if corridor.source_coastline != "south_america_pacific_mainland":
            raise ValueError(
                "Humboldt coastal corridor must use south_america_pacific_mainland"
            )
        if corridor.coastline_type != "continental_mainland_only":
            raise ValueError(
                "Humboldt coastal corridor coastline_type must be continental_mainland_only"
            )
        if corridor.include_islands:
            raise ValueError("Islands cannot be buffer sources for the Humboldt corridor")
        if not corridor.ocean_only:
            raise ValueError("Humboldt coastal corridor must retain all land as an exclusion mask")
        if corridor.buffer_distance_nm <= 0 or corridor.buffer_distance_km <= 0:
            raise ValueError("Coastal corridor distance must be greater than zero")
        expected_km = nautical_miles_to_kilometres(corridor.buffer_distance_nm)
        if not math.isclose(expected_km, corridor.buffer_distance_km, abs_tol=1.0e-9):
            raise ValueError(
                f"Coastal corridor distance is inconsistent: {corridor.buffer_distance_nm:g} nm "
                f"equals {expected_km:g} km"
            )
        south, north = corridor.latitude_bounds
        validate_latitude_bounds((south, north))
        if not math.isclose(north, 2.0) or not math.isclose(south, -45.0):
            raise ValueError("Humboldt coastal corridor must be limited from 2°N to 45°S")

    overlay_values = values.get("map_overlays", {})
    if not isinstance(overlay_values, Mapping):
        raise ValueError("geography.map_overlays must be a mapping")
    overlays = {str(key): bool(value) for key, value in overlay_values.items()}
    return GeographyRegistry(
        coordinate_convention=convention,
        display_domain=display,
        analysis_domains=analysis,
        standard_regions=regions,
        coastal_corridors=corridors,
        map_overlays=overlays,
        alias_index=alias_index,
    )


def get_display_domain(registry: GeographyRegistry) -> GeographicDomain:
    return registry.display_domain


def get_analysis_domain(registry: GeographyRegistry, identifier: str) -> GeographicDomain:
    try:
        return registry.analysis_domains[identifier]
    except KeyError as exc:
        raise KeyError(f"Unknown analysis domain: {identifier}") from exc


def resolve_standard_region_alias(registry: GeographyRegistry, alias: str) -> str:
    normalized = str(alias).strip().casefold()
    try:
        return registry.alias_index[normalized]
    except KeyError as exc:
        raise KeyError(f"Unknown standard-region alias: {alias}") from exc


def get_standard_region(registry: GeographyRegistry, identifier_or_alias: str) -> StandardRegion:
    identifier = resolve_standard_region_alias(registry, identifier_or_alias)
    return registry.standard_regions[identifier]


def iter_standard_regions(registry: GeographyRegistry) -> Iterator[StandardRegion]:
    return iter(registry.standard_regions.values())


def iter_analysis_domains(registry: GeographyRegistry) -> Iterator[GeographicDomain]:
    return iter(registry.analysis_domains.values())


def rectangle_vertices(
    region: StandardRegion | GeographicDomain | GeographicBounds,
) -> tuple[tuple[float, float], ...]:
    """Return deterministic west-south-first closed rectangle vertices."""
    bounds = region if isinstance(region, GeographicBounds) else region.bounds
    return (
        (bounds.west, bounds.south),
        (bounds.east, bounds.south),
        (bounds.east, bounds.north),
        (bounds.west, bounds.north),
        (bounds.west, bounds.south),
    )


def rectangle_polygon(
    region: StandardRegion | GeographicDomain | GeographicBounds,
) -> BaseGeometry:
    if Polygon is None:
        raise RuntimeError("Shapely is required to create geographic polygons")
    polygon = Polygon(rectangle_vertices(region))
    if not geometry_is_valid(polygon):
        raise ValueError("Rectangle geometry is invalid")
    return polygon


def geometry_bounds(geometry: BaseGeometry) -> GeographicBounds:
    if not geometry_is_valid(geometry):
        raise ValueError("Geometry must be finite, non-empty, and valid")
    west, south, east, north = map(float, geometry.bounds)
    return GeographicBounds(west, east, south, north)


def geometry_is_valid(geometry: BaseGeometry | None) -> bool:
    if geometry is None or getattr(geometry, "is_empty", True):
        return False
    if not bool(getattr(geometry, "is_valid", False)):
        return False
    if get_coordinates is None:
        return False
    coordinates = np.asarray(get_coordinates(geometry), dtype=float)
    return bool(coordinates.size and np.isfinite(coordinates).all())


def _polygonal(geometry: BaseGeometry) -> BaseGeometry:
    if geometry.geom_type in {"Polygon", "MultiPolygon"}:
        return geometry
    polygons: list[BaseGeometry] = []
    for item in getattr(geometry, "geoms", ()):  # GeometryCollection
        if item.geom_type == "Polygon":
            polygons.append(item)
        elif item.geom_type == "MultiPolygon":
            polygons.extend(item.geoms)
    return unary_union(polygons) if polygons else Polygon()


def build_geodesic_coastal_corridor(
    coastline: BaseGeometry,
    spec: CoastalCorridorSpec,
    *,
    clip_bounds: GeographicBounds,
    land_geometry: BaseGeometry | None = None,
    segment_height_degrees: float = 5.0,
) -> BaseGeometry:
    """Buffer a real or test coastline in local metric projections.

    The output is WGS84.  If ``spec.ocean_only`` is true, a valid land geometry
    is required so that the landward half of the buffer can be removed.
    """
    if CRS is None or Transformer is None or LineString is None:
        raise RuntimeError("PyProj and Shapely are required for coastal corridors")
    if not geometry_is_valid(coastline):
        raise ValueError("Coastline geometry must be finite, non-empty, and valid")
    if coastline.geom_type not in {"LineString", "MultiLineString"}:
        raise TypeError("Coastline geometry must be a LineString or MultiLineString")
    if spec.include_islands or spec.coastline_type != "continental_mainland_only":
        raise ValueError("Coastal corridor source must be continental mainland only")
    if spec.buffer_distance_nm <= 0 or spec.buffer_distance_km <= 0:
        raise ValueError("Coastal corridor distance must be greater than zero")
    if segment_height_degrees <= 0 or not math.isfinite(segment_height_degrees):
        raise ValueError("segment_height_degrees must be positive and finite")
    expected_km = nautical_miles_to_kilometres(spec.buffer_distance_nm)
    if not math.isclose(expected_km, spec.buffer_distance_km, abs_tol=1.0e-9):
        raise ValueError("Configured nautical-mile and kilometre distances are inconsistent")
    if spec.ocean_only and not geometry_is_valid(land_geometry):
        raise ValueError("A valid land geometry is required for an ocean-only corridor")

    south, north = spec.latitude_bounds
    working_bounds = GeographicBounds(
        clip_bounds.west,
        clip_bounds.east,
        max(clip_bounds.south, south),
        min(clip_bounds.north, north),
    )
    clipping_polygon = box(
        working_bounds.west,
        working_bounds.south,
        working_bounds.east,
        working_bounds.north,
    )
    clipped_coastline = make_valid(coastline).intersection(clipping_polygon)
    line_parts: list[BaseGeometry] = []
    if clipped_coastline.geom_type == "LineString":
        line_parts = [clipped_coastline]
    elif clipped_coastline.geom_type == "MultiLineString":
        line_parts = list(clipped_coastline.geoms)
    elif clipped_coastline.geom_type == "GeometryCollection":
        line_parts = [
            item
            for item in clipped_coastline.geoms
            if item.geom_type in {"LineString", "MultiLineString"} and not item.is_empty
        ]
    clipped_coastline = unary_union(line_parts)
    if clipped_coastline.is_empty:
        raise ValueError("Coastline does not intersect the configured coastal domain")

    wgs84 = CRS.from_epsg(4326)
    buffered_bands: list[BaseGeometry] = []
    band_south = working_bounds.south
    overlap_degrees = max(spec.buffer_distance_km / 90.0, 1.25)
    while band_south < working_bounds.north - 1.0e-12:
        band_north = min(band_south + segment_height_degrees, working_bounds.north)
        expanded = box(
            working_bounds.west,
            max(working_bounds.south, band_south - overlap_degrees),
            working_bounds.east,
            min(working_bounds.north, band_north + overlap_degrees),
        )
        segment = clipped_coastline.intersection(expanded)
        if not segment.is_empty:
            latitude_0 = (band_south + band_north) / 2.0
            longitude_0 = working_bounds.central_longitude
            local = CRS.from_proj4(
                f"+proj=aeqd +lat_0={latitude_0:.12f} +lon_0={longitude_0:.12f} "
                "+datum=WGS84 +units=m +no_defs"
            )
            forward = Transformer.from_crs(wgs84, local, always_xy=True).transform
            inverse = Transformer.from_crs(local, wgs84, always_xy=True).transform
            projected = transform(forward, segment)
            buffered = projected.buffer(
                spec.buffer_distance_km * 1000.0,
                quad_segs=16,
                cap_style="round",
                join_style="round",
            )
            restored = transform(inverse, buffered)
            core_band = box(
                working_bounds.west,
                band_south,
                working_bounds.east,
                band_north,
            )
            clipped = make_valid(restored).intersection(core_band)
            if not clipped.is_empty:
                buffered_bands.append(clipped)
        band_south = band_north

    if not buffered_bands:
        raise ValueError("No coastal corridor geometry could be constructed")
    corridor = make_valid(unary_union(buffered_bands)).intersection(clipping_polygon)
    if spec.ocean_only:
        corridor = make_valid(corridor.difference(make_valid(land_geometry)))
    corridor = _polygonal(make_valid(corridor))
    corridor = normalize(corridor)
    if not geometry_is_valid(corridor):
        raise ValueError("Constructed coastal corridor is empty, invalid, or non-finite")
    return corridor
