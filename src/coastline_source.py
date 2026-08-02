"""Strictly local Natural Earth coastline discovery and loading.

No function in this module calls ``cartopy.io.shapereader.natural_earth`` or
another downloader.  Candidate paths are constructed directly beneath known
Cartopy data directories and must already contain the shapefile sidecars.
"""

from __future__ import annotations

from contextlib import contextmanager
from dataclasses import dataclass
import logging
from pathlib import Path
from typing import Iterator, Sequence

from shapely import make_valid, normalize
from shapely.geometry import Point, Polygon, box
from shapely.geometry.base import BaseGeometry
from shapely.ops import unary_union

from src.geography import GeographicBounds, geometry_is_valid

try:
    import cartopy
    from cartopy.io import Downloader
    from cartopy.io import shapereader
except ImportError:  # pragma: no cover - Cartopy is pinned but remains optional.
    cartopy = None
    Downloader = None
    shapereader = None


LOGGER = logging.getLogger(__name__)

SOUTH_AMERICA_SELECTION_BOUNDS = GeographicBounds(-90.0, -30.0, -60.0, 15.0)
SOUTH_AMERICA_MAINLAND_CONTROL_POINTS = (
    (-78.5, -1.3),  # Continental Ecuador
    (-75.0, -12.0),  # Continental Peru
    (-71.0, -33.0),  # Continental Chile
)


@dataclass(frozen=True)
class CoastlineSourceStatus:
    available: bool
    source: str
    resolution: str | None
    path: str | None
    land_available: bool
    borders_available: bool
    message: str
    coastline_type: str | None = None
    include_islands: bool | None = None
    land_exclusion_mask: str | None = None

    def to_dict(self) -> dict[str, object]:
        return {
            "available": self.available,
            "source": self.source,
            "resolution": self.resolution,
            "path": self.path,
            "land_available": self.land_available,
            "borders_available": self.borders_available,
            "message": self.message,
            "coastline_type": self.coastline_type,
            "include_islands": self.include_islands,
            "land_exclusion_mask": self.land_exclusion_mask,
        }


@dataclass(frozen=True)
class LocalCoastlineSource:
    """Validated local paths plus non-private serializable metadata."""

    resolution: str
    root: Path
    coastline_path: Path
    land_path: Path | None
    borders_path: Path | None

    @property
    def status(self) -> CoastlineSourceStatus:
        relative = self.coastline_path.relative_to(self.root).as_posix()
        return CoastlineSourceStatus(
            available=True,
            source="Cartopy Natural Earth local cache",
            resolution=self.resolution,
            path=relative,
            land_available=self.land_path is not None,
            borders_available=self.borders_path is not None,
            message=(
                "Local land geometry is available for deterministic mainland-coastline "
                "extraction; no download is required."
            ),
            coastline_type="continental_mainland_only",
            include_islands=False,
            land_exclusion_mask="all_land_including_islands",
        )


@dataclass(frozen=True)
class LocalGeometries:
    """Separated rendering, mainland-source, and full exclusion geometries."""

    display_coastline: BaseGeometry
    all_land: BaseGeometry | None
    borders: BaseGeometry | None
    mainland_land: BaseGeometry | None
    mainland_source_coastline: BaseGeometry | None
    island_land: BaseGeometry | None


def _complete_shapefile(path: Path) -> bool:
    return all(path.with_suffix(suffix).is_file() for suffix in (".shp", ".shx", ".dbf"))


def _cartopy_data_directories() -> tuple[Path, ...]:
    if cartopy is None:
        return ()
    candidates = [
        cartopy.config.get("pre_existing_data_dir"),
        cartopy.config.get("data_dir"),
        cartopy.config.get("repo_data_dir"),
    ]
    unique: list[Path] = []
    for value in candidates:
        if not value:
            continue
        path = Path(value).expanduser()
        if path not in unique:
            unique.append(path)
    return tuple(unique)


def find_local_coastline_source(
    *,
    data_directories: Sequence[str | Path] | None = None,
    preferred_resolutions: Sequence[str] = ("10m", "50m", "110m"),
) -> LocalCoastlineSource | None:
    """Return the best complete local coastline without initiating downloads."""
    roots = (
        tuple(Path(value).expanduser() for value in data_directories)
        if data_directories is not None
        else _cartopy_data_directories()
    )
    for resolution in preferred_resolutions:
        for root in roots:
            physical = root / "shapefiles" / "natural_earth" / "physical"
            cultural = root / "shapefiles" / "natural_earth" / "cultural"
            coastline = physical / f"ne_{resolution}_coastline.shp"
            if not _complete_shapefile(coastline):
                continue
            land_candidate = physical / f"ne_{resolution}_land.shp"
            borders_candidate = cultural / f"ne_{resolution}_admin_0_boundary_lines_land.shp"
            source = LocalCoastlineSource(
                resolution=resolution,
                root=root,
                coastline_path=coastline,
                land_path=(land_candidate if _complete_shapefile(land_candidate) else None),
                borders_path=(
                    borders_candidate if _complete_shapefile(borders_candidate) else None
                ),
            )
            LOGGER.info(
                "Using local Natural Earth coastline (%s; %s)",
                resolution,
                source.status.path,
            )
            return source
    return None


def coastline_source_status(
    source: LocalCoastlineSource | None,
) -> CoastlineSourceStatus:
    if source is not None:
        return source.status
    return CoastlineSourceStatus(
        available=False,
        source="Cartopy Natural Earth local cache",
        resolution=None,
        path=None,
        land_available=False,
        borders_available=False,
        message=(
            "No complete local Natural Earth coastline shapefile was found. "
            "The coastal corridor is unavailable and no download was attempted."
        ),
        coastline_type="continental_mainland_only",
        include_islands=False,
        land_exclusion_mask="all_land_including_islands",
    )


def _read_geometries(path: Path) -> list[BaseGeometry]:
    if shapereader is None:
        raise RuntimeError("Cartopy shapereader is unavailable")
    reader = shapereader.Reader(path)
    try:
        return [geometry for geometry in reader.geometries() if not geometry.is_empty]
    finally:
        close = getattr(reader, "close", None)
        if callable(close):
            close()


def _lineal(geometry: BaseGeometry) -> BaseGeometry:
    if geometry.geom_type in {"LineString", "MultiLineString"}:
        return geometry
    parts: list[BaseGeometry] = []
    for item in getattr(geometry, "geoms", ()):
        if item.geom_type == "LineString":
            parts.append(item)
        elif item.geom_type == "MultiLineString":
            parts.extend(item.geoms)
    return unary_union(parts)


def _polygon_parts(geometry: BaseGeometry) -> list[Polygon]:
    """Flatten polygonal geometry into non-empty components."""
    if geometry.geom_type == "Polygon":
        return [geometry]
    parts: list[Polygon] = []
    for item in getattr(geometry, "geoms", ()):
        if item.geom_type == "Polygon":
            parts.append(item)
        elif item.geom_type in {"MultiPolygon", "GeometryCollection"}:
            parts.extend(_polygon_parts(item))
    return [part for part in parts if not part.is_empty]


def _polygonal(geometry: BaseGeometry) -> BaseGeometry:
    parts = _polygon_parts(geometry)
    return unary_union(parts) if parts else Polygon()


def identify_south_america_mainland(
    land_geometry: BaseGeometry,
    *,
    control_points: Sequence[tuple[float, float]] = SOUTH_AMERICA_MAINLAND_CONTROL_POINTS,
) -> Polygon:
    """Select the connected continental component validated by three countries.

    Candidates must intersect a broad South America selection domain and cover
    continental control points in Ecuador, Peru, and Chile.  The component with
    the largest area inside that domain wins; area, bounds, and normalized WKB
    provide deterministic tie-breaking.
    """
    if not geometry_is_valid(land_geometry):
        raise ValueError("Complete land geometry must be finite, non-empty, and valid")
    selection_domain = box(
        SOUTH_AMERICA_SELECTION_BOUNDS.west,
        SOUTH_AMERICA_SELECTION_BOUNDS.south,
        SOUTH_AMERICA_SELECTION_BOUNDS.east,
        SOUTH_AMERICA_SELECTION_BOUNDS.north,
    )
    controls = tuple(
        Point(float(longitude), float(latitude))
        for longitude, latitude in control_points
    )
    if len(controls) < 3:
        raise ValueError("At least three mainland control points are required")
    candidates = [
        component
        for component in _polygon_parts(make_valid(land_geometry))
        if component.intersects(selection_domain)
        and all(component.covers(point) for point in controls)
    ]
    if not candidates:
        raise ValueError(
            "No connected land component contains the configured continental "
            "control points for Ecuador, Peru, and Chile"
        )

    def rank(component: Polygon) -> tuple[float, float, tuple[float, ...], str]:
        normalized = normalize(component)
        return (
            -float(component.intersection(selection_domain).area),
            -float(component.area),
            tuple(float(value) for value in component.bounds),
            normalized.wkb_hex,
        )

    mainland = normalize(make_valid(sorted(candidates, key=rank)[0]))
    if mainland.geom_type != "Polygon" or not geometry_is_valid(mainland):
        raise ValueError("Selected South America mainland component is invalid")
    if not all(mainland.covers(point) for point in controls):
        raise ValueError("Selected mainland failed continental control-point validation")
    return mainland


def extract_mainland_source_coastline(
    mainland: Polygon,
    bounds: GeographicBounds,
) -> BaseGeometry:
    """Extract the true exterior first, then select its Humboldt-domain segment.

    Intersecting the exterior line after extraction prevents artificial sides of
    the rectangular work domain from becoming coastline.  Interior rings and
    disconnected island polygons are never source lines.
    """
    if mainland.geom_type != "Polygon" or not geometry_is_valid(mainland):
        raise ValueError("Mainland must be a finite, valid Polygon")
    clipping = box(bounds.west, bounds.south, bounds.east, bounds.north)
    coastline = normalize(_lineal(mainland.exterior.intersection(clipping)))
    if not geometry_is_valid(coastline):
        raise ValueError("Continental Pacific coastline is unavailable in the requested bounds")
    return coastline


def load_local_coastline(
    source: LocalCoastlineSource,
    bounds: GeographicBounds,
    *,
    mainland_coastline_bounds: GeographicBounds | None = None,
) -> LocalGeometries:
    """Load local geometry and derive an island-free mainland buffer source."""
    clipping = box(bounds.west, bounds.south, bounds.east, bounds.north)
    display_coastline = normalize(_lineal(
        make_valid(unary_union(_read_geometries(source.coastline_path))).intersection(clipping)
    ))
    if not geometry_is_valid(display_coastline):
        raise ValueError("Local coastline is empty or invalid inside the requested bounds")

    all_land: BaseGeometry | None = None
    mainland_land: BaseGeometry | None = None
    mainland_source_coastline: BaseGeometry | None = None
    island_land: BaseGeometry | None = None
    if source.land_path is not None:
        complete_land = normalize(
            _polygonal(make_valid(unary_union(_read_geometries(source.land_path))))
        )
        if not geometry_is_valid(complete_land):
            raise ValueError("Complete local land geometry is invalid")
        mainland_complete = identify_south_america_mainland(complete_land)
        all_land_candidate = normalize(
            _polygonal(make_valid(complete_land.intersection(clipping)))
        )
        all_land = None if all_land_candidate.is_empty else all_land_candidate
        mainland_candidate = normalize(
            _polygonal(make_valid(mainland_complete.intersection(clipping)))
        )
        mainland_land = None if mainland_candidate.is_empty else mainland_candidate
        if all_land is None or not geometry_is_valid(all_land):
            raise ValueError("Local land geometry is invalid inside the requested bounds")
        if mainland_land is None or not geometry_is_valid(mainland_land):
            raise ValueError("South America mainland is unavailable inside the display domain")
        islands_candidate = normalize(
            _polygonal(make_valid(all_land.difference(mainland_land)))
        )
        island_land = None if islands_candidate.is_empty else islands_candidate
        if island_land is not None and not geometry_is_valid(island_land):
            raise ValueError("Separated island geometry is invalid")
        mainland_source_coastline = extract_mainland_source_coastline(
            mainland_complete,
            mainland_coastline_bounds or bounds,
        )

    borders: BaseGeometry | None = None
    if source.borders_path is not None:
        candidate = _lineal(
            make_valid(unary_union(_read_geometries(source.borders_path))).intersection(clipping)
        )
        borders = candidate if not candidate.is_empty else None
        if borders is not None and not geometry_is_valid(borders):
            raise ValueError("Local border geometry is invalid inside the requested bounds")
    return LocalGeometries(
        display_coastline=display_coastline,
        all_land=all_land,
        borders=borders,
        mainland_land=mainland_land,
        mainland_source_coastline=mainland_source_coastline,
        island_land=island_land,
    )


@contextmanager
def ensure_no_network_download() -> Iterator[None]:
    """Fail immediately if Cartopy attempts to acquire a missing resource."""
    if Downloader is None:
        yield
        return
    original = Downloader.acquire_resource

    def blocked(*_args: object, **_kwargs: object) -> object:
        raise RuntimeError("Network downloads are disabled for the geographic foundation")

    Downloader.acquire_resource = blocked  # type: ignore[method-assign]
    try:
        yield
    finally:
        Downloader.acquire_resource = original  # type: ignore[method-assign]
