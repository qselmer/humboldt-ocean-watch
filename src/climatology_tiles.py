"""Deterministic non-overlapping spatial tiles for climatology production."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from typing import Any, Iterable

from src.geography import GeographicBounds


@dataclass(frozen=True)
class TileGrid:
    tile_id: str
    row: int
    column: int
    bounds: GeographicBounds
    latitude_start: int
    latitude_stop: int
    longitude_start: int
    longitude_stop: int
    target_cells: int

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["bounds"] = self.bounds.to_dict()
        return values


def coordinate_count(lower: float, upper: float, resolution: float) -> int:
    if resolution <= 0 or lower >= upper:
        raise ValueError("Ordered bounds and a positive resolution are required")
    intervals = (upper - lower) / resolution
    rounded = round(intervals)
    if not math.isclose(intervals, rounded, rel_tol=0.0, abs_tol=1.0e-6):
        raise ValueError("Bounds must align to the configured regular grid")
    return int(rounded) + 1


def build_spatial_tiles(
    bounds: GeographicBounds,
    target_resolution: float,
    tile_target_cells: int,
) -> tuple[TileGrid, ...]:
    """Partition target-grid indices once, with shared coordinate edges only.

    Index intervals are half-open. Requested coordinate bounds may share their
    outer coordinate for safe subsetting, while ownership remains unique.
    """
    if tile_target_cells < 1:
        raise ValueError("tile_target_cells must be positive")
    nlat = coordinate_count(bounds.south, bounds.north, target_resolution)
    nlon = coordinate_count(bounds.west, bounds.east, target_resolution)
    lat_span = max(1, min(nlat, int(math.sqrt(tile_target_cells))))
    lon_span = max(1, min(nlon, tile_target_cells // lat_span))
    def partitions(size: int, maximum: int) -> tuple[tuple[int, int], ...]:
        count = math.ceil(size / maximum)
        base, remainder = divmod(size, count)
        lengths = [base + (1 if index < remainder else 0) for index in range(count)]
        if any(length < 2 for length in lengths):
            raise ValueError("Tile plan would create a degenerate one-coordinate rectangle")
        result = []
        start = 0
        for length in lengths:
            result.append((start, start + length))
            start += length
        return tuple(result)

    tiles: list[TileGrid] = []
    for row, (lat_start, lat_stop) in enumerate(partitions(nlat, lat_span)):
        for column, (lon_start, lon_stop) in enumerate(partitions(nlon, lon_span)):
            south = bounds.south + lat_start * target_resolution
            north = bounds.south + (lat_stop - 1) * target_resolution
            west = bounds.west + lon_start * target_resolution
            east = bounds.west + (lon_stop - 1) * target_resolution
            tiles.append(TileGrid(
                tile_id=f"r{row:03d}_c{column:03d}", row=row, column=column,
                bounds=GeographicBounds(west, east, south, north),
                latitude_start=lat_start, latitude_stop=lat_stop,
                longitude_start=lon_start, longitude_stop=lon_stop,
                target_cells=(lat_stop - lat_start) * (lon_stop - lon_start),
            ))
    return tuple(tiles)


def validate_tile_coverage(
    tiles: Iterable[TileGrid], bounds: GeographicBounds, resolution: float
) -> None:
    items = tuple(tiles)
    expected_lat = coordinate_count(bounds.south, bounds.north, resolution)
    expected_lon = coordinate_count(bounds.west, bounds.east, resolution)
    owned: set[tuple[int, int]] = set()
    for tile in items:
        for latitude in range(tile.latitude_start, tile.latitude_stop):
            for longitude in range(tile.longitude_start, tile.longitude_stop):
                key = (latitude, longitude)
                if key in owned:
                    raise ValueError(f"Duplicate target cell in tile {tile.tile_id}")
                owned.add(key)
    expected = expected_lat * expected_lon
    if len(owned) != expected:
        raise ValueError(f"Tile grid covers {len(owned)} cells; expected {expected}")
