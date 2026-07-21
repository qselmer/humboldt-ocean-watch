"""Spherical latitude-longitude grid geometry for daily thermal patches.

Cell areas use the exact spherical quadrilateral expression
``R**2 * |sin(phi_north) - sin(phi_south)| * |lambda_east-lambda_west|``.
The documented IUGG mean Earth radius is used throughout. Patch covariance
coordinates use a local equirectangular projection centred on the patch;
this is appropriate for the compact 10 by 10 degree Nino 1+2 domain.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import xarray as xr

from src.quality_control import coordinate_names

EARTH_RADIUS_KM = 6371.0088


@dataclass(frozen=True)
class GridGeometry:
    """Area and exposed-edge geometry aligned to one two-dimensional grid."""

    area_km2: xr.DataArray
    latitude_edges: np.ndarray
    longitude_edges_unwrapped: np.ndarray
    north_edge_km: np.ndarray
    south_edge_km: np.ndarray
    east_edge_km: np.ndarray
    west_edge_km: np.ndarray
    row_before_edge_km: np.ndarray
    row_after_edge_km: np.ndarray
    column_before_edge_km: np.ndarray
    column_after_edge_km: np.ndarray
    latitude_name: str
    longitude_name: str


def _finite_centres(values: np.ndarray, name: str) -> np.ndarray:
    centres = np.asarray(values, dtype=float)
    if centres.ndim != 1:
        raise ValueError(f"{name} coordinate must be one-dimensional")
    if centres.size < 2:
        raise ValueError(f"{name} coordinate requires at least two cell centres")
    if not np.isfinite(centres).all():
        raise ValueError(f"{name} coordinate must contain only finite values")
    return centres


def unwrap_longitudes(longitude: np.ndarray) -> np.ndarray:
    """Unwrap longitudes while retaining their original convention and order."""
    values = _finite_centres(longitude, "longitude")
    return np.rad2deg(np.unwrap(np.deg2rad(values), discont=np.pi))


def infer_cell_edges(
    centres: np.ndarray,
    *,
    coordinate: str,
) -> np.ndarray:
    """Infer cell edges from monotonic centres using midpoint extrapolation.

    Latitude edges are clipped to the physical poles. Longitudes are first
    unwrapped, allowing either -180..180 or 0..360 coordinates and grids that
    cross the convention seam.
    """
    if coordinate not in {"latitude", "longitude"}:
        raise ValueError("coordinate must be 'latitude' or 'longitude'")
    values = _finite_centres(centres, coordinate)
    working = unwrap_longitudes(values) if coordinate == "longitude" else values
    differences = np.diff(working)
    if bool((differences == 0).any()) or not (
        bool((differences > 0).all()) or bool((differences < 0).all())
    ):
        raise ValueError(f"{coordinate} coordinate must be strictly monotonic")
    edges = np.empty(working.size + 1, dtype=float)
    edges[1:-1] = (working[:-1] + working[1:]) / 2.0
    edges[0] = working[0] - differences[0] / 2.0
    edges[-1] = working[-1] + differences[-1] / 2.0
    if coordinate == "latitude":
        if bool((np.abs(values) > 90.0).any()):
            raise ValueError("Latitude centres must lie between -90 and 90 degrees")
        edges = np.clip(edges, -90.0, 90.0)
    elif bool((np.abs(np.diff(edges)) >= 360.0).any()):
        raise ValueError("Longitude cell widths must be less than 360 degrees")
    if bool((np.diff(edges) == 0).any()):
        raise ValueError(f"Inferred {coordinate} cell edges have zero width")
    return edges


def build_grid_geometry(field: xr.DataArray) -> GridGeometry:
    """Calculate positive spherical cell areas and edge lengths in kilometres."""
    if not isinstance(field, xr.DataArray):
        raise TypeError("Grid geometry input must be an xarray.DataArray")
    latitude_name, longitude_name = coordinate_names(field)
    latitude = np.asarray(field[latitude_name].values, dtype=float)
    longitude = np.asarray(field[longitude_name].values, dtype=float)
    if field[latitude_name].dims != (latitude_name,):
        raise ValueError("Latitude coordinate must be one-dimensional and index latitude")
    if field[longitude_name].dims != (longitude_name,):
        raise ValueError("Longitude coordinate must be one-dimensional and index longitude")
    latitude_edges = infer_cell_edges(latitude, coordinate="latitude")
    longitude_edges = infer_cell_edges(longitude, coordinate="longitude")

    south = np.minimum(latitude_edges[:-1], latitude_edges[1:])
    north = np.maximum(latitude_edges[:-1], latitude_edges[1:])
    longitude_width = np.abs(np.diff(np.deg2rad(longitude_edges)))
    latitude_height = np.abs(np.diff(np.deg2rad(latitude_edges)))
    band_factor = np.abs(
        np.sin(np.deg2rad(north)) - np.sin(np.deg2rad(south))
    )
    area = EARTH_RADIUS_KM**2 * band_factor[:, None] * longitude_width[None, :]
    if not np.isfinite(area).all() or bool((area <= 0).any()):
        raise ValueError("Calculated grid-cell areas must be finite and positive")

    north_length = (
        EARTH_RADIUS_KM
        * np.cos(np.deg2rad(north))[:, None]
        * longitude_width[None, :]
    )
    south_length = (
        EARTH_RADIUS_KM
        * np.cos(np.deg2rad(south))[:, None]
        * longitude_width[None, :]
    )
    meridional_length = EARTH_RADIUS_KM * latitude_height[:, None]
    east_length = np.broadcast_to(meridional_length, area.shape).copy()
    west_length = east_length.copy()

    latitude_ascending = latitude[-1] > latitude[0]
    longitude_ascending = unwrap_longitudes(longitude)[-1] > unwrap_longitudes(longitude)[0]
    row_before = south_length if latitude_ascending else north_length
    row_after = north_length if latitude_ascending else south_length
    column_before = west_length if longitude_ascending else east_length
    column_after = east_length if longitude_ascending else west_length
    area_array = xr.DataArray(
        area,
        dims=(latitude_name, longitude_name),
        coords={latitude_name: field[latitude_name], longitude_name: field[longitude_name]},
        name="cell_area",
        attrs={
            "long_name": "spherical grid-cell area",
            "units": "km2",
            "earth_radius_km": EARTH_RADIUS_KM,
            "method": "spherical latitude-longitude quadrilateral from inferred centre edges",
        },
    )
    return GridGeometry(
        area_km2=area_array,
        latitude_edges=latitude_edges,
        longitude_edges_unwrapped=longitude_edges,
        north_edge_km=north_length,
        south_edge_km=south_length,
        east_edge_km=east_length,
        west_edge_km=west_length,
        row_before_edge_km=row_before,
        row_after_edge_km=row_after,
        column_before_edge_km=column_before,
        column_after_edge_km=column_after,
        latitude_name=latitude_name,
        longitude_name=longitude_name,
    )


def spherical_cell_areas(field: xr.DataArray) -> xr.DataArray:
    """Convenience wrapper returning spherical grid-cell area in square km."""
    return build_grid_geometry(field).area_km2


def weighted_geographic_centroid(
    mask: np.ndarray,
    geometry: GridGeometry,
) -> tuple[float, float]:
    """Return area-weighted latitude and circular-longitude centroid."""
    selected = np.asarray(mask, dtype=bool)
    area = np.asarray(geometry.area_km2.values, dtype=float)
    if selected.shape != area.shape:
        raise ValueError("Centroid mask is incompatible with grid geometry")
    weights = np.where(selected, area, 0.0)
    denominator = float(weights.sum())
    if denominator <= 0:
        return np.nan, np.nan
    latitude = np.asarray(geometry.area_km2[geometry.latitude_name].values, dtype=float)
    longitude = np.asarray(geometry.area_km2[geometry.longitude_name].values, dtype=float)
    longitude_grid, latitude_grid = np.meshgrid(longitude, latitude)
    centroid_latitude = float(np.sum(weights * latitude_grid) / denominator)
    radians = np.deg2rad(longitude_grid)
    sine = float(np.sum(weights * np.sin(radians)))
    cosine = float(np.sum(weights * np.cos(radians)))
    centroid_longitude = float(np.rad2deg(np.arctan2(sine, cosine)))
    if np.all((longitude >= 0.0) & (longitude <= 360.0)):
        centroid_longitude %= 360.0
    return centroid_latitude, centroid_longitude


def local_kilometre_coordinates(
    geometry: GridGeometry,
    *,
    origin_latitude: float,
    origin_longitude: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Project cell centres into local east/north kilometre coordinates."""
    latitude = np.asarray(geometry.area_km2[geometry.latitude_name].values, dtype=float)
    longitude = np.asarray(geometry.area_km2[geometry.longitude_name].values, dtype=float)
    longitude_grid, latitude_grid = np.meshgrid(longitude, latitude)
    longitude_delta = (longitude_grid - origin_longitude + 180.0) % 360.0 - 180.0
    x = (
        EARTH_RADIUS_KM
        * np.deg2rad(longitude_delta)
        * np.cos(np.deg2rad(origin_latitude))
    )
    y = EARTH_RADIUS_KM * np.deg2rad(latitude_grid - origin_latitude)
    return x, y


def patch_perimeter_km(mask: np.ndarray, geometry: GridGeometry) -> float:
    """Sum exposed rook edges, accounting for latitude-dependent side length."""
    selected = np.asarray(mask, dtype=bool)
    shape = geometry.area_km2.shape
    if selected.shape != shape:
        raise ValueError("Perimeter mask is incompatible with grid geometry")
    row_before_neighbour = np.zeros(shape, dtype=bool)
    row_before_neighbour[1:, :] = selected[:-1, :]
    row_after_neighbour = np.zeros(shape, dtype=bool)
    row_after_neighbour[:-1, :] = selected[1:, :]
    column_before_neighbour = np.zeros(shape, dtype=bool)
    column_before_neighbour[:, 1:] = selected[:, :-1]
    column_after_neighbour = np.zeros(shape, dtype=bool)
    column_after_neighbour[:, :-1] = selected[:, 1:]
    perimeter = (
        np.sum(geometry.row_before_edge_km[selected & ~row_before_neighbour])
        + np.sum(geometry.row_after_edge_km[selected & ~row_after_neighbour])
        + np.sum(geometry.column_before_edge_km[selected & ~column_before_neighbour])
        + np.sum(geometry.column_after_edge_km[selected & ~column_after_neighbour])
    )
    return float(perimeter)


def edge_cell_mask(mask: np.ndarray) -> np.ndarray:
    """Return patch cells with at least one exposed rook edge."""
    selected = np.asarray(mask, dtype=bool)
    if selected.ndim != 2:
        raise ValueError("Edge-cell mask must be two-dimensional")
    neighbours = np.zeros(selected.shape, dtype=np.int8)
    neighbours[1:, :] += selected[:-1, :]
    neighbours[:-1, :] += selected[1:, :]
    neighbours[:, 1:] += selected[:, :-1]
    neighbours[:, :-1] += selected[:, 1:]
    return selected & (neighbours < 4)
