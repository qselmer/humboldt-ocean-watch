"""Deterministic area-aware block aggregation for regular SST grids."""

from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Any

import numpy as np
import xarray as xr


@dataclass(frozen=True)
class SSTAggregationMetadata:
    aggregation_applied: bool
    method: str
    source_latitude_resolution_degrees: float
    source_longitude_resolution_degrees: float
    target_resolution_degrees: float
    latitude_factor: int
    longitude_factor: int
    minimum_valid_fraction: float
    partial_edge_blocks: bool

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _regular_resolution(
    coordinate: xr.DataArray, *, tolerance: float, name: str
) -> float:
    values = np.asarray(coordinate.values, dtype=float)
    if values.ndim != 1 or values.size < 2 or not np.isfinite(values).all():
        raise ValueError(f"{name} coordinate must contain at least two finite values")
    differences = np.diff(values)
    if not bool((differences > 0).all()):
        raise ValueError(f"{name} coordinate must be strictly increasing")
    resolution = float(np.median(differences))
    if not np.allclose(differences, resolution, rtol=0.0, atol=tolerance):
        raise ValueError(f"{name} coordinate is not regular within tolerance {tolerance}")
    return resolution


def _factor(
    source: float, target: float, *, tolerance: float, coordinate: str
) -> int:
    if source > target + tolerance:
        raise ValueError(f"Cannot upsample {coordinate} from {source:g}° to {target:g}°")
    ratio = target / source
    nearest = round(ratio)
    if nearest < 1 or not math.isclose(ratio, nearest, rel_tol=0.0, abs_tol=tolerance):
        raise ValueError(
            f"Target/source resolution factor for {coordinate} must be approximately integer; "
            f"received {ratio:g}"
        )
    return int(nearest)


def aggregate_sst_to_target_resolution(
    dataset: xr.Dataset,
    target_resolution_degrees: float,
    *,
    minimum_valid_fraction: float = 0.80,
    regular_grid_tolerance: float = 0.001,
    integer_factor_tolerance: float = 0.001,
) -> tuple[xr.Dataset, SSTAggregationMetadata]:
    """Aggregate regular SST blocks with cosine-latitude weighting.

    Partial blocks are retained at eastern and northern edges and are subjected
    to the same weighted valid-coverage gate as complete blocks.
    """
    required = {"time", "latitude", "longitude"}
    if required - set(dataset.coords) or "sst" not in dataset:
        raise ValueError("SST aggregation requires time, latitude, longitude, and sst")
    if not 0.0 < minimum_valid_fraction <= 1.0:
        raise ValueError("minimum_valid_fraction must be in (0, 1]")
    if not math.isfinite(target_resolution_degrees) or target_resolution_degrees <= 0:
        raise ValueError("target_resolution_degrees must be finite and positive")
    if dataset.sst.attrs.get("units") != "degrees_Celsius":
        raise ValueError("SST aggregation requires degrees_Celsius")
    lat_resolution = _regular_resolution(
        dataset.latitude, tolerance=regular_grid_tolerance, name="latitude"
    )
    lon_resolution = _regular_resolution(
        dataset.longitude, tolerance=regular_grid_tolerance, name="longitude"
    )
    lat_factor = _factor(
        lat_resolution,
        target_resolution_degrees,
        tolerance=integer_factor_tolerance,
        coordinate="latitude",
    )
    lon_factor = _factor(
        lon_resolution,
        target_resolution_degrees,
        tolerance=integer_factor_tolerance,
        coordinate="longitude",
    )
    applied = lat_factor > 1 or lon_factor > 1
    partial = (
        dataset.sizes["latitude"] % lat_factor != 0
        or dataset.sizes["longitude"] % lon_factor != 0
    )
    metadata = SSTAggregationMetadata(
        aggregation_applied=applied,
        method=("cosine_latitude_weighted_block_mean" if applied else "none"),
        source_latitude_resolution_degrees=lat_resolution,
        source_longitude_resolution_degrees=lon_resolution,
        target_resolution_degrees=float(target_resolution_degrees),
        latitude_factor=lat_factor,
        longitude_factor=lon_factor,
        minimum_valid_fraction=float(minimum_valid_fraction),
        partial_edge_blocks=partial,
    )
    if not applied:
        result = dataset.copy()
        result.attrs.update(metadata.to_dict())
        return result, metadata

    field = dataset.sst.transpose("time", "latitude", "longitude")
    weights = np.cos(np.deg2rad(dataset.latitude)).broadcast_like(
        field.isel(time=0, drop=True)
    )
    coarsen = {"latitude": lat_factor, "longitude": lon_factor}
    valid = field.notnull()
    numerator = (field * weights).where(valid).coarsen(
        **coarsen, boundary="pad"
    ).sum(skipna=True)
    valid_weight = weights.where(valid).coarsen(
        **coarsen, boundary="pad"
    ).sum(skipna=True)
    total_weight = weights.coarsen(**coarsen, boundary="pad").sum(skipna=True)
    coverage = (valid_weight / total_weight).transpose(*field.dims)
    aggregated = (numerator / valid_weight).where(
        (coverage >= minimum_valid_fraction) & (valid_weight > 0)
    )
    if bool(np.isinf(aggregated).any()):
        raise ValueError("Aggregation produced infinite SST values")
    result = aggregated.to_dataset(name="sst")
    result.sst.attrs = dict(dataset.sst.attrs)
    result.sst.attrs.update(
        aggregation_method="cosine_latitude_weighted_block_mean",
        minimum_valid_fraction=float(minimum_valid_fraction),
    )
    result["valid_coverage"] = coverage
    result.valid_coverage.attrs.update(units="1", long_name="weighted valid SST coverage")
    result.attrs = dict(dataset.attrs)
    result.attrs.update(metadata.to_dict())
    return result, metadata
