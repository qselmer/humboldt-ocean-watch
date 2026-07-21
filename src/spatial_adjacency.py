"""Grid adjacency, Moran I, and daily threshold-patch characterization.

Adjacency is represented as directed sparse edge vectors. Moran weights are
row-standardized over the valid-neighbour graph by default. Patch metrics are
daily snapshots only: labels are never linked between analysis dates.
"""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
from typing import Literal

import numpy as np
from scipy import ndimage

from src.metric_result import MetricResult, not_calculated

Connectivity = Literal["rook", "queen"]


@dataclass(frozen=True)
class GridAdjacency:
    """Directed grid-neighbour edge list suitable for sparse calculations."""

    rows: np.ndarray
    columns: np.ndarray
    shape: tuple[int, int]
    connectivity: Connectivity
    row_standardized: bool = True

    @property
    def n_edges(self) -> int:
        return int(self.rows.size)


def _offsets(connectivity: Connectivity) -> tuple[tuple[int, int], ...]:
    if connectivity == "rook":
        return ((-1, 0), (1, 0), (0, -1), (0, 1))
    if connectivity == "queen":
        return tuple(
            (row, column)
            for row in (-1, 0, 1)
            for column in (-1, 0, 1)
            if (row, column) != (0, 0)
        )
    raise ValueError("Connectivity must be 'rook' or 'queen'")


@lru_cache(maxsize=16)
def build_adjacency(
    shape: tuple[int, int],
    *,
    connectivity: Connectivity = "queen",
    row_standardized: bool = True,
) -> GridAdjacency:
    """Build directed rook or queen adjacency without a GIS dependency."""
    if len(shape) != 2 or shape[0] < 1 or shape[1] < 1:
        raise ValueError("Grid shape must contain two positive dimensions")
    rows: list[int] = []
    columns: list[int] = []
    height, width = shape
    for row in range(height):
        for column in range(width):
            source = row * width + column
            for row_offset, column_offset in _offsets(connectivity):
                neighbour_row = row + row_offset
                neighbour_column = column + column_offset
                if 0 <= neighbour_row < height and 0 <= neighbour_column < width:
                    rows.append(source)
                    columns.append(neighbour_row * width + neighbour_column)
    return GridAdjacency(
        rows=np.asarray(rows, dtype=np.int64),
        columns=np.asarray(columns, dtype=np.int64),
        shape=shape,
        connectivity=connectivity,
        row_standardized=row_standardized,
    )


def calculate_moran_i(
    values: np.ndarray,
    adjacency: GridAdjacency,
    *,
    valid_weights: np.ndarray | None = None,
    minimum_valid_coverage: float = 0.80,
) -> MetricResult:
    """Calculate global Moran I, restandardizing rows after masking invalid cells."""
    data = np.asarray(values, dtype=float)
    if data.shape != adjacency.shape:
        raise ValueError("Moran field shape is incompatible with adjacency")
    weights = np.ones(data.shape, dtype=float) if valid_weights is None else np.asarray(valid_weights, dtype=float)
    if weights.shape != data.shape:
        raise ValueError("Moran coverage weights are incompatible with the field")
    if not np.isfinite(weights).all() or bool((weights < 0).any()):
        raise ValueError("Moran coverage weights must be finite and non-negative")
    flat = data.ravel()
    flat_weights = weights.ravel()
    valid = np.isfinite(flat) & (flat_weights > 0)
    total_weight = float(flat_weights.sum())
    coverage = float(flat_weights[valid].sum() / total_weight) if total_weight > 0 else np.nan
    metric = f"moran_i_{adjacency.connectivity}"
    n_valid = int(valid.sum())
    if not np.isfinite(coverage) or coverage < minimum_valid_coverage:
        return not_calculated(
            metric, f"Valid coverage {coverage:.3f} is below {minimum_valid_coverage:.3f}",
            unit="1", family="spatial_autocorrelation", n_observations=n_valid,
            valid_coverage=coverage,
            metadata={"connectivity": adjacency.connectivity, "row_standardized": adjacency.row_standardized},
        )
    if n_valid < 3:
        return not_calculated(
            metric, "At least three valid cells are required", unit="1",
            family="spatial_autocorrelation", n_observations=n_valid,
            valid_coverage=coverage,
        )
    row = adjacency.rows
    column = adjacency.columns
    edge_valid = valid[row] & valid[column]
    row = row[edge_valid]
    column = column[edge_valid]
    if row.size == 0:
        return not_calculated(
            metric, "Adjacency contains no valid neighbour pairs", unit="1",
            family="spatial_autocorrelation", n_observations=n_valid,
            valid_coverage=coverage,
        )
    centred = flat - float(np.mean(flat[valid]))
    denominator = float(np.sum(centred[valid] ** 2))
    if denominator <= 0:
        return not_calculated(
            metric, "Spatial variance is zero", unit="1", family="spatial_autocorrelation",
            n_observations=n_valid, valid_coverage=coverage,
        )
    if adjacency.row_standardized:
        row_counts = np.bincount(row, minlength=flat.size)
        edge_weights = 1.0 / row_counts[row]
    else:
        edge_weights = np.ones(row.size, dtype=float)
    weight_sum = float(edge_weights.sum())
    if weight_sum <= 0:
        return not_calculated(
            metric, "Adjacency weight denominator is zero", unit="1",
            family="spatial_autocorrelation", n_observations=n_valid,
            valid_coverage=coverage,
        )
    numerator = float(np.sum(edge_weights * centred[row] * centred[column]))
    value = n_valid / weight_sum * numerator / denominator
    return MetricResult(
        metric=metric, value=value, status="valid", n_observations=n_valid,
        valid_coverage=coverage, unit="1", family="spatial_autocorrelation",
        metadata={"connectivity": adjacency.connectivity, "row_standardized": adjacency.row_standardized},
    )


def label_patches(
    threshold_mask: np.ndarray,
    *,
    connectivity: Connectivity = "queen",
    minimum_patch_cells: int = 1,
) -> tuple[np.ndarray, int]:
    """Label daily patches and remove labels smaller than the cell threshold."""
    if minimum_patch_cells < 1:
        raise ValueError("minimum_patch_cells must be positive")
    mask = np.asarray(threshold_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Patch mask must be two-dimensional")
    structure = ndimage.generate_binary_structure(2, 1 if connectivity == "rook" else 2)
    if connectivity not in {"rook", "queen"}:
        raise ValueError("Connectivity must be 'rook' or 'queen'")
    raw_labels, count = ndimage.label(mask, structure=structure)
    retained = np.zeros(mask.shape, dtype=np.int32)
    next_label = 0
    for label in range(1, count + 1):
        cells = raw_labels == label
        if int(cells.sum()) >= minimum_patch_cells:
            next_label += 1
            retained[cells] = next_label
    return retained, next_label


def calculate_patch_features(
    threshold_mask: np.ndarray,
    latitude: np.ndarray,
    longitude: np.ndarray,
    *,
    area_weights: np.ndarray | None = None,
    valid_mask: np.ndarray | None = None,
    connectivity: Connectivity = "queen",
    minimum_patch_cells: int = 1,
) -> list[MetricResult]:
    """Characterize retained daily patches; fragmentation is 1-sum(area shares squared)."""
    mask = np.asarray(threshold_mask, dtype=bool)
    if mask.ndim != 2:
        raise ValueError("Patch mask must be two-dimensional")
    lat = np.asarray(latitude, dtype=float)
    lon = np.asarray(longitude, dtype=float)
    if lat.ndim != 1 or lon.ndim != 1 or mask.shape != (lat.size, lon.size):
        raise ValueError("Patch coordinates are incompatible with the mask")
    area = np.ones(mask.shape, dtype=float) if area_weights is None else np.asarray(area_weights, dtype=float)
    if area.shape != mask.shape:
        raise ValueError("Patch area weights are incompatible with the mask")
    if not np.isfinite(area).all() or bool((area <= 0).any()):
        raise ValueError("Patch area weights must be finite and positive")
    available = np.ones(mask.shape, dtype=bool) if valid_mask is None else np.asarray(valid_mask, dtype=bool)
    if available.shape != mask.shape:
        raise ValueError("Patch valid mask is incompatible with the threshold mask")
    labels, count = label_patches(mask & available, connectivity=connectivity, minimum_patch_cells=minimum_patch_cells)
    retained = labels > 0
    available_area = float(area[available].sum())
    coverage = float(available_area / area.sum()) if area.sum() > 0 else np.nan
    areas = np.asarray([area[labels == label].sum() for label in range(1, count + 1)], dtype=float)
    total_area = float(areas.sum())
    n_cells = int(retained.sum())
    common = dict(n_observations=n_cells, valid_coverage=coverage, family="patch")

    results = [
        MetricResult(metric="patch_count", value=float(count), unit="count", **common),
        MetricResult(metric="total_threshold_area", value=total_area, unit="weighted_area", **common),
        MetricResult(metric="patch_density", value=float(count / available_area) if available_area > 0 else np.nan, unit="per_weighted_area", **common),
    ]
    undefined_names = {
        "largest_patch_fraction": "1", "edge_cell_fraction": "1",
        "centroid_dispersion": "degrees", "orientation": "degrees",
        "elongation": "1",
    }
    if count == 0 or total_area <= 0:
        results.extend([
            MetricResult(metric="largest_patch_area", value=0.0, unit="weighted_area", **common),
            MetricResult(metric="mean_patch_area", value=0.0, unit="weighted_area", **common),
            MetricResult(metric="patch_area_standard_deviation", value=0.0, unit="weighted_area", **common),
            MetricResult(metric="fragmentation_index", value=0.0, unit="1", **common),
        ])
        results.extend(
            not_calculated(name, "No retained cells exceed the threshold", unit=unit, **common)
            for name, unit in undefined_names.items()
        )
        return results

    largest = float(areas.max())
    shares = areas / total_area
    results.extend([
        MetricResult(metric="largest_patch_area", value=largest, unit="weighted_area", **common),
        MetricResult(metric="largest_patch_fraction", value=largest / total_area, unit="1", **common),
        MetricResult(metric="mean_patch_area", value=float(areas.mean()), unit="weighted_area", **common),
        MetricResult(metric="patch_area_standard_deviation", value=float(areas.std()), unit="weighted_area", **common),
        MetricResult(metric="fragmentation_index", value=float(1.0 - np.sum(shares**2)), unit="1", **common),
    ])

    structure = ndimage.generate_binary_structure(2, 1 if connectivity == "rook" else 2).astype(int)
    neighbour_count = ndimage.convolve(retained.astype(int), structure, mode="constant", cval=0) - retained.astype(int)
    expected_neighbours = int(structure.sum() - 1)
    edge_cells = retained & (neighbour_count < expected_neighbours)
    results.append(MetricResult(metric="edge_cell_fraction", value=float(edge_cells.sum() / n_cells), unit="1", **common))

    lon_grid, lat_grid = np.meshgrid(lon, lat)
    patch_weights = area * retained
    weight_sum = float(patch_weights.sum())
    centre_lon = float(np.sum(patch_weights * lon_grid) / weight_sum)
    centre_lat = float(np.sum(patch_weights * lat_grid) / weight_sum)
    dx = lon_grid - centre_lon
    dy = lat_grid - centre_lat
    dispersion = float(np.sqrt(np.sum(patch_weights * (dx**2 + dy**2)) / weight_sum))
    results.append(MetricResult(metric="centroid_dispersion", value=dispersion, unit="degrees", **common))

    covariance = np.asarray([
        [np.sum(patch_weights * dx * dx), np.sum(patch_weights * dx * dy)],
        [np.sum(patch_weights * dx * dy), np.sum(patch_weights * dy * dy)],
    ]) / weight_sum
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    if eigenvalues[-1] <= 0:
        results.extend([
            not_calculated("orientation", "Patch geometry has no spatial variance", unit="degrees", **common),
            not_calculated("elongation", "Patch geometry has no spatial variance", unit="1", **common),
        ])
    else:
        major = eigenvectors[:, -1]
        orientation = float(np.degrees(np.arctan2(major[1], major[0])))
        results.append(MetricResult(metric="orientation", value=orientation, unit="degrees", **common))
        if eigenvalues[0] <= np.finfo(float).eps:
            results.append(not_calculated("elongation", "Minor-axis variance is zero", unit="1", **common))
        else:
            results.append(MetricResult(metric="elongation", value=float(np.sqrt(eigenvalues[-1] / eigenvalues[0])), unit="1", **common))
    return results
