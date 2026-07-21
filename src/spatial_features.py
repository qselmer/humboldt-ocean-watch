"""Advanced daily spatial features for gridded thermal fields.

Threshold area is the percentage of valid cosine-weighted area. Weighted
excess is the valid-area mean of ``max(field - threshold, 0)``. Threshold
centroids use excess magnitude times area as their weights.
"""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr
from scipy import ndimage

from src.metric_result import MetricResult, not_calculated
from src.quality_control import coordinate_names, cosine_latitude_weights
from src.series_bank import weighted_median, weighted_quantile
from src.spatial_adjacency import (
    Connectivity,
    build_adjacency,
    calculate_moran_i,
    calculate_patch_features,
)

SPATIAL_COLUMNS = [
    "date", "variable", "family", "metric", "value", "unit", "status",
    "reason", "n_observations", "valid_coverage", "threshold", "connectivity",
]


def _metric(
    name: str,
    value: float,
    *,
    family: str,
    unit: str,
    n: int,
    coverage: float,
    metadata: dict[str, Any] | None = None,
) -> MetricResult:
    return MetricResult(
        metric=name, value=value, family=family, unit=unit,
        n_observations=n, valid_coverage=coverage, metadata=metadata or {},
    )


def _not_calculated_family(
    names: Iterable[tuple[str, str]],
    *,
    family: str,
    reason: str,
    n: int,
    coverage: float,
) -> list[MetricResult]:
    return [
        not_calculated(
            name, reason, unit=unit, family=family,
            n_observations=n, valid_coverage=coverage,
        )
        for name, unit in names
    ]


def _weighted_gradient(
    values: np.ndarray,
    coordinate: np.ndarray,
    weights: np.ndarray,
) -> float:
    valid = np.isfinite(values) & np.isfinite(coordinate) & np.isfinite(weights) & (weights > 0)
    if int(valid.sum()) < 2 or np.unique(coordinate[valid]).size < 2:
        return np.nan
    x = coordinate[valid]
    y = values[valid]
    w = weights[valid]
    x_mean = float(np.sum(w * x) / np.sum(w))
    y_mean = float(np.sum(w * y) / np.sum(w))
    denominator = float(np.sum(w * (x - x_mean) ** 2))
    return float(np.sum(w * (x - x_mean) * (y - y_mean)) / denominator) if denominator > 0 else np.nan


def _local_standard_deviation(values: np.ndarray, window_cells: int) -> np.ndarray:
    if window_cells < 1 or window_cells % 2 == 0:
        raise ValueError("local_window_cells must be a positive odd integer")
    valid = np.isfinite(values)
    kernel = np.ones((window_cells, window_cells), dtype=float)
    count = ndimage.convolve(valid.astype(float), kernel, mode="constant", cval=0.0)
    total = ndimage.convolve(np.where(valid, values, 0.0), kernel, mode="constant", cval=0.0)
    square_total = ndimage.convolve(np.where(valid, values**2, 0.0), kernel, mode="constant", cval=0.0)
    with np.errstate(invalid="ignore", divide="ignore"):
        local_mean = total / count
        variance = np.maximum(square_total / count - local_mean**2, 0.0)
    return np.where(count >= 2, np.sqrt(variance), np.nan)


def calculate_spatial_features(
    field: xr.DataArray,
    *,
    variable: str,
    weights: xr.DataArray | np.ndarray | None = None,
    threshold: float = 2.0,
    minimum_valid_coverage: float = 0.80,
    connectivity: Connectivity = "queen",
    minimum_patch_cells: int = 4,
    moran_weights: str = "row_standardized",
    local_window_cells: int = 3,
) -> list[MetricResult]:
    """Calculate the complete daily spatial feature set for a 2-D field."""
    if not isinstance(field, xr.DataArray):
        raise TypeError("Spatial feature input must be an xarray.DataArray")
    lat_name, lon_name = coordinate_names(field)
    if "time" in field.dims or set(field.dims) != {lat_name, lon_name}:
        raise ValueError("Spatial feature input must be exactly two-dimensional after date selection")
    if not np.issubdtype(field.dtype, np.number):
        raise TypeError("Spatial feature input must be numeric")
    field = field.transpose(lat_name, lon_name)
    values = np.asarray(field.values, dtype=float)
    if np.isinf(values).any():
        raise ValueError("Spatial feature input contains infinite values")
    finite = np.isfinite(values)
    n_valid = int(finite.sum())

    if weights is None:
        weight_array = np.asarray(cosine_latitude_weights(field).values, dtype=float)
    elif isinstance(weights, xr.DataArray):
        weight_array = np.asarray(weights.broadcast_like(field).transpose(lat_name, lon_name).values, dtype=float)
    else:
        weight_array = np.asarray(weights, dtype=float)
    if weight_array.shape != values.shape:
        raise ValueError("Spatial weights are incompatible with the field")
    if not np.isfinite(weight_array).all() or bool((weight_array < 0).any()):
        raise ValueError("Spatial weights must be finite and non-negative")
    total_weight = float(weight_array.sum())
    valid_weight = float(weight_array[finite].sum())
    coverage = float(valid_weight / total_weight) if total_weight > 0 else np.nan
    unit = str(field.attrs.get("units", "1"))
    if total_weight <= 0:
        raise ValueError("Spatial weight denominator must be positive")

    valid_values = values[finite]
    valid_weights = weight_array[finite]
    coverage_result = _metric(
        "valid_coverage", coverage, family="coverage", unit="1",
        n=n_valid, coverage=coverage,
    )
    result: list[MetricResult] = [coverage_result]
    required_by_family = {
        "state": [(name, unit) for name in ("weighted_mean", "weighted_median", "weighted_p10", "weighted_p25", "weighted_p75", "weighted_p90")],
        "heterogeneity": [(name, f"{unit} squared" if name == "weighted_variance" else unit) for name in ("weighted_variance", "weighted_standard_deviation", "weighted_mad", "weighted_iqr")],
        "threshold": [("area_over_threshold", "%"), ("weighted_excess_above_threshold", unit)],
        "centroid": [("centroid_latitude", "degrees_north"), ("centroid_longitude", "degrees_east"), ("centroid_dispersion", "degrees")],
        "gradient": [("latitudinal_gradient", f"{unit} per degree"), ("longitudinal_gradient", f"{unit} per degree"), ("thermal_gradient_magnitude", f"{unit} per degree")],
        "texture": [("local_spatial_standard_deviation", unit)],
    }
    if not np.isfinite(coverage) or coverage < minimum_valid_coverage:
        reason = f"Valid coverage {coverage:.3f} is below {minimum_valid_coverage:.3f}"
        for family, names in required_by_family.items():
            result.extend(_not_calculated_family(names, family=family, reason=reason, n=n_valid, coverage=coverage))
        for adjacency_type in ("rook", "queen"):
            result.append(calculate_moran_i(
                values, build_adjacency(values.shape, connectivity=adjacency_type),
                valid_weights=weight_array, minimum_valid_coverage=minimum_valid_coverage,
            ))
        patch_names = (
            ("patch_count", "count"), ("total_threshold_area", "weighted_area"),
            ("largest_patch_area", "weighted_area"), ("largest_patch_fraction", "1"),
            ("patch_density", "per_weighted_area"), ("mean_patch_area", "weighted_area"),
            ("patch_area_standard_deviation", "weighted_area"), ("fragmentation_index", "1"),
            ("edge_cell_fraction", "1"), ("centroid_dispersion", "degrees"),
            ("orientation", "degrees"), ("elongation", "1"),
        )
        result.extend(_not_calculated_family(patch_names, family="patch", reason=reason, n=n_valid, coverage=coverage))
        return result
    if n_valid == 0 or valid_weight <= 0:
        reason = "No finite cells have positive spatial weight"
        for family, names in required_by_family.items():
            result.extend(_not_calculated_family(names, family=family, reason=reason, n=n_valid, coverage=coverage))
        return result

    mean = float(np.sum(valid_values * valid_weights) / valid_weight)
    quantiles = {
        "weighted_p10": weighted_quantile(valid_values, valid_weights, 0.10),
        "weighted_p25": weighted_quantile(valid_values, valid_weights, 0.25),
        "weighted_p75": weighted_quantile(valid_values, valid_weights, 0.75),
        "weighted_p90": weighted_quantile(valid_values, valid_weights, 0.90),
    }
    median = weighted_median(valid_values, valid_weights)
    variance = float(np.sum(valid_weights * (valid_values - mean) ** 2) / valid_weight)
    state_values = {"weighted_mean": mean, "weighted_median": median, **quantiles}
    result.extend(_metric(name, value, family="state", unit=unit, n=n_valid, coverage=coverage) for name, value in state_values.items())
    heterogeneity = {
        "weighted_variance": (variance, f"{unit} squared"),
        "weighted_standard_deviation": (float(np.sqrt(variance)), unit),
        "weighted_mad": (weighted_median(np.abs(valid_values - median), valid_weights), unit),
        "weighted_iqr": (quantiles["weighted_p75"] - quantiles["weighted_p25"], unit),
    }
    result.extend(_metric(name, value, family="heterogeneity", unit=value_unit, n=n_valid, coverage=coverage) for name, (value, value_unit) in heterogeneity.items())

    threshold_mask = finite & (values > threshold)
    threshold_weight = float(weight_array[threshold_mask].sum())
    area_percentage = 100.0 * threshold_weight / valid_weight
    excess = np.where(threshold_mask, values - threshold, 0.0)
    weighted_excess = float(np.sum(weight_array * excess) / valid_weight)
    threshold_metadata = {"threshold": threshold}
    result.extend([
        _metric("area_over_threshold", area_percentage, family="threshold", unit="%", n=n_valid, coverage=coverage, metadata=threshold_metadata),
        _metric("weighted_excess_above_threshold", weighted_excess, family="threshold", unit=unit, n=n_valid, coverage=coverage, metadata=threshold_metadata),
    ])

    lon_grid, lat_grid = np.meshgrid(np.asarray(field[lon_name], dtype=float), np.asarray(field[lat_name], dtype=float))
    centroid_weights = weight_array * excess
    centroid_denominator = float(centroid_weights.sum())
    if centroid_denominator <= 0:
        result.extend(_not_calculated_family(
            (("centroid_latitude", "degrees_north"), ("centroid_longitude", "degrees_east"), ("centroid_dispersion", "degrees")),
            family="centroid", reason=f"No cells exceed threshold {threshold:g}", n=n_valid, coverage=coverage,
        ))
    else:
        centroid_lat = float(np.sum(centroid_weights * lat_grid) / centroid_denominator)
        centroid_lon = float(np.sum(centroid_weights * lon_grid) / centroid_denominator)
        dispersion = float(np.sqrt(np.sum(centroid_weights * ((lat_grid - centroid_lat) ** 2 + (lon_grid - centroid_lon) ** 2)) / centroid_denominator))
        result.extend([
            _metric("centroid_latitude", centroid_lat, family="centroid", unit="degrees_north", n=n_valid, coverage=coverage, metadata=threshold_metadata),
            _metric("centroid_longitude", centroid_lon, family="centroid", unit="degrees_east", n=n_valid, coverage=coverage, metadata=threshold_metadata),
            _metric("centroid_dispersion", dispersion, family="centroid", unit="degrees", n=n_valid, coverage=coverage, metadata=threshold_metadata),
        ])

    lat_gradient = _weighted_gradient(values.ravel(), lat_grid.ravel(), weight_array.ravel())
    lon_gradient = _weighted_gradient(values.ravel(), lon_grid.ravel(), weight_array.ravel())
    for name, value in (("latitudinal_gradient", lat_gradient), ("longitudinal_gradient", lon_gradient)):
        if np.isfinite(value):
            result.append(_metric(name, value, family="gradient", unit=f"{unit} per degree", n=n_valid, coverage=coverage))
        else:
            result.append(not_calculated(name, "Gradient requires at least two distinct coordinates", unit=f"{unit} per degree", family="gradient", n_observations=n_valid, valid_coverage=coverage))

    local_std = _local_standard_deviation(values, local_window_cells)
    local_valid = np.isfinite(local_std) & finite
    local_denominator = float(weight_array[local_valid].sum())
    if local_denominator > 0:
        local_value = float(np.sum(weight_array[local_valid] * local_std[local_valid]) / local_denominator)
        result.append(_metric("local_spatial_standard_deviation", local_value, family="texture", unit=unit, n=int(local_valid.sum()), coverage=coverage))
    else:
        result.append(not_calculated("local_spatial_standard_deviation", "No local neighbourhood has two finite cells", unit=unit, family="texture", n_observations=n_valid, valid_coverage=coverage))

    latitudes = np.asarray(field[lat_name], dtype=float)
    longitudes = np.asarray(field[lon_name], dtype=float)
    if latitudes.size >= 2 and longitudes.size >= 2 and np.unique(latitudes).size >= 2 and np.unique(longitudes).size >= 2:
        with np.errstate(invalid="ignore", divide="ignore"):
            gradient_lat, gradient_lon = np.gradient(values, latitudes, longitudes)
            magnitude = np.sqrt(gradient_lat**2 + gradient_lon**2)
        gradient_valid = np.isfinite(magnitude) & finite
        denominator = float(weight_array[gradient_valid].sum())
        if denominator > 0:
            result.append(_metric("thermal_gradient_magnitude", float(np.sum(weight_array[gradient_valid] * magnitude[gradient_valid]) / denominator), family="gradient", unit=f"{unit} per degree", n=int(gradient_valid.sum()), coverage=coverage))
        else:
            result.append(not_calculated("thermal_gradient_magnitude", "No finite spatial gradients are available", unit=f"{unit} per degree", family="gradient", n_observations=n_valid, valid_coverage=coverage))
    else:
        result.append(not_calculated("thermal_gradient_magnitude", "Thermal gradients require at least two coordinates per axis", unit=f"{unit} per degree", family="gradient", n_observations=n_valid, valid_coverage=coverage))

    row_standardized = moran_weights == "row_standardized"
    if moran_weights not in {"row_standardized", "binary"}:
        raise ValueError("moran_weights must be 'row_standardized' or 'binary'")
    for adjacency_type in ("rook", "queen"):
        result.append(calculate_moran_i(
            values,
            build_adjacency(values.shape, connectivity=adjacency_type, row_standardized=row_standardized),
            valid_weights=weight_array,
            minimum_valid_coverage=minimum_valid_coverage,
        ))

    result.extend(calculate_patch_features(
        threshold_mask,
        latitudes,
        longitudes,
        area_weights=np.where(weight_array > 0, weight_array, np.finfo(float).eps),
        valid_mask=finite & (weight_array > 0),
        connectivity=connectivity,
        minimum_patch_cells=minimum_patch_cells,
    ))
    return result


def build_spatial_feature_table(
    fields: xr.Dataset,
    *,
    variables: Iterable[str] = ("sst", "anomaly", "zscore"),
    threshold: float = 2.0,
    minimum_valid_coverage: float = 0.80,
    connectivity: Connectivity = "queen",
    minimum_patch_cells: int = 4,
    moran_weights: str = "row_standardized",
    local_window_cells: int = 3,
) -> pd.DataFrame:
    """Build the required long-format daily spatial feature table."""
    if "time" not in fields.coords:
        raise ValueError("Spatial feature dataset requires a time coordinate")
    rows: list[dict[str, Any]] = []
    for variable in variables:
        if variable not in fields:
            continue
        for index, date in enumerate(pd.DatetimeIndex(fields.time.values)):
            results = calculate_spatial_features(
                fields[variable].isel(time=index, drop=True), variable=variable,
                threshold=threshold, minimum_valid_coverage=minimum_valid_coverage,
                connectivity=connectivity, minimum_patch_cells=minimum_patch_cells,
                moran_weights=moran_weights, local_window_cells=local_window_cells,
            )
            for metric in results:
                rows.append({
                    "date": date, "variable": variable, "family": metric.family,
                    "metric": metric.metric, "value": metric.value, "unit": metric.unit,
                    "status": metric.status, "reason": metric.reason,
                    "n_observations": metric.n_observations,
                    "valid_coverage": metric.valid_coverage,
                    "threshold": threshold if metric.family in {"threshold", "centroid", "patch"} else np.nan,
                    "connectivity": metric.metadata.get("connectivity", connectivity if metric.family == "patch" else None),
                })
    table = pd.DataFrame(rows, columns=SPATIAL_COLUMNS)
    if not table.empty:
        table["date"] = pd.to_datetime(table["date"])
        table = table.sort_values(["date", "variable", "family", "metric"], ignore_index=True)
    return table
