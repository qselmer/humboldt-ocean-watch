"""Patch-level geometry, intensity, and daily regional summaries.

Areas and perimeters use :mod:`src.grid_geometry`. Orientation is measured
counter-clockwise from local east and normalized to the axial range
[-90, 90) degrees. Major and minor axis lengths are four times the square
root of the corresponding area-weighted covariance eigenvalue (a two-sigma
full axis). Centroid dispersion is the area-weighted RMS cell-centre distance
from the patch centroid.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from src.grid_geometry import (
    GridGeometry,
    edge_cell_mask,
    local_kilometre_coordinates,
    patch_perimeter_km,
    unwrap_longitudes,
    weighted_geographic_centroid,
)
from src.metric_result import MetricResult, not_calculated
from src.patch_detection import PatchDetectionConfig, PatchDetectionResult
from src.series_bank import weighted_mean, weighted_median, weighted_quantile

PATCH_COLUMNS = [
    "date", "patch_id", "source_variable", "threshold_type", "threshold_value",
    "direction", "connectivity", "cell_count", "area_km2",
    "area_fraction_of_valid_region", "area_fraction_of_threshold_total",
    "equivalent_radius_km", "centroid_latitude", "centroid_longitude",
    "minimum_latitude", "maximum_latitude", "minimum_longitude",
    "maximum_longitude", "mean_source_value", "median_source_value",
    "minimum_source_value", "maximum_source_value", "p10", "p90",
    "mean_exceedance", "maximum_exceedance",
    "cumulative_exceedance_km2_units", "perimeter_km", "compactness",
    "perimeter_area_ratio", "orientation_degrees", "elongation",
    "major_axis_length_km", "minor_axis_length_km", "centroid_dispersion_km",
    "edge_cell_fraction", "touches_north_boundary", "touches_south_boundary",
    "touches_east_boundary", "touches_west_boundary", "valid_coverage",
    "climatology_method", "data_mode", "status", "reason",
]

DAILY_PATCH_SUMMARY_COLUMNS = [
    "date", "patch_count", "total_patch_area_km2", "threshold_area_fraction",
    "largest_patch_area_km2", "largest_patch_fraction", "mean_patch_area_km2",
    "median_patch_area_km2", "patch_area_standard_deviation_km2",
    "patch_density_per_10000_km2", "fragmentation_index",
    "dominant_patch_fraction", "area_weighted_mean_patch_intensity",
    "maximum_patch_intensity", "area_weighted_centroid_latitude",
    "area_weighted_centroid_longitude", "patch_centroid_dispersion_km",
    "valid_coverage", "status", "reason",
]


def compactness_result(
    area_km2: float,
    perimeter_km: float,
    *,
    cell_count: int,
    valid_coverage: float,
) -> MetricResult:
    """Return ``4*pi*area/perimeter**2`` or an explicit undefined result."""
    if not np.isfinite(perimeter_km) or perimeter_km <= 0:
        return not_calculated(
            "compactness",
            "Compactness requires a finite positive perimeter",
            unit="1",
            family="patch_shape",
            n_observations=cell_count,
            valid_coverage=valid_coverage,
        )
    if not np.isfinite(area_km2) or area_km2 <= 0:
        return not_calculated(
            "compactness",
            "Compactness requires a finite positive area",
            unit="1",
            family="patch_shape",
            n_observations=cell_count,
            valid_coverage=valid_coverage,
        )
    return MetricResult(
        metric="compactness",
        value=float(4.0 * np.pi * area_km2 / perimeter_km**2),
        unit="1",
        family="patch_shape",
        n_observations=cell_count,
        valid_coverage=valid_coverage,
    )


def _mask_array(
    mask: xr.DataArray | np.ndarray | None,
    template: xr.DataArray,
) -> np.ndarray:
    if mask is None:
        return np.asarray(template.notnull().values, dtype=bool)
    if isinstance(mask, xr.DataArray):
        try:
            aligned = mask.sel(
                latitude=template.latitude,
                longitude=template.longitude,
            ).transpose("latitude", "longitude")
        except (KeyError, ValueError) as exc:
            raise ValueError("Ocean-domain mask is incompatible with patch labels") from exc
        values = np.asarray(aligned.values)
    else:
        values = np.asarray(mask)
    if values.shape != template.shape:
        raise ValueError("Ocean-domain mask shape is incompatible with patch labels")
    return values.astype(bool)


def _shape_metrics(
    cells: np.ndarray,
    geometry: GridGeometry,
    *,
    centroid_latitude: float,
    centroid_longitude: float,
    valid_coverage: float,
) -> tuple[dict[str, float], list[str]]:
    areas = np.asarray(geometry.area_km2.values, dtype=float)
    weights = areas[cells]
    denominator = float(weights.sum())
    x, y = local_kilometre_coordinates(
        geometry,
        origin_latitude=centroid_latitude,
        origin_longitude=centroid_longitude,
    )
    selected_x = x[cells]
    selected_y = y[cells]
    centred_x = selected_x - float(np.sum(weights * selected_x) / denominator)
    centred_y = selected_y - float(np.sum(weights * selected_y) / denominator)
    dispersion = float(
        np.sqrt(np.sum(weights * (selected_x**2 + selected_y**2)) / denominator)
    )
    covariance = np.asarray(
        [
            [np.sum(weights * centred_x * centred_x), np.sum(weights * centred_x * centred_y)],
            [np.sum(weights * centred_x * centred_y), np.sum(weights * centred_y * centred_y)],
        ],
        dtype=float,
    ) / denominator
    eigenvalues, eigenvectors = np.linalg.eigh(covariance)
    eigenvalues = np.maximum(eigenvalues, 0.0)
    minor_variance, major_variance = float(eigenvalues[0]), float(eigenvalues[-1])
    major_axis = float(4.0 * np.sqrt(major_variance))
    minor_axis = float(4.0 * np.sqrt(minor_variance))
    reasons: list[str] = []
    if major_variance <= np.finfo(float).eps:
        orientation = np.nan
        elongation = np.nan
        reasons.append("Patch cell centres have no two-dimensional spatial variance")
    else:
        major_vector = eigenvectors[:, -1]
        orientation = float(
            (np.degrees(np.arctan2(major_vector[1], major_vector[0])) + 90.0) % 180.0
            - 90.0
        )
        if minor_variance <= np.finfo(float).eps:
            elongation = np.nan
            reasons.append("Patch minor-axis variance is zero; elongation is undefined")
        else:
            elongation = float(np.sqrt(major_variance / minor_variance))
    perimeter = patch_perimeter_km(cells, geometry)
    area = denominator
    compactness = compactness_result(
        area,
        perimeter,
        cell_count=int(cells.sum()),
        valid_coverage=valid_coverage,
    )
    if compactness.status != "valid":
        reasons.append(compactness.reason or "Patch compactness is undefined")
    return {
        "perimeter_km": perimeter,
        "compactness": float(compactness.value),
        "perimeter_area_ratio": float(perimeter / area) if area > 0 else np.nan,
        "orientation_degrees": orientation,
        "elongation": elongation,
        "major_axis_length_km": major_axis,
        "minor_axis_length_km": minor_axis,
        "centroid_dispersion_km": dispersion,
        "edge_cell_fraction": float(edge_cell_mask(cells).sum() / cells.sum()),
    }, reasons


def _boundary_flags(cells: np.ndarray, geometry: GridGeometry) -> dict[str, bool]:
    latitude = np.asarray(geometry.area_km2[geometry.latitude_name].values, dtype=float)
    longitude = unwrap_longitudes(
        np.asarray(geometry.area_km2[geometry.longitude_name].values, dtype=float)
    )
    north_index = int(np.argmax(latitude))
    south_index = int(np.argmin(latitude))
    east_index = int(np.argmax(longitude))
    west_index = int(np.argmin(longitude))
    return {
        "touches_north_boundary": bool(cells[north_index, :].any()),
        "touches_south_boundary": bool(cells[south_index, :].any()),
        "touches_east_boundary": bool(cells[:, east_index].any()),
        "touches_west_boundary": bool(cells[:, west_index].any()),
    }


def characterize_daily_patches(
    detection: PatchDetectionResult,
    *,
    date: Any,
    config: PatchDetectionConfig,
    climatology_method: str | None,
    data_mode: str,
    ocean_domain_mask: xr.DataArray | np.ndarray | None = None,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    """Calculate all patch rows and one daily regional summary."""
    labels = np.asarray(detection.patch_id.values, dtype=np.int32)
    source = np.asarray(detection.validated.field.values, dtype=float)
    threshold = np.asarray(detection.validated.threshold.values, dtype=float)
    exceedance = np.asarray(detection.exceedance.values, dtype=float)
    valid = np.asarray(detection.valid_ocean_mask.values, dtype=bool)
    geometry = detection.validated.geometry
    areas = np.asarray(detection.validated.cell_area_km2.values, dtype=float)
    domain = _mask_array(ocean_domain_mask, detection.patch_id)
    domain_area = float(areas[domain].sum())
    valid_area = float(areas[valid].sum())
    if domain_area <= 0:
        raise ValueError("Ocean-domain area denominator must be positive")
    valid_coverage = float(valid_area / domain_area)
    retained = labels > 0
    total_patch_area = float(areas[retained].sum())
    latitude = np.asarray(detection.patch_id.latitude.values, dtype=float)
    longitude = np.asarray(detection.patch_id.longitude.values, dtype=float)
    longitude_grid, latitude_grid = np.meshgrid(longitude, latitude)
    date_value = pd.Timestamp(date).normalize()
    rows: list[dict[str, Any]] = []

    for patch_identifier in range(1, detection.retained_patch_count + 1):
        cells = labels == patch_identifier
        cell_count = int(cells.sum())
        patch_areas = areas[cells]
        area = float(patch_areas.sum())
        source_values = source[cells]
        threshold_values = threshold[cells]
        excess_values = exceedance[cells]
        if not np.isfinite(excess_values).all() or bool((excess_values < -1e-10).any()):
            raise ValueError("Patch exceedance must be finite and non-negative inside retained cells")
        excess_values = np.maximum(excess_values, 0.0)
        centroid_latitude, centroid_longitude = weighted_geographic_centroid(cells, geometry)
        shape, shape_reasons = _shape_metrics(
            cells,
            geometry,
            centroid_latitude=centroid_latitude,
            centroid_longitude=centroid_longitude,
            valid_coverage=valid_coverage,
        )
        reasons = list(shape_reasons)
        row = {
            "date": date_value,
            "patch_id": patch_identifier,
            "source_variable": config.source_variable,
            "threshold_type": config.threshold_type,
            "threshold_value": weighted_mean(threshold_values, patch_areas),
            "direction": config.direction,
            "connectivity": config.connectivity,
            "cell_count": cell_count,
            "area_km2": area,
            "area_fraction_of_valid_region": area / valid_area if valid_area > 0 else np.nan,
            "area_fraction_of_threshold_total": area / total_patch_area if total_patch_area > 0 else np.nan,
            "equivalent_radius_km": float(np.sqrt(area / np.pi)),
            "centroid_latitude": centroid_latitude,
            "centroid_longitude": centroid_longitude,
            "minimum_latitude": float(latitude_grid[cells].min()),
            "maximum_latitude": float(latitude_grid[cells].max()),
            "minimum_longitude": float(longitude_grid[cells].min()),
            "maximum_longitude": float(longitude_grid[cells].max()),
            "mean_source_value": weighted_mean(source_values, patch_areas),
            "median_source_value": weighted_median(source_values, patch_areas),
            "minimum_source_value": float(source_values.min()),
            "maximum_source_value": float(source_values.max()),
            "p10": weighted_quantile(source_values, patch_areas, 0.10),
            "p90": weighted_quantile(source_values, patch_areas, 0.90),
            "mean_exceedance": weighted_mean(excess_values, patch_areas),
            "maximum_exceedance": float(excess_values.max()),
            "cumulative_exceedance_km2_units": float(np.sum(excess_values * patch_areas)),
            **shape,
            **_boundary_flags(cells, geometry),
            "valid_coverage": valid_coverage,
            "climatology_method": climatology_method,
            "data_mode": data_mode,
            "status": "warning" if reasons else "valid",
            "reason": "; ".join(reasons) if reasons else None,
        }
        rows.append(row)
    patches = pd.DataFrame(rows, columns=PATCH_COLUMNS)
    summary = summarize_daily_patches(
        patches,
        labels=labels,
        valid_mask=valid,
        ocean_domain_mask=domain,
        geometry=geometry,
        date=date_value,
        no_patch_reason=detection.reason,
    )
    return patches, summary


def summarize_daily_patches(
    patches: pd.DataFrame,
    *,
    labels: np.ndarray,
    valid_mask: np.ndarray,
    ocean_domain_mask: np.ndarray,
    geometry: GridGeometry,
    date: Any,
    no_patch_reason: str | None = None,
) -> dict[str, Any]:
    """Summarize retained daily patches using spherical area denominators.

    ``largest_patch_fraction`` uses valid regional area, while
    ``dominant_patch_fraction`` uses total retained patch area. Consequently,
    fragmentation is ``1 - dominant_patch_fraction``.
    """
    label_values = np.asarray(labels, dtype=np.int32)
    valid = np.asarray(valid_mask, dtype=bool)
    domain = np.asarray(ocean_domain_mask, dtype=bool)
    areas = np.asarray(geometry.area_km2.values, dtype=float)
    if label_values.shape != areas.shape or valid.shape != areas.shape or domain.shape != areas.shape:
        raise ValueError("Daily patch summary masks are incompatible with grid geometry")
    domain_area = float(areas[domain].sum())
    valid_area = float(areas[valid].sum())
    if domain_area <= 0:
        raise ValueError("Ocean-domain area denominator must be positive")
    coverage = float(valid_area / domain_area)
    patch_count = len(patches)
    base: dict[str, Any] = {
        "date": pd.Timestamp(date).normalize(),
        "patch_count": patch_count,
        "valid_coverage": coverage,
    }
    if patch_count == 0:
        reason = no_patch_reason or "No patches exceeded the configured criteria"
        result = {
            **base,
            "total_patch_area_km2": 0.0,
            "threshold_area_fraction": 0.0,
            "largest_patch_area_km2": 0.0,
            "largest_patch_fraction": 0.0,
            "mean_patch_area_km2": np.nan,
            "median_patch_area_km2": np.nan,
            "patch_area_standard_deviation_km2": np.nan,
            "patch_density_per_10000_km2": 0.0,
            "fragmentation_index": np.nan,
            "dominant_patch_fraction": np.nan,
            "area_weighted_mean_patch_intensity": np.nan,
            "maximum_patch_intensity": np.nan,
            "area_weighted_centroid_latitude": np.nan,
            "area_weighted_centroid_longitude": np.nan,
            "patch_centroid_dispersion_km": np.nan,
            "status": "not_calculated",
            "reason": reason,
        }
        return {column: result[column] for column in DAILY_PATCH_SUMMARY_COLUMNS}
    patch_areas = patches.area_km2.to_numpy(dtype=float)
    total = float(patch_areas.sum())
    largest = float(patch_areas.max())
    dominant = largest / total
    retained = label_values > 0
    centroid_latitude, centroid_longitude = weighted_geographic_centroid(retained, geometry)
    centroid_x, centroid_y = local_kilometre_coordinates(
        geometry,
        origin_latitude=centroid_latitude,
        origin_longitude=centroid_longitude,
    )
    centroid_positions_x: list[float] = []
    centroid_positions_y: list[float] = []
    for patch_identifier in patches.patch_id.astype(int):
        cells = label_values == patch_identifier
        weights = areas[cells]
        centroid_positions_x.append(float(np.sum(weights * centroid_x[cells]) / weights.sum()))
        centroid_positions_y.append(float(np.sum(weights * centroid_y[cells]) / weights.sum()))
    patch_centroid_dispersion = float(
        np.sqrt(
            np.sum(
                patch_areas
                * (
                    np.asarray(centroid_positions_x) ** 2
                    + np.asarray(centroid_positions_y) ** 2
                )
            )
            / total
        )
    )
    warnings = patches.loc[patches.status == "warning", "reason"].dropna().astype(str).unique()
    result = {
        **base,
        "total_patch_area_km2": total,
        "threshold_area_fraction": total / valid_area if valid_area > 0 else np.nan,
        "largest_patch_area_km2": largest,
        "largest_patch_fraction": largest / valid_area if valid_area > 0 else np.nan,
        "mean_patch_area_km2": float(patch_areas.mean()),
        "median_patch_area_km2": float(np.median(patch_areas)),
        "patch_area_standard_deviation_km2": float(patch_areas.std()),
        "patch_density_per_10000_km2": float(patch_count / valid_area * 10000.0),
        "fragmentation_index": float(1.0 - dominant),
        "dominant_patch_fraction": dominant,
        "area_weighted_mean_patch_intensity": weighted_mean(
            patches.mean_exceedance.to_numpy(dtype=float), patch_areas
        ),
        "maximum_patch_intensity": float(patches.maximum_exceedance.max()),
        "area_weighted_centroid_latitude": centroid_latitude,
        "area_weighted_centroid_longitude": centroid_longitude,
        "patch_centroid_dispersion_km": patch_centroid_dispersion,
        "status": "warning" if warnings.size else "valid",
        "reason": "; ".join(warnings) if warnings.size else None,
    }
    return {column: result[column] for column in DAILY_PATCH_SUMMARY_COLUMNS}


def daily_summary_frame(rows: list[dict[str, Any]]) -> pd.DataFrame:
    """Return a stable daily summary schema, including zero-patch dates."""
    return pd.DataFrame(rows, columns=DAILY_PATCH_SUMMARY_COLUMNS)
