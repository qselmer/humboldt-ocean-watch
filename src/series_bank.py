"""Daily long-format spatial series bank with explicit result status."""

from __future__ import annotations

from collections.abc import Iterable
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from src.feature_metadata import CENTROID_FEATURES, FEATURES, FeatureDefinition, unit_for
from src.metric_result import MetricResult, not_calculated
from src.quality_control import coordinate_names, coverage_gate, cosine_latitude_weights

SERIES_BANK_COLUMNS = [
    "date", "family", "metric", "value", "unit", "status", "reason",
    "n_observations", "valid_coverage", "window_days", "climatology_method", "data_mode",
]


def _valid_vectors(values: Any, weights: Any) -> tuple[np.ndarray, np.ndarray]:
    data = np.asarray(values, dtype=float).ravel()
    weight = np.asarray(weights, dtype=float).ravel()
    if data.shape != weight.shape:
        raise ValueError("Values and weights must have identical flattened shapes")
    if not np.isfinite(weight).all():
        raise ValueError("Spatial weights must be finite")
    if bool((weight < 0).any()):
        raise ValueError("Spatial weights cannot be negative")
    valid = np.isfinite(data) & (weight > 0)
    return data[valid], weight[valid]


def weighted_mean(values: Any, weights: Any) -> float:
    data, weight = _valid_vectors(values, weights)
    denominator = float(weight.sum())
    return float(np.sum(data * weight) / denominator) if denominator > 0 else np.nan


def weighted_quantile(values: Any, weights: Any, quantile: float) -> float:
    """Return a linearly interpolated weighted quantile using weight midpoints."""
    if not 0.0 <= quantile <= 1.0:
        raise ValueError("Quantile must be between 0 and 1")
    data, weight = _valid_vectors(values, weights)
    denominator = float(weight.sum())
    if denominator <= 0:
        return np.nan
    order = np.argsort(data, kind="stable")
    ordered_data = data[order]
    ordered_weight = weight[order]
    positions = (np.cumsum(ordered_weight) - 0.5 * ordered_weight) / denominator
    return float(np.interp(quantile, positions, ordered_data, left=ordered_data[0], right=ordered_data[-1]))


def weighted_median(values: Any, weights: Any) -> float:
    return weighted_quantile(values, weights, 0.5)


def weighted_mean_result(
    metric: str,
    values: Any,
    weights: Any,
    *,
    unit: str,
    family: str = "state",
) -> MetricResult:
    data, weight = _valid_vectors(values, weights)
    denominator = float(weight.sum())
    coverage = float(denominator / np.asarray(weights, dtype=float).sum()) if np.asarray(weights, dtype=float).sum() > 0 else np.nan
    if denominator <= 0:
        return not_calculated(
            metric, "Valid spatial denominator is zero", unit=unit, family=family,
            n_observations=len(data), valid_coverage=coverage,
        )
    return MetricResult(
        metric=metric,
        value=float(np.sum(data * weight) / denominator),
        n_observations=len(data),
        valid_coverage=coverage,
        unit=unit,
        family=family,
    )


def _gradient(values: np.ndarray, coordinate: np.ndarray, weights: np.ndarray) -> float:
    valid = np.isfinite(values) & np.isfinite(coordinate) & np.isfinite(weights) & (weights > 0)
    if valid.sum() < 2 or np.unique(coordinate[valid]).size < 2:
        return np.nan
    x = coordinate[valid]
    y = values[valid]
    w = weights[valid]
    x_mean = np.sum(w * x) / np.sum(w)
    y_mean = np.sum(w * y) / np.sum(w)
    denominator = np.sum(w * (x - x_mean) ** 2)
    return float(np.sum(w * (x - x_mean) * (y - y_mean)) / denominator) if denominator > 0 else np.nan


def _base_result(
    feature: FeatureDefinition,
    variable: str,
    value: float,
    *,
    variable_unit: str,
    n_observations: int,
    coverage: float,
    status: str = "valid",
    reason: str | None = None,
) -> MetricResult:
    return MetricResult(
        metric=feature.metric_name(variable),
        value=value,
        status=status,  # type: ignore[arg-type]
        reason=reason,
        n_observations=n_observations,
        valid_coverage=coverage,
        unit=unit_for(variable_unit, feature.unit_kind),
        family=feature.family,
        metadata={"variable": variable, "statistic": feature.name},
    )


def calculate_daily_features(
    field: xr.DataArray,
    *,
    variable: str,
    weights: xr.DataArray | None = None,
    minimum_valid_coverage: float = 0.80,
    warm_anomaly_threshold_c: float = 2.0,
) -> list[MetricResult]:
    """Calculate one day's spatial feature set for one explicit variable."""
    lat, lon = coordinate_names(field)
    if "time" in field.dims or set(field.dims) != {lat, lon}:
        raise ValueError("Daily feature input must be exactly two-dimensional latitude/longitude")
    field = field.transpose(lat, lon)
    if not np.issubdtype(field.dtype, np.number):
        raise TypeError("Daily feature input must be numeric")
    spatial_weights = weights if weights is not None else cosine_latitude_weights(field.expand_dims(time=[0]))
    if "time" in spatial_weights.dims:
        spatial_weights = spatial_weights.isel(time=0, drop=True)
    spatial_weights = spatial_weights.broadcast_like(field)
    raw_weights = np.asarray(spatial_weights.values, dtype=float)
    if not np.isfinite(raw_weights).all() or bool((raw_weights < 0).any()):
        raise ValueError("Spatial weights must be finite and non-negative")
    values = np.asarray(field.values, dtype=float)
    finite = np.isfinite(values)
    n_valid = int(finite.sum())
    total_cells = values.size
    total_weight = float(raw_weights.sum())
    valid_weight = float(raw_weights[finite].sum())
    coverage = valid_weight / total_weight if total_weight > 0 else np.nan
    cell_fraction = n_valid / total_cells if total_cells else np.nan
    variable_unit = str(field.attrs.get("units", "1"))
    data, weight = _valid_vectors(values, raw_weights)

    definitions = {feature.name: feature for feature in FEATURES}
    computed: dict[str, float] = {
        "weighted_mean": weighted_mean(data, weight),
        "weighted_median": weighted_median(data, weight),
        "spatial_minimum": float(np.min(data)) if data.size else np.nan,
        "spatial_maximum": float(np.max(data)) if data.size else np.nan,
        "spatial_p10": weighted_quantile(data, weight, 0.10),
        "spatial_p25": weighted_quantile(data, weight, 0.25),
        "spatial_p75": weighted_quantile(data, weight, 0.75),
        "spatial_p90": weighted_quantile(data, weight, 0.90),
        "valid_cell_fraction": cell_fraction,
        "weighted_valid_coverage": coverage,
        "valid_observation_count": float(n_valid),
        "missing_fraction": 1.0 - cell_fraction if np.isfinite(cell_fraction) else np.nan,
        "constant_field_flag": float(data.size > 0 and np.all(data == data[0])),
    }
    mean = computed["weighted_mean"]
    computed["weighted_standard_deviation"] = (
        float(np.sqrt(np.sum(weight * (data - mean) ** 2) / np.sum(weight))) if weight.sum() > 0 else np.nan
    )
    median = computed["weighted_median"]
    computed["weighted_mad"] = weighted_median(np.abs(data - median), weight)
    computed["weighted_iqr"] = computed["spatial_p75"] - computed["spatial_p25"]
    computed["p90_minus_p10"] = computed["spatial_p90"] - computed["spatial_p10"]
    lon_grid, lat_grid = np.meshgrid(field[lon].values, field[lat].values)
    computed["latitudinal_gradient"] = _gradient(values.ravel(), lat_grid.ravel(), raw_weights.ravel())
    computed["longitudinal_gradient"] = _gradient(values.ravel(), lon_grid.ravel(), raw_weights.ravel())

    results: list[MetricResult] = []
    gate_families = {"state", "heterogeneity", "gradient"}
    for feature in FEATURES:
        unit = unit_for(variable_unit, feature.unit_kind)
        gate = coverage_gate(
            feature.metric_name(variable), coverage, minimum_valid_coverage,
            unit=unit, family=feature.family, n_observations=n_valid,
        ) if feature.family in gate_families else None
        results.append(gate or _base_result(
            feature, variable, computed[feature.name], variable_unit=variable_unit,
            n_observations=n_valid, coverage=coverage,
        ))

    if variable == "anomaly":
        centroid_gate = coverage_gate(
            "anomaly.warm_centroid_latitude", coverage, minimum_valid_coverage,
            unit="degrees_north", family="centroid", n_observations=n_valid,
        )
        threshold_mask = finite & (values >= warm_anomaly_threshold_c)
        if centroid_gate is not None:
            for feature in CENTROID_FEATURES:
                results.append(not_calculated(
                    feature.metric_name(variable), centroid_gate.reason or "Insufficient coverage",
                    unit=unit_for(variable_unit, feature.unit_kind), family="centroid",
                    n_observations=n_valid, valid_coverage=coverage,
                    metadata={"variable": variable, "threshold_c": warm_anomaly_threshold_c},
                ))
        elif not threshold_mask.any():
            for feature in CENTROID_FEATURES:
                results.append(not_calculated(
                    feature.metric_name(variable),
                    f"No valid cells meet the warm-anomaly threshold of {warm_anomaly_threshold_c:g} °C",
                    unit=unit_for(variable_unit, feature.unit_kind), family="centroid",
                    n_observations=n_valid, valid_coverage=coverage,
                    metadata={"variable": variable, "threshold_c": warm_anomaly_threshold_c},
                ))
        else:
            centroid_weights = raw_weights * np.where(threshold_mask, np.maximum(values, 0.0), 0.0)
            denominator = centroid_weights.sum()
            centroid_values = {
                "warm_centroid_latitude": float(np.sum(centroid_weights * lat_grid) / denominator),
                "warm_centroid_longitude": float(np.sum(centroid_weights * lon_grid) / denominator),
            }
            for feature in CENTROID_FEATURES:
                results.append(_base_result(
                    feature, variable, centroid_values[feature.name], variable_unit=variable_unit,
                    n_observations=n_valid, coverage=coverage,
                ))
    return results


def _row(
    date: pd.Timestamp,
    result: MetricResult,
    *,
    climatology_method: str | None,
    data_mode: str,
) -> dict[str, Any]:
    return {
        "date": date,
        "family": result.family,
        "metric": result.metric,
        "value": result.value,
        "unit": result.unit,
        "status": result.status,
        "reason": result.reason,
        "n_observations": result.n_observations,
        "valid_coverage": result.valid_coverage,
        "window_days": result.window_days,
        "climatology_method": climatology_method,
        "data_mode": data_mode,
    }


def build_series_bank(
    fields: xr.Dataset,
    *,
    variables: Iterable[str] = ("sst", "anomaly", "zscore"),
    weights: xr.DataArray | None = None,
    minimum_valid_coverage: float = 0.80,
    warm_anomaly_threshold_c: float = 2.0,
    climatology_method: str | None = None,
    data_mode: str = "unknown",
) -> pd.DataFrame:
    """Transform daily spatial cubes into a variable-qualified long bank."""
    if "time" not in fields.coords:
        raise ValueError("Series-bank fields require a time coordinate")
    rows: list[dict[str, Any]] = []
    available = [variable for variable in variables if variable in fields]
    if not available:
        raise ValueError("None of the requested series-bank variables are available")
    for variable in available:
        field = fields[variable]
        variable_weights = weights
        for index, timestamp in enumerate(pd.DatetimeIndex(fields.time.values)):
            daily_weights = variable_weights
            if daily_weights is not None and "time" in daily_weights.dims:
                daily_weights = daily_weights.isel(time=index, drop=True)
            results = calculate_daily_features(
                field.isel(time=index, drop=True),
                variable=variable,
                weights=daily_weights,
                minimum_valid_coverage=minimum_valid_coverage,
                warm_anomaly_threshold_c=warm_anomaly_threshold_c,
            )
            rows.extend(
                _row(timestamp, result, climatology_method=climatology_method, data_mode=data_mode)
                for result in results
            )
    bank = pd.DataFrame(rows, columns=SERIES_BANK_COLUMNS)
    bank["date"] = pd.to_datetime(bank["date"])
    return bank.sort_values(["date", "family", "metric"], ignore_index=True)
