"""Explicit structural validation and quality-control reporting."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Literal

import numpy as np
import pandas as pd
import xarray as xr

from src.export_utils import export_json
from src.metric_result import MetricResult, not_calculated

DuplicatePolicy = Literal["error", "first", "last", "mean"]
CELSIUS_UNITS = frozenset({"c", "°c", "degc", "celsius", "degree_celsius", "degrees_celsius"})
KELVIN_UNITS = frozenset({"k", "kelvin", "degree_kelvin", "degrees_kelvin"})
DIMENSIONLESS_UNITS = frozenset({"1", "dimensionless", "standardized", "zscore", "z-score"})


@dataclass
class QCReport:
    source_file: str | None
    data_mode: str | None
    variable: str
    units: str
    start_date: str | None
    end_date: str | None
    number_of_dates: int
    duplicated_dates: list[str]
    temporal_interval_summary: dict[str, Any]
    expected_frequency: str
    spatial_dimensions: dict[str, int]
    spatial_bounds: dict[str, list[float]]
    valid_data_fraction: float
    weighted_spatial_coverage: float
    constant_field_dates: list[str]
    all_nan_dates: list[str]
    warnings: list[str] = field(default_factory=list)
    errors: list[str] = field(default_factory=list)
    overall_status: str = "valid"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def coordinate_names(data: xr.DataArray | xr.Dataset) -> tuple[str, str]:
    lat = "latitude" if "latitude" in data.coords else "lat" if "lat" in data.coords else None
    lon = "longitude" if "longitude" in data.coords else "lon" if "lon" in data.coords else None
    if lat is None:
        raise ValueError("Input is missing a latitude or lat coordinate")
    if lon is None:
        raise ValueError("Input is missing a longitude or lon coordinate")
    return lat, lon


def recognize_units(units: str | None) -> Literal["celsius", "kelvin", "dimensionless"]:
    normalized = str(units or "").strip().lower()
    if normalized in CELSIUS_UNITS:
        return "celsius"
    if normalized in KELVIN_UNITS:
        return "kelvin"
    if normalized in DIMENSIONLESS_UNITS:
        return "dimensionless"
    raise ValueError(f"Unknown or unsupported units: {units!r}")


def _resolve_duplicates(data: xr.DataArray, policy: DuplicatePolicy) -> xr.DataArray:
    dates = pd.DatetimeIndex(data.time.values)
    if policy == "error":
        raise ValueError("Duplicate dates detected; configure first, last, or mean to resolve them")
    if policy in {"first", "last"}:
        keep = ~dates.duplicated(keep=policy)
        return data.isel(time=np.flatnonzero(keep))
    if policy == "mean":
        attrs = dict(data.attrs)
        result = data.groupby("time").mean("time", skipna=True)
        result.attrs = attrs
        return result
    raise ValueError(f"Unsupported duplicate policy: {policy!r}")


def validate_spatial_auxiliary(
    data: xr.DataArray,
    auxiliary: xr.DataArray | None,
    *,
    name: str,
    strictly_positive: bool = False,
) -> xr.DataArray | None:
    if auxiliary is None:
        return None
    lat, lon = coordinate_names(data)
    if not set(auxiliary.dims).issubset({lat, lon, "time"}):
        raise ValueError(f"{name} has incompatible dimensions: {auxiliary.dims}")
    for dim in auxiliary.dims:
        if auxiliary.sizes[dim] != data.sizes[dim]:
            raise ValueError(f"{name} size for {dim} is incompatible with the data")
    try:
        broadcast = auxiliary.broadcast_like(data)
    except ValueError as exc:
        raise ValueError(f"{name} cannot be broadcast to the data dimensions") from exc
    values = np.asarray(broadcast.values)
    if not np.isfinite(values).all():
        raise ValueError(f"{name} must contain only finite values")
    if strictly_positive and bool((values <= 0).any()):
        raise ValueError(f"{name} must contain only positive values")
    if not strictly_positive and bool((values < 0).any()):
        raise ValueError(f"{name} cannot contain negative values")
    return broadcast


def cosine_latitude_weights(data: xr.DataArray) -> xr.DataArray:
    lat, lon = coordinate_names(data)
    weights = np.cos(np.deg2rad(data[lat])).rename("spatial_weight")
    spatial_template = data.isel(time=0, drop=True) if "time" in data.dims else data
    weights = weights.broadcast_like(spatial_template)
    return validate_spatial_auxiliary(spatial_template, weights, name="spatial weights")  # type: ignore[return-value]


def validate_input(
    value: xr.Dataset | xr.DataArray,
    *,
    variable: str | None = None,
    expected_frequency: str = "1D",
    duplicate_policy: DuplicatePolicy = "error",
    maximum_gap_days: int = 2,
    allow_irregular_intervals: bool = True,
    fail_on_unknown_units: bool = True,
    minimum_temporal_observations: int = 10,
    minimum_valid_coverage: float = 0.80,
    mask: xr.DataArray | None = None,
    weights: xr.DataArray | None = None,
    cell_areas: xr.DataArray | None = None,
    source_file: str | Path | None = None,
    data_mode: str | None = None,
) -> tuple[xr.DataArray, xr.DataArray, QCReport]:
    """Validate and explicitly resolve recoverable temporal input problems."""
    if isinstance(value, xr.Dataset):
        if variable is None:
            raise ValueError("A variable name is required when validating an xarray Dataset")
        if variable not in value:
            raise ValueError(f"Dataset does not contain variable {variable!r}")
        data = value[variable]
    elif isinstance(value, xr.DataArray):
        data = value
        variable = variable or data.name or "unnamed"
    else:
        raise TypeError("Input must be an xarray Dataset or DataArray")
    if "time" not in data.coords or "time" not in data.dims:
        raise ValueError("Input must contain a time coordinate and dimension")
    lat, lon = coordinate_names(data)
    expected_dims = {"time", lat, lon}
    if set(data.dims) != expected_dims:
        raise ValueError(f"Field dimensions must be exactly time, {lat}, {lon}; got {data.dims}")
    if not np.issubdtype(data.dtype, np.number):
        raise TypeError(f"Field {variable!r} must be numeric")
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(data.time.values, errors="raise"))
    except (ValueError, TypeError) as exc:
        raise ValueError("Time coordinate contains unreadable dates") from exc
    if dates.empty:
        raise ValueError("Time coordinate contains no dates")

    warnings: list[str] = []
    errors: list[str] = []
    duplicated = sorted({stamp.isoformat() for stamp in dates[dates.duplicated(keep=False)]})
    if not dates.is_monotonic_increasing:
        data = data.sortby("time")
        dates = pd.DatetimeIndex(data.time.values)
        warnings.append("Dates were unsorted and have been sorted chronologically")
    if duplicated:
        data = _resolve_duplicates(data, duplicate_policy)
        dates = pd.DatetimeIndex(data.time.values)
        warnings.append(f"Resolved {len(duplicated)} duplicated date(s) using policy {duplicate_policy}")

    try:
        unit_kind = recognize_units(data.attrs.get("units"))
    except ValueError as exc:
        if fail_on_unknown_units:
            raise
        unit_kind = "dimensionless"
        warnings.append(str(exc))

    supplied_weights = validate_spatial_auxiliary(data, weights, name="spatial weights")
    validate_spatial_auxiliary(data, cell_areas, name="cell areas", strictly_positive=True)
    if mask is not None:
        compatible_mask = validate_spatial_auxiliary(data, mask.astype(float), name="mask")
        assert compatible_mask is not None
        data = data.where(compatible_mask.astype(bool))
    spatial_weights = supplied_weights if supplied_weights is not None else cosine_latitude_weights(data).broadcast_like(data)

    raw = np.asarray(data.values)
    finite = np.isfinite(raw)
    if np.isinf(raw).any():
        errors.append("Field contains infinite values")
    per_date_finite = finite.reshape(data.sizes["time"], -1)
    all_nan_indices = np.flatnonzero(per_date_finite.sum(axis=1) == 0)
    all_nan_dates = [dates[index].isoformat() for index in all_nan_indices]
    if len(all_nan_dates) == len(dates):
        errors.append("Field contains no finite observations")
    constant_dates: list[str] = []
    for index, row in enumerate(per_date_finite):
        values = raw[index].ravel()[row]
        if values.size and np.all(values == values[0]):
            constant_dates.append(dates[index].isoformat())

    intervals = np.diff(dates.values).astype("timedelta64[ns]").astype(np.int64) / 86_400_000_000_000
    try:
        expected_range = pd.date_range("2000-01-01", periods=2, freq=expected_frequency)
        expected_days = (expected_range[1] - expected_range[0]).total_seconds() / 86400
    except (ValueError, TypeError) as exc:
        raise ValueError(f"Unsupported expected temporal frequency: {expected_frequency!r}") from exc
    regular = bool(intervals.size == 0 or np.allclose(intervals, expected_days))
    if not regular:
        message = f"Temporal intervals are irregular relative to expected frequency {expected_frequency}"
        if allow_irregular_intervals:
            warnings.append(message)
        else:
            errors.append(message)
    if intervals.size and float(np.max(intervals)) > maximum_gap_days:
        warnings.append(f"Maximum temporal gap exceeds {maximum_gap_days} days")
    if len(dates) < minimum_temporal_observations:
        warnings.append(
            f"Only {len(dates)} observations are available; minimum is {minimum_temporal_observations}"
        )

    valid_fraction = float(finite.sum() / finite.size)
    weight_values = np.asarray(spatial_weights.values, dtype=float)
    total_weight = float(weight_values.sum())
    if total_weight <= 0:
        weighted_coverage = np.nan
        errors.append("Spatial weights have a zero denominator")
    else:
        weighted_coverage = float(weight_values[finite].sum() / total_weight)
    if np.isfinite(weighted_coverage) and weighted_coverage < minimum_valid_coverage:
        warnings.append(
            f"Weighted valid coverage {weighted_coverage:.3f} is below {minimum_valid_coverage:.3f}"
        )

    interval_summary = {
        "regular": regular,
        "minimum_days": float(np.min(intervals)) if intervals.size else None,
        "median_days": float(np.median(intervals)) if intervals.size else None,
        "maximum_days": float(np.max(intervals)) if intervals.size else None,
    }
    report = QCReport(
        source_file=str(source_file) if source_file is not None else None,
        data_mode=data_mode,
        variable=str(variable),
        units=str(data.attrs.get("units")),
        start_date=dates[0].isoformat(),
        end_date=dates[-1].isoformat(),
        number_of_dates=len(dates),
        duplicated_dates=duplicated,
        temporal_interval_summary=interval_summary,
        expected_frequency=expected_frequency,
        spatial_dimensions={lat: data.sizes[lat], lon: data.sizes[lon]},
        spatial_bounds={lat: [float(data[lat].min()), float(data[lat].max())], lon: [float(data[lon].min()), float(data[lon].max())]},
        valid_data_fraction=valid_fraction,
        weighted_spatial_coverage=weighted_coverage,
        constant_field_dates=constant_dates,
        all_nan_dates=all_nan_dates,
        warnings=warnings,
        errors=errors,
        overall_status="invalid" if errors else "warning" if warnings else "valid",
    )
    data.attrs["recognized_unit_kind"] = unit_kind
    return data, spatial_weights, report


def coverage_gate(
    metric: str, coverage: float, minimum: float, *, unit: str, family: str,
    n_observations: int,
) -> MetricResult | None:
    if not np.isfinite(coverage) or coverage < minimum:
        return not_calculated(
            metric,
            f"Valid spatial coverage {coverage:.3f} is below required {minimum:.3f}" if np.isfinite(coverage) else "Valid spatial denominator is zero",
            unit=unit,
            family=family,
            n_observations=n_observations,
            valid_coverage=coverage,
        )
    return None


def temporal_observation_gate(
    metric: str, n_observations: int, minimum: int, *, unit: str, family: str,
    coverage: float = np.nan,
) -> MetricResult | None:
    if n_observations < minimum:
        return not_calculated(
            metric,
            f"Only {n_observations} temporal observations; minimum is {minimum}",
            unit=unit,
            family=family,
            n_observations=n_observations,
            valid_coverage=coverage,
        )
    return None


def save_qc_report(report: QCReport, path: str | Path) -> Path:
    return export_json(report.to_dict(), path)
