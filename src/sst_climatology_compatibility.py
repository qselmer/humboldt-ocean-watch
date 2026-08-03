"""Strict, explainable compatibility checks for SST and climatology datasets."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import xarray as xr

from src.sst_climatology_spec import SSTClimatologySpec
from src.sst_domains import SSTDomainSpec


@dataclass(frozen=True)
class ClimatologyCompatibilityResult:
    compatible: bool
    status: str
    errors: tuple[str, ...]
    warnings: tuple[str, ...]
    domain_id: str
    source_mode: str | None
    climatology_mode: str | None
    source_product_family: str | None
    climatology_product_family: str | None
    source_resolution: tuple[float | None, float | None]
    climatology_resolution: tuple[float | None, float | None]
    coordinate_match: bool
    coverage: float | None
    method: str | None
    reference_period: tuple[int, int] | None
    allowed_operations: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        values["errors"] = list(self.errors)
        values["warnings"] = list(self.warnings)
        values["source_resolution"] = list(self.source_resolution)
        values["climatology_resolution"] = list(self.climatology_resolution)
        values["reference_period"] = (
            None if self.reference_period is None else list(self.reference_period)
        )
        values["allowed_operations"] = list(self.allowed_operations)
        return values


def missing_climatology_result(
    domain_id: str, source_mode: str | None = None
) -> ClimatologyCompatibilityResult:
    return ClimatologyCompatibilityResult(
        compatible=False,
        status="missing_climatology",
        errors=("No local climatology file is available",),
        warnings=(),
        domain_id=domain_id,
        source_mode=source_mode,
        climatology_mode=None,
        source_product_family=None,
        climatology_product_family=None,
        source_resolution=(None, None),
        climatology_resolution=(None, None),
        coordinate_match=False,
        coverage=None,
        method=None,
        reference_period=None,
        allowed_operations=(),
    )


def _resolution(dataset: xr.Dataset) -> tuple[float | None, float | None]:
    values: list[float | None] = []
    for coordinate in ("latitude", "longitude"):
        if coordinate not in dataset.coords or dataset[coordinate].size < 2:
            values.append(None)
            continue
        differences = np.diff(np.asarray(dataset[coordinate].values, dtype=float))
        finite = np.abs(differences[np.isfinite(differences)])
        values.append(None if finite.size == 0 else float(np.median(finite)))
    return values[0], values[1]


def _coordinate_match(
    source: xr.Dataset, climatology: xr.Dataset, tolerance: float
) -> bool:
    for coordinate in ("latitude", "longitude"):
        if coordinate not in source.coords or coordinate not in climatology.coords:
            return False
        left = np.sort(np.asarray(source[coordinate].values, dtype=float))
        right = np.sort(np.asarray(climatology[coordinate].values, dtype=float))
        if left.shape != right.shape or not np.allclose(
            left, right, rtol=0.0, atol=tolerance, equal_nan=False
        ):
            return False
    return True


def _first_attr(dataset: xr.Dataset, *names: str) -> str | None:
    for name in names:
        value = dataset.attrs.get(name)
        if value not in (None, ""):
            return str(value)
    return None


def _method(dataset: xr.Dataset) -> str | None:
    explicit = _first_attr(dataset, "climatology_method", "method")
    if explicit:
        return explicit
    if "climatological_day" in dataset.dims or "dayofyear" in dataset.dims:
        return "daily_smoothed"
    if "month" in dataset.dims:
        return "monthly"
    return None


def _reference_period(dataset: xr.Dataset) -> tuple[int, int] | None:
    raw = dataset.attrs.get("reference_period")
    if isinstance(raw, str):
        cleaned = raw.replace("–", "-").replace("to", "-").replace(",", "-")
        parts = [part.strip() for part in cleaned.split("-") if part.strip()]
    elif isinstance(raw, (list, tuple, np.ndarray)):
        parts = list(raw)
    else:
        start = dataset.attrs.get("reference_start")
        end = dataset.attrs.get("reference_end")
        parts = [] if start is None or end is None else [start, end]
    if len(parts) != 2:
        return None
    try:
        return int(parts[0]), int(parts[1])
    except (TypeError, ValueError):
        return None


def _mean_name(dataset: xr.Dataset) -> str | None:
    for name in ("climatology_mean", "climatological_mean"):
        if name in dataset.data_vars:
            return name
    return None


def _std_name(dataset: xr.Dataset) -> str | None:
    for name in ("climatology_std", "climatological_standard_deviation"):
        if name in dataset.data_vars:
            return name
    return None


def _unit(dataset: xr.Dataset, variable: str | None) -> str | None:
    return None if variable is None else str(dataset[variable].attrs.get("units", ""))


def validate_sst_climatology_compatibility(
    sst_dataset: xr.Dataset,
    climatology_dataset: xr.Dataset | None,
    domain_spec: SSTDomainSpec,
    climatology_spec: SSTClimatologySpec,
) -> ClimatologyCompatibilityResult:
    """Validate scientific compatibility without interpolation or extrapolation."""
    if not isinstance(sst_dataset, xr.Dataset):
        raise TypeError("sst_dataset must be an xarray.Dataset")
    if domain_spec.domain_id != climatology_spec.domain_id:
        raise ValueError("Domain and climatology specifications must describe the same domain")
    if climatology_dataset is None:
        return missing_climatology_result(
            domain_spec.domain_id, str(sst_dataset.attrs.get("source_mode", "unknown"))
        )

    errors: list[str] = []
    warnings: list[str] = []
    status = "compatible"
    detected_statuses: list[str] = []

    def mark(value: str) -> None:
        detected_statuses.append(value)
    source_mode = _first_attr(sst_dataset, "source_mode", "data_mode") or "unknown"
    climatology_mode = _first_attr(
        climatology_dataset, "climatology_source_mode", "source_mode"
    ) or "unknown"
    source_family = _first_attr(sst_dataset, "source_product_family", "product_family")
    climatology_family = _first_attr(
        climatology_dataset, "source_product_family", "product_family"
    )
    source_domain = _first_attr(sst_dataset, "domain_id")
    climate_domain = _first_attr(climatology_dataset, "domain_id")
    source_geography = _first_attr(sst_dataset, "geography_id")
    climate_geography = _first_attr(climatology_dataset, "geography_id")

    if source_domain not in (None, domain_spec.domain_id) or climate_domain != domain_spec.domain_id:
        mark("domain_mismatch")
        errors.append(
            f"Expected domain_id {domain_spec.domain_id!r}; SST={source_domain!r}, climatology={climate_domain!r}"
        )
    elif source_geography not in (None, domain_spec.geography_id) or climate_geography not in (
        None,
        domain_spec.geography_id,
    ):
        mark("domain_mismatch")
        errors.append(
            f"Expected geography_id {domain_spec.geography_id!r}; SST={source_geography!r}, climatology={climate_geography!r}"
        )

    synthetic_climatology = climatology_mode in {
        "demo",
        "synthetic",
        "synthetic_demonstration",
    } or climatology_family == "synthetic_demo"
    if source_mode in {"live", "Copernicus cached data"} and synthetic_climatology:
        mark("synthetic_live_mismatch")
        errors.append("Synthetic demonstration climatology cannot be applied to live SST")
    expected_family = (
        "synthetic_demo" if source_mode in {"demo", "Synthetic demonstration data"} else climatology_spec.source_product_family
    )
    if source_family is None:
        source_family = expected_family
        warnings.append("SST product family was inferred from configured source mode")
    if climatology_family is None:
        errors.append("Climatology source_product_family is missing")
        mark("invalid_climatology")
    elif source_family != climatology_family:
        mark("product_family_mismatch")
        errors.append(
            f"SST product family {source_family!r} differs from climatology {climatology_family!r}"
        )

    source_resolution = _resolution(sst_dataset)
    climate_resolution = _resolution(climatology_dataset)
    comparable = all(value is not None for value in (*source_resolution, *climate_resolution))
    if not comparable or any(
        not np.isclose(left, right, rtol=0.0, atol=climatology_spec.resolution_tolerance_degrees)
        for left, right in zip(source_resolution, climate_resolution)
        if left is not None and right is not None
    ):
        mark("resolution_mismatch")
        errors.append(
            f"SST resolution {source_resolution} differs from climatology {climate_resolution}"
        )
    coordinates_match = _coordinate_match(
        sst_dataset, climatology_dataset, climatology_spec.coordinate_tolerance_degrees
    )
    if not coordinates_match:
        mark("coordinate_mismatch")
        errors.append("SST and climatology coordinates do not match without interpolation")

    mean_name = _mean_name(climatology_dataset)
    std_name = _std_name(climatology_dataset)
    if "sst" not in sst_dataset or str(sst_dataset.sst.attrs.get("units", "")) != "degrees_Celsius":
        mark("invalid_climatology")
        errors.append("SST variable with degrees_Celsius units is required")
    if mean_name is None:
        mark("missing_mean")
        errors.append("Climatology mean variable is missing")
    elif _unit(climatology_dataset, mean_name) != "degrees_Celsius":
        mark("invalid_climatology")
        errors.append("Climatology mean units must be degrees_Celsius")

    method = _method(climatology_dataset)
    calendar = (_first_attr(climatology_dataset, "calendar", "calendar_mapping") or "").lower()
    if method == "daily_smoothed":
        if climatology_dataset.sizes.get("climatological_day", climatology_dataset.sizes.get("dayofyear")) != 366:
            mark("unsupported_calendar")
            errors.append("Daily climatology must use the stable 366-day calendar")
        if calendar and not any(token in calendar for token in ("stable", "gregorian", "leap year 2000")):
            mark("unsupported_calendar")
            errors.append(f"Unsupported climatology calendar: {calendar}")
    elif method == "monthly":
        if climatology_dataset.sizes.get("month") != 12:
            mark("unsupported_calendar")
            errors.append("Monthly climatology must contain 12 months")
    else:
        mark("unsupported_calendar")
        errors.append("Climatology has no supported daily or monthly dimension")

    period = _reference_period(climatology_dataset)
    if not synthetic_climatology and period != climatology_spec.reference_period:
        mark("invalid_climatology")
        errors.append(
            f"Reference period {period!r} does not match configured {climatology_spec.reference_period!r}"
        )
    if synthetic_climatology and period is None:
        warnings.append("Synthetic demonstration reference period is intentionally non-observational")

    coverage: float | None = None
    if mean_name is not None:
        mean_field = climatology_dataset[mean_name]
        finite_count = mean_field.count()
        if hasattr(finite_count.data, "compute"):
            finite_count = finite_count.compute()
        coverage = float(finite_count / mean_field.size) if mean_field.size else 0.0
        if coverage < climatology_spec.minimum_valid_coverage:
            mark("insufficient_coverage")
            errors.append(
                f"Finite climatology coverage {coverage:.3f} is below {climatology_spec.minimum_valid_coverage:.3f}"
            )
        infinite = np.isinf(mean_field).any()
        if hasattr(infinite.data, "compute"):
            infinite = infinite.compute()
        if bool(infinite):
            mark("invalid_climatology")
            errors.append("Climatology mean contains infinite values")

    optional_status: str | None = None
    if std_name is None:
        optional_status = "missing_standard_deviation"
        warnings.append("Standard deviation is missing; z-score will not be calculated")
    elif _unit(climatology_dataset, std_name) != "degrees_Celsius":
        mark("invalid_climatology")
        errors.append("Climatology standard-deviation units must be degrees_Celsius")
    missing_percentiles = [
        name for name in ("threshold_p10", "threshold_p90") if name not in climatology_dataset
    ]
    if missing_percentiles:
        optional_status = optional_status or "missing_percentiles"
        warnings.append(
            "Percentile thresholds are missing: " + ", ".join(missing_percentiles)
        )

    compatible = not errors
    status = (
        detected_statuses[0]
        if detected_statuses
        else optional_status or "compatible"
    )
    allowed: list[str] = []
    if compatible:
        allowed.append("anomaly")
        if std_name is not None:
            allowed.append("z_score")
        if not missing_percentiles:
            allowed.append("percentile_exceedance")
    return ClimatologyCompatibilityResult(
        compatible=compatible,
        status=status,
        errors=tuple(errors),
        warnings=tuple(warnings),
        domain_id=domain_spec.domain_id,
        source_mode=source_mode,
        climatology_mode=climatology_mode,
        source_product_family=source_family,
        climatology_product_family=climatology_family,
        source_resolution=source_resolution,
        climatology_resolution=climate_resolution,
        coordinate_match=coordinates_match,
        coverage=coverage,
        method=method,
        reference_period=period,
        allowed_operations=tuple(allowed),
    )
