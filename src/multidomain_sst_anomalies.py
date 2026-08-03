"""Explicit multidomain SST anomaly calculations over compatible baselines."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np
import xarray as xr

from src.daily_climatology import calendar_dayofyear
from src.multidomain_climatology import ClimatologyDomainResult
from src.multidomain_sst import SSTDomainResult
from src.sst_climatology_compatibility import ClimatologyCompatibilityResult
from src.sst_climatology_spec import SSTClimatologySpec, load_sst_climatology_specs


@dataclass(frozen=True)
class AnomalyDomainResult:
    dataset: xr.Dataset | None
    status: str
    domain_id: str
    compatibility: ClimatologyCompatibilityResult
    anomaly_variables: tuple[str, ...]
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    climatology_method: str | None
    reference_period: tuple[int, int] | None
    source_mode: str | None

    def to_status_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "domain_id": self.domain_id,
            "compatibility": self.compatibility.to_dict(),
            "anomaly_variables": list(self.anomaly_variables),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "climatology_method": self.climatology_method,
            "reference_period": (
                None if self.reference_period is None else list(self.reference_period)
            ),
            "source_mode": self.source_mode,
        }


def _variable(dataset: xr.Dataset, *names: str) -> xr.DataArray | None:
    for name in names:
        if name in dataset:
            return dataset[name]
    return None


def _match(field: xr.DataArray, dates: xr.DataArray, method: str) -> xr.DataArray:
    if method == "daily_smoothed":
        dimension = "climatological_day" if "climatological_day" in field.dims else "dayofyear"
        indexer = calendar_dayofyear(dates)
        if dimension == "dayofyear":
            indexer = indexer.rename("dayofyear")
        return field.sel({dimension: indexer})
    if method == "monthly":
        return field.sel(month=dates.dt.month)
    raise ValueError(f"Unsupported climatology method: {method}")


def calculate_sst_anomaly_domain(
    sst_dataset: xr.Dataset,
    climatology: ClimatologyDomainResult,
    spec: SSTClimatologySpec,
    *,
    standard_deviation_epsilon: float = 1.0e-12,
) -> AnomalyDomainResult:
    """Calculate anomaly fields only after the compatibility contract passes."""
    compatibility = climatology.compatibility
    source_mode = str(sst_dataset.attrs.get("source_mode", "unknown"))
    if climatology.dataset is None:
        status = (
            "not_calculated_missing_climatology"
            if climatology.status == "missing_climatology"
            else "not_calculated_incompatible"
        )
        return AnomalyDomainResult(
            None,
            status,
            spec.domain_id,
            compatibility,
            (),
            climatology.warnings,
            climatology.errors,
            climatology.method,
            climatology.reference_period,
            source_mode,
        )
    if not compatibility.compatible:
        return AnomalyDomainResult(
            None,
            "not_calculated_incompatible",
            spec.domain_id,
            compatibility,
            (),
            compatibility.warnings,
            compatibility.errors,
            climatology.method,
            climatology.reference_period,
            source_mode,
        )
    if not np.isfinite(standard_deviation_epsilon) or standard_deviation_epsilon <= 0:
        raise ValueError("standard_deviation_epsilon must be finite and positive")

    climate = climatology.dataset.sortby("latitude").sortby("longitude")
    source = sst_dataset.sortby("latitude").sortby("longitude")
    method = climatology.method or compatibility.method
    if method is None:
        raise ValueError("Compatible climatology has no method")
    mean = _variable(climate, "climatology_mean", "climatological_mean")
    if mean is None:
        raise ValueError("Compatibility allowed anomaly calculation without a mean")
    matched_mean = _match(mean, source.time, method)
    anomaly = (source.sst - matched_mean).rename("sst_anomaly_c")
    anomaly.attrs.update(
        units="degrees_Celsius",
        long_name="sea surface temperature anomaly",
        calculation="sst minus compatible date-matched climatology_mean",
    )

    spatial = source.sizes["latitude"] * source.sizes["longitude"]
    coverage = (source.sst.notnull().sum(("latitude", "longitude")) / spatial).rename(
        "valid_coverage"
    )
    coverage.attrs.update(units="1", long_name="finite SST spatial coverage")
    valid_dates = coverage >= spec.minimum_valid_coverage
    anomaly = anomaly.where(valid_dates)
    result = xr.Dataset({"sst": source.sst, "sst_anomaly_c": anomaly, "valid_coverage": coverage})
    variables = ["sst_anomaly_c", "valid_coverage"]
    warnings = list(climatology.warnings)

    std = _variable(climate, "climatology_std", "climatological_standard_deviation")
    if std is not None:
        matched_std = _match(std, source.time, method)
        valid_std = np.isfinite(matched_std) & (matched_std > standard_deviation_epsilon)
        z_score = xr.where(valid_std & valid_dates, anomaly / matched_std, np.nan).rename(
            "sst_z_score"
        )
        z_score.attrs.update(
            units="1",
            long_name="standardized sea surface temperature anomaly",
            standard_deviation_epsilon=float(standard_deviation_epsilon),
            invalid_standard_deviation_status="not_calculated",
        )
        result["sst_z_score"] = z_score
        variables.append("sst_z_score")
        invalid_std = bool((~valid_std & matched_std.notnull()).any())
        if invalid_std:
            warnings.append("Zero or invalid climatology standard deviation produced NaN z-scores")

    p10 = _variable(climate, "threshold_p10")
    p90 = _variable(climate, "threshold_p90")
    if p10 is not None and p90 is not None:
        matched_p10 = _match(p10, source.time, method)
        matched_p90 = _match(p90, source.time, method)
        finite = source.sst.notnull() & matched_p10.notnull() & matched_p90.notnull()
        result["below_p10"] = xr.where(
            finite & valid_dates, source.sst < matched_p10, np.nan
        )
        result["exceeds_p90"] = xr.where(
            finite & valid_dates, source.sst > matched_p90, np.nan
        )
        result["percentile_status"] = xr.where(
            finite & valid_dates, "calculated", "not_calculated"
        )
        variables.extend(("below_p10", "exceeds_p90", "percentile_status"))

    if not bool(valid_dates.any()):
        status = "not_calculated_insufficient_coverage"
        result = None
        variables = []
    elif std is None:
        status = "partial_missing_std"
    elif p10 is None or p90 is None:
        status = "partial_missing_percentiles"
    else:
        status = "calculated"
    if result is not None:
        result.attrs.update(
            domain_id=spec.domain_id,
            geography_id=spec.geography_id,
            source_mode=source_mode,
            climatology_source_mode=climatology.climatology_source_mode or "unknown",
            climatology_method=method,
            reference_period=f"{spec.reference_start}-{spec.reference_end}",
            fallback_used=str(climatology.fallback_used).lower(),
            fallback_reason=climatology.fallback_reason or "",
            compatibility_status=compatibility.status,
            official_enso_classification="not_produced",
        )
    return AnomalyDomainResult(
        result,
        status,
        spec.domain_id,
        compatibility,
        tuple(variables),
        tuple(warnings),
        (),
        method,
        climatology.reference_period,
        source_mode,
    )


def calculate_multidomain_sst_anomalies(
    config: Mapping[str, Any],
    sst_results: Mapping[str, SSTDomainResult],
    climatology_results: Mapping[str, ClimatologyDomainResult],
    domain_ids: Sequence[str] | None = None,
) -> dict[str, AnomalyDomainResult]:
    specs = load_sst_climatology_specs(config)
    defaults = config.get("sst", {}).get("climatology_defaults", {})
    epsilon = float(defaults.get("standard_deviation_epsilon", 1.0e-12))
    selected = tuple(domain_ids or sst_results.keys())
    results: dict[str, AnomalyDomainResult] = {}
    for domain_id in selected:
        source = sst_results[domain_id]
        climate = climatology_results[domain_id]
        if source.dataset is None:
            results[domain_id] = AnomalyDomainResult(
                None,
                "not_calculated_incompatible",
                domain_id,
                climate.compatibility,
                (),
                source.warnings,
                source.errors or ("SST dataset is unavailable",),
                climate.method,
                climate.reference_period,
                source.source_mode,
            )
            continue
        results[domain_id] = calculate_sst_anomaly_domain(
            source.dataset,
            climate,
            specs[domain_id],
            standard_deviation_epsilon=epsilon,
        )
    return results
