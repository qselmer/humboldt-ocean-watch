"""Failure-isolated local climatology selection for configured SST domains."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import xarray as xr

from src.multidomain_sst import SSTDomainResult
from src.sst_climatology_compatibility import (
    ClimatologyCompatibilityResult,
    missing_climatology_result,
    validate_sst_climatology_compatibility,
)
from src.sst_climatology_spec import SSTClimatologySpec, load_sst_climatology_specs
from src.sst_domains import load_sst_domain_specs
from src.utils import resolve_project_path


@dataclass(frozen=True)
class ClimatologyDomainResult:
    dataset: xr.Dataset | None
    status: str
    path: Path | None
    method: str | None
    source_type: str | None
    compatibility: ClimatologyCompatibilityResult
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    reference_period: tuple[int, int] | None
    latest_validation_metadata: Mapping[str, Any]
    fallback_used: bool = False
    fallback_reason: str | None = None
    climatology_source_mode: str | None = None

    def to_status_dict(self) -> dict[str, Any]:
        return {
            "status": self.status,
            "path": None if self.path is None else str(self.path),
            "method": self.method,
            "source_type": self.source_type,
            "compatibility": self.compatibility.to_dict(),
            "warnings": list(self.warnings),
            "errors": list(self.errors),
            "reference_period": (
                None if self.reference_period is None else list(self.reference_period)
            ),
            "latest_validation_metadata": dict(self.latest_validation_metadata),
            "fallback_used": self.fallback_used,
            "fallback_reason": self.fallback_reason,
            "climatology_source_mode": self.climatology_source_mode,
        }


def _load(path: Path, *, method: str, source_mode: str, spec: SSTClimatologySpec) -> xr.Dataset:
    dataset = xr.open_dataset(path, chunks={})
    dataset.attrs = dict(dataset.attrs)
    dataset.attrs.setdefault("domain_id", spec.domain_id)
    dataset.attrs.setdefault("geography_id", spec.geography_id)
    dataset.attrs.setdefault(
        "source_product_family",
        "synthetic_demo" if source_mode == "synthetic_demonstration" else spec.source_product_family,
    )
    dataset.attrs.setdefault("climatology_method", method)
    dataset.attrs.setdefault("reference_period", f"{spec.reference_start}-{spec.reference_end}")
    dataset.attrs["climatology_source_mode"] = source_mode
    return dataset


def load_multidomain_climatologies(
    config: Mapping[str, Any],
    sst_results: Mapping[str, SSTDomainResult],
    domain_ids: Sequence[str] | None = None,
) -> dict[str, ClimatologyDomainResult]:
    """Select compatible local climatology per domain without global failure."""
    domain_specs = load_sst_domain_specs(config).domains
    climate_specs = load_sst_climatology_specs(config)
    selected = tuple(domain_ids or sst_results.keys())
    results: dict[str, ClimatologyDomainResult] = {}
    for domain_id in selected:
        if domain_id not in domain_specs or domain_id not in sst_results:
            raise ValueError(f"Unknown or unloaded SST domain: {domain_id}")
        source = sst_results[domain_id]
        spec = climate_specs[domain_id]
        if source.dataset is None:
            compatibility = missing_climatology_result(domain_id, source.source_mode)
            results[domain_id] = ClimatologyDomainResult(
                None,
                "missing_sst",
                None,
                None,
                None,
                compatibility,
                source.warnings,
                source.errors or ("SST dataset is unavailable",),
                None,
                compatibility.to_dict(),
            )
            continue

        attempts: list[tuple[Path, str, str, bool, str | None]] = []
        daily = resolve_project_path(spec.daily_path)
        monthly = resolve_project_path(spec.monthly_path)
        if daily.exists():
            attempts.append((daily, "daily_smoothed", "real", False, None))
        if spec.allow_monthly_fallback and monthly.exists():
            attempts.append(
                (
                    monthly,
                    "monthly",
                    "real",
                    True,
                    "Daily climatology is missing or incompatible",
                )
            )
        if source.source_mode == "demo" and spec.demo_path is not None:
            demo = resolve_project_path(spec.demo_path)
            if demo.exists():
                attempts.append(
                    (
                        demo,
                        "daily_smoothed",
                        "synthetic_demonstration",
                        bool(attempts),
                        "No compatible real climatology is available for demonstration SST",
                    )
                )

        attempt_warnings: list[str] = []
        last_compatibility = missing_climatology_result(domain_id, source.source_mode)
        last_path: Path | None = None
        for path, method, source_type, fallback_used, fallback_reason in attempts:
            last_path = path
            try:
                dataset = _load(path, method=method, source_mode=source_type, spec=spec)
                compatibility = validate_sst_climatology_compatibility(
                    source.dataset, dataset, domain_specs[domain_id], spec
                )
            except (OSError, TypeError, ValueError) as exc:
                attempt_warnings.append(f"{path}: {exc}")
                continue
            last_compatibility = compatibility
            if not compatibility.compatible:
                attempt_warnings.append(
                    f"{path}: {compatibility.status}: {'; '.join(compatibility.errors)}"
                )
                continue
            results[domain_id] = ClimatologyDomainResult(
                dataset=dataset,
                status="available",
                path=path,
                method=method,
                source_type=source_type,
                compatibility=compatibility,
                warnings=tuple(attempt_warnings) + compatibility.warnings,
                errors=(),
                reference_period=compatibility.reference_period,
                latest_validation_metadata=compatibility.to_dict(),
                fallback_used=fallback_used,
                fallback_reason=fallback_reason if fallback_used else None,
                climatology_source_mode=source_type,
            )
            break
        else:
            status = (
                "missing_climatology" if not attempts else "invalid_climatology"
            )
            results[domain_id] = ClimatologyDomainResult(
                dataset=None,
                status=status,
                path=last_path,
                method=None,
                source_type=None,
                compatibility=last_compatibility,
                warnings=tuple(attempt_warnings),
                errors=last_compatibility.errors,
                reference_period=None,
                latest_validation_metadata=last_compatibility.to_dict(),
            )
    return results
