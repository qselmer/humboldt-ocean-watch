"""Offline, failure-isolated orchestration for configured SST domains."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import pandas as pd
import xarray as xr

from src.data_loader import DEFAULT_SST_ALIASES, describe_sst_dataset, load_sst_domain_dataset
from src.sst_aggregation import SSTAggregationMetadata, aggregate_sst_to_target_resolution
from src.sst_domains import SSTDomainSpec, load_sst_domain_specs
from src.utils import resolve_project_path


@dataclass(frozen=True)
class SSTDomainResult:
    dataset: xr.Dataset | None
    source_dataset: xr.Dataset | None
    status: str
    spec: SSTDomainSpec
    source_mode: str | None
    source_path: Path | None
    latest_date: str | None
    warnings: tuple[str, ...]
    errors: tuple[str, ...]
    aggregation_metadata: Mapping[str, Any]
    climatology_available: bool

    def to_status_dict(self) -> dict[str, Any]:
        description = (
            None if self.dataset is None else describe_sst_dataset(self.dataset)
        )
        return {
            "domain_id": self.spec.domain_id,
            "label": self.spec.label,
            "role": self.spec.role,
            "status": self.status,
            "source_mode": self.source_mode,
            "source_path": None if self.source_path is None else str(self.source_path),
            "latest_date": self.latest_date,
            "bounds": self.spec.bounds.to_dict(),
            "target_resolution_degrees": self.spec.target_resolution_degrees,
            "dataset": description,
            "aggregation": dict(self.aggregation_metadata),
            "climatology_status": self.spec.climatology_status,
            "climatology_available": self.climatology_available,
            "warnings": list(self.warnings),
            "errors": list(self.errors),
        }


def _existing_path(candidates: Sequence[Path | None]) -> Path | None:
    for configured in candidates:
        if configured is None:
            continue
        path = resolve_project_path(configured)
        if path.exists():
            return path
    return None


def _climatology_available(spec: SSTDomainSpec) -> bool:
    candidates = (spec.climatology_daily_path, spec.climatology_monthly_path)
    return any(
        path is not None and resolve_project_path(path).exists() for path in candidates
    )


def _empty_result(
    spec: SSTDomainSpec,
    status: str,
    *,
    warnings: Sequence[str] = (),
    errors: Sequence[str] = (),
    source_mode: str | None = None,
    source_path: Path | None = None,
) -> SSTDomainResult:
    return SSTDomainResult(
        dataset=None,
        source_dataset=None,
        status=status,
        spec=spec,
        source_mode=source_mode,
        source_path=source_path,
        latest_date=None,
        warnings=tuple(warnings),
        errors=tuple(errors),
        aggregation_metadata={},
        climatology_available=_climatology_available(spec),
    )


def load_multidomain_sst(
    config: Mapping[str, Any],
    domain_ids: Sequence[str] | None = None,
    *,
    allow_demo: bool | None = None,
) -> dict[str, SSTDomainResult]:
    """Load local SST domains independently; never generate files or use network."""
    registry = load_sst_domain_specs(config)
    selected = tuple(domain_ids or ("pacific_context", "humboldt_coastal"))
    aliases = tuple(config.get("data", {}).get("sst_aliases", DEFAULT_SST_ALIASES))
    aggregation = config.get("sst", {}).get("aggregation", {})
    results: dict[str, SSTDomainResult] = {}
    for domain_id in selected:
        if domain_id not in registry.domains:
            raise ValueError(f"Unknown SST domain requested: {domain_id}")
        spec = registry.domains[domain_id]
        warnings: list[str] = []
        live = _existing_path((spec.live_path, spec.legacy_live_path))
        demo_allowed = spec.allow_demo_fallback if allow_demo is None else (
            bool(allow_demo) and spec.allow_demo_fallback
        )
        if live is not None:
            source_path, source_mode, status = live, "live", "available_live"
            if spec.legacy_live_path is not None and live == resolve_project_path(spec.legacy_live_path):
                warnings.append("Using legacy-compatible live path")
        else:
            demo = _existing_path((spec.demo_path, spec.legacy_demo_path)) if demo_allowed else None
            if demo is None:
                message = "No local live SST file is available"
                if demo_allowed:
                    message += " and no configured demo file exists"
                else:
                    message += "; demo fallback is disabled"
                results[domain_id] = _empty_result(spec, "missing", warnings=(message,))
                continue
            source_path, source_mode, status = demo, "demo", "available_demo"
            if spec.legacy_demo_path is not None and demo == resolve_project_path(spec.legacy_demo_path):
                warnings.append("Using legacy-compatible demo path")
        try:
            source_dataset = load_sst_domain_dataset(
                source_path, spec, aliases=aliases, source_mode=source_mode
            )
            prepared, metadata = aggregate_sst_to_target_resolution(
                source_dataset,
                spec.target_resolution_degrees,
                minimum_valid_fraction=float(
                    aggregation.get("minimum_valid_fraction", 0.80)
                ),
                regular_grid_tolerance=float(
                    aggregation.get("regular_grid_tolerance", 0.001)
                ),
                integer_factor_tolerance=float(
                    aggregation.get("integer_factor_tolerance", 0.001)
                ),
            )
        except (OSError, TypeError, ValueError) as exc:
            results[domain_id] = _empty_result(
                spec,
                "invalid",
                warnings=warnings,
                errors=(str(exc),),
                source_mode=source_mode,
                source_path=source_path,
            )
            continue
        latest = pd.Timestamp(prepared.time.values.max()).date().isoformat()
        results[domain_id] = SSTDomainResult(
            dataset=prepared,
            source_dataset=source_dataset,
            status=status,
            spec=spec,
            source_mode=source_mode,
            source_path=source_path,
            latest_date=latest,
            warnings=tuple(warnings),
            errors=(),
            aggregation_metadata=metadata.to_dict(),
            climatology_available=_climatology_available(spec),
        )
    return results
