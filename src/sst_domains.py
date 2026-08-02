"""Immutable SST domain specifications resolved from the geography registry."""

from __future__ import annotations

from dataclasses import dataclass
import math
from pathlib import Path
from typing import Any, Mapping

from src.geography import GeographicBounds, load_geography_registry


@dataclass(frozen=True)
class SSTDomainSpec:
    """Validated SST configuration for one geographic domain."""

    domain_id: str
    label: str
    role: str
    bounds: GeographicBounds
    target_resolution_degrees: float
    live_path: Path
    demo_path: Path
    allow_demo_fallback: bool
    geography_type: str
    geography_id: str
    climatology_status: str
    climatology_daily_path: Path | None = None
    climatology_monthly_path: Path | None = None
    legacy_live_path: Path | None = None
    legacy_demo_path: Path | None = None

    @property
    def longitude_bounds(self) -> tuple[float, float]:
        return self.bounds.longitude

    @property
    def latitude_bounds(self) -> tuple[float, float]:
        return self.bounds.latitude

    def to_dict(self) -> dict[str, Any]:
        return {
            "domain_id": self.domain_id,
            "label": self.label,
            "role": self.role,
            "geography_type": self.geography_type,
            "geography_id": self.geography_id,
            **self.bounds.to_dict(),
            "target_resolution_degrees": self.target_resolution_degrees,
            "live_path": str(self.live_path),
            "demo_path": str(self.demo_path),
            "legacy_live_path": (
                None if self.legacy_live_path is None else str(self.legacy_live_path)
            ),
            "legacy_demo_path": (
                None if self.legacy_demo_path is None else str(self.legacy_demo_path)
            ),
            "allow_demo_fallback": self.allow_demo_fallback,
            "climatology_status": self.climatology_status,
            "climatology_daily_path": (
                None
                if self.climatology_daily_path is None
                else str(self.climatology_daily_path)
            ),
            "climatology_monthly_path": (
                None
                if self.climatology_monthly_path is None
                else str(self.climatology_monthly_path)
            ),
        }


@dataclass(frozen=True)
class SSTDomainRegistry:
    coordinate_convention: str
    active_operational_domain: str
    domains: Mapping[str, SSTDomainSpec]


def _optional_path(values: Mapping[str, Any], key: str) -> Path | None:
    value = values.get(key)
    return None if value in (None, "") else Path(str(value))


def _resolve_bounds(
    geography_type: str, geography_id: str, geography: Any
) -> GeographicBounds:
    if geography_type == "standard_region":
        try:
            return geography.standard_regions[geography_id].bounds
        except KeyError as exc:
            raise ValueError(f"Unknown standard-region geography_id: {geography_id}") from exc
    if geography_type == "analysis_domain":
        try:
            return geography.analysis_domains[geography_id].bounds
        except KeyError as exc:
            raise ValueError(f"Unknown analysis-domain geography_id: {geography_id}") from exc
    raise ValueError(
        f"Unsupported geography_type {geography_type!r}; expected standard_region or analysis_domain"
    )


def load_sst_domain_specs(config: Mapping[str, Any]) -> SSTDomainRegistry:
    """Resolve and validate all SST domain references without duplicating bounds."""
    values = config.get("sst")
    if not isinstance(values, Mapping):
        raise ValueError("Configuration must contain an 'sst' mapping")
    geography = load_geography_registry(config)
    convention = str(values.get("coordinate_convention", ""))
    if convention != geography.coordinate_convention:
        raise ValueError("SST and geography coordinate conventions must match")
    domain_values = values.get("domains")
    if not isinstance(domain_values, Mapping) or not domain_values:
        raise ValueError("sst.domains must be a non-empty mapping")

    domains: dict[str, SSTDomainSpec] = {}
    for raw_id, raw in domain_values.items():
        domain_id = str(raw_id)
        if domain_id in domains:
            raise ValueError(f"Duplicate SST domain ID: {domain_id}")
        if not isinstance(raw, Mapping):
            raise ValueError(f"SST domain {domain_id!r} must be a mapping")
        geography_type = str(raw.get("geography_type", ""))
        geography_id = str(raw.get("geography_id", ""))
        bounds = _resolve_bounds(geography_type, geography_id, geography)
        try:
            resolution = float(raw.get("target_resolution_degrees"))
        except (TypeError, ValueError) as exc:
            raise ValueError(f"SST domain {domain_id!r} requires a numeric resolution") from exc
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError(f"SST domain {domain_id!r} resolution must be finite and positive")
        climatology = raw.get("climatology", {})
        if not isinstance(climatology, Mapping):
            raise ValueError(f"SST domain {domain_id!r} climatology must be a mapping")
        label = str(raw.get("label", "")).strip()
        role = str(raw.get("role", "")).strip()
        if not label or not role:
            raise ValueError(f"SST domain {domain_id!r} requires label and role")
        live_path = _optional_path(raw, "live_path")
        demo_path = _optional_path(raw, "demo_path")
        if live_path is None or demo_path is None:
            raise ValueError(f"SST domain {domain_id!r} requires live_path and demo_path")
        domains[domain_id] = SSTDomainSpec(
            domain_id=domain_id,
            label=label,
            role=role,
            bounds=bounds,
            target_resolution_degrees=resolution,
            live_path=live_path,
            demo_path=demo_path,
            allow_demo_fallback=bool(raw.get("allow_demo_fallback", False)),
            geography_type=geography_type,
            geography_id=geography_id,
            climatology_status=str(climatology.get("status", "unknown")),
            climatology_daily_path=_optional_path(climatology, "daily_path"),
            climatology_monthly_path=_optional_path(climatology, "monthly_path"),
            legacy_live_path=_optional_path(raw, "legacy_live_path"),
            legacy_demo_path=_optional_path(raw, "legacy_demo_path"),
        )

    required = {"nino12", "pacific_context", "humboldt_coastal"}
    if set(domains) != required:
        raise ValueError(
            "SST domains must be exactly nino12, pacific_context, and humboldt_coastal"
        )
    active = str(values.get("active_operational_domain", ""))
    if active not in domains:
        raise ValueError(f"Unknown active operational SST domain: {active!r}")

    pacific = domains["pacific_context"].bounds
    for region_id in ("nino34", "nino3", "nino12"):
        if not pacific.contains(geography.standard_regions[region_id].bounds):
            raise ValueError(f"Pacific context does not fully contain {region_id}")

    coastal = domains["humboldt_coastal"]
    corridor = geography.coastal_corridors["humboldt_60nm"]
    tolerance = max(1.0e-10, coastal.target_resolution_degrees / 2.0)
    if (
        corridor.latitude_bounds[0] < coastal.bounds.south - tolerance
        or corridor.latitude_bounds[1] > coastal.bounds.north + tolerance
    ):
        raise ValueError("Humboldt coastal SST domain does not contain the coastal corridor")
    if corridor.include_islands or corridor.coastline_type != "continental_mainland_only":
        raise ValueError("Humboldt coastal corridor must exclude islands as buffer sources")

    return SSTDomainRegistry(
        coordinate_convention=convention,
        active_operational_domain=active,
        domains=domains,
    )
