"""Immutable, domain-specific SST climatology configuration."""

from __future__ import annotations

from dataclasses import asdict, dataclass
import math
from pathlib import Path
from typing import Any, Mapping

from src.sst_domains import load_sst_domain_specs


SUPPORTED_METHODS = frozenset({"daily_smoothed", "monthly"})


@dataclass(frozen=True)
class SSTClimatologySpec:
    domain_id: str
    geography_id: str
    method: str
    reference_start: int
    reference_end: int
    daily_path: Path
    monthly_path: Path
    demo_path: Path | None
    status: str
    source_product_family: str
    target_resolution_degrees: float
    sampling_half_window_days: int
    smoothing_window_days: int
    minimum_valid_coverage: float
    leap_day_method: str
    allow_monthly_fallback: bool
    allow_synthetic_with_live_data: bool
    coordinate_tolerance_degrees: float
    resolution_tolerance_degrees: float

    @property
    def reference_period(self) -> tuple[int, int]:
        return self.reference_start, self.reference_end

    def to_dict(self) -> dict[str, Any]:
        values = asdict(self)
        for key in ("daily_path", "monthly_path", "demo_path"):
            values[key] = None if values[key] is None else str(values[key])
        values["reference_period"] = list(self.reference_period)
        return values


def _path(values: Mapping[str, Any], key: str, domain_id: str) -> Path:
    value = values.get(key)
    if value in (None, ""):
        raise ValueError(f"SST climatology {domain_id!r} requires {key}")
    return Path(str(value))


def load_sst_climatology_specs(
    config: Mapping[str, Any],
) -> dict[str, SSTClimatologySpec]:
    """Load and strictly validate one independent climatology spec per SST domain."""
    registry = load_sst_domain_specs(config)
    sst = config.get("sst")
    if not isinstance(sst, Mapping):
        raise ValueError("Configuration must contain an 'sst' mapping")
    defaults = sst.get("climatology_defaults", {})
    if not isinstance(defaults, Mapping):
        raise ValueError("sst.climatology_defaults must be a mapping")
    configured = sst.get("domains")
    if not isinstance(configured, Mapping):
        raise ValueError("sst.domains must be a mapping")

    specs: dict[str, SSTClimatologySpec] = {}
    used_paths: dict[Path, str] = {}
    for domain_id, domain_spec in registry.domains.items():
        raw_domain = configured[domain_id]
        raw = raw_domain.get("climatology", {})
        if not isinstance(raw, Mapping):
            raise ValueError(f"SST domain {domain_id!r} climatology must be a mapping")
        period = raw.get(
            "reference_period",
            [defaults.get("reference_start", 1991), defaults.get("reference_end", 2020)],
        )
        if not isinstance(period, (list, tuple)) or len(period) != 2:
            raise ValueError(f"SST climatology {domain_id!r} reference_period must have two years")
        start, end = int(period[0]), int(period[1])
        if start > end:
            raise ValueError(f"SST climatology {domain_id!r} reference period is reversed")
        method = str(raw.get("method", defaults.get("primary_method", "daily_smoothed")))
        if method not in SUPPORTED_METHODS:
            raise ValueError(f"Unsupported SST climatology method for {domain_id!r}: {method}")
        resolution = float(
            raw.get("target_resolution_degrees", domain_spec.target_resolution_degrees)
        )
        minimum = float(
            raw.get("minimum_valid_coverage", defaults.get("minimum_valid_coverage", 0.80))
        )
        if not math.isfinite(resolution) or resolution <= 0:
            raise ValueError(f"SST climatology {domain_id!r} resolution must be positive")
        if not math.isfinite(minimum) or not 0.0 <= minimum <= 1.0:
            raise ValueError(f"SST climatology {domain_id!r} minimum coverage must be in [0, 1]")
        family = str(raw.get("source_product_family", "")).strip()
        if not family:
            raise ValueError(f"SST climatology {domain_id!r} requires source_product_family")
        daily = _path(raw, "daily_path", domain_id)
        monthly = _path(raw, "monthly_path", domain_id)
        demo_value = raw.get("demo_path")
        demo = None if demo_value in (None, "") else Path(str(demo_value))
        for candidate in (daily, monthly, demo):
            if candidate is None:
                continue
            normalized = Path(candidate.as_posix().lower())
            owner = used_paths.get(normalized)
            if owner is not None and owner != domain_id:
                raise ValueError(
                    f"Climatology path {candidate} is shared by {owner!r} and {domain_id!r}"
                )
            used_paths[normalized] = domain_id
        specs[domain_id] = SSTClimatologySpec(
            domain_id=domain_id,
            geography_id=domain_spec.geography_id,
            method=method,
            reference_start=start,
            reference_end=end,
            daily_path=daily,
            monthly_path=monthly,
            demo_path=demo,
            status=str(raw.get("status", "unknown")),
            source_product_family=family,
            target_resolution_degrees=resolution,
            sampling_half_window_days=int(
                raw.get("sampling_half_window_days", defaults.get("sampling_half_window_days", 5))
            ),
            smoothing_window_days=int(
                raw.get("smoothing_window_days", defaults.get("smoothing_window_days", 31))
            ),
            minimum_valid_coverage=minimum,
            leap_day_method=str(
                raw.get(
                    "leap_day_method",
                    defaults.get("leap_day_method", "stable_366_feb29_bin60_mar01_bin61"),
                )
            ),
            allow_monthly_fallback=bool(
                raw.get("allow_monthly_fallback", defaults.get("fallback_method") == "monthly")
            ),
            allow_synthetic_with_live_data=bool(
                raw.get(
                    "allow_synthetic_with_live_data",
                    defaults.get("allow_synthetic_with_live_data", False),
                )
            ),
            coordinate_tolerance_degrees=float(
                defaults.get("coordinate_tolerance_degrees", 1.0e-6)
            ),
            resolution_tolerance_degrees=float(
                defaults.get("resolution_tolerance_degrees", 1.0e-6)
            ),
        )

    operational = config.get("climatology", {})
    nino = specs["nino12"]
    if Path(str(operational.get("daily_path"))) != nino.daily_path:
        raise ValueError("nino12 daily climatology path must preserve the operational path")
    if Path(str(operational.get("monthly_path"))) != nino.monthly_path:
        raise ValueError("nino12 monthly climatology path must preserve the operational path")
    nino_names = {nino.daily_path.name.lower(), nino.monthly_path.name.lower()}
    for domain_id in ("pacific_context", "humboldt_coastal"):
        contextual = specs[domain_id]
        if contextual.daily_path.name.lower() in nino_names or contextual.monthly_path.name.lower() in nino_names:
            raise ValueError(f"Contextual climatology {domain_id!r} points to a nino12 file")
    return specs
