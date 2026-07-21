"""Versioned, language-neutral schema primitives for scientific briefs."""

from __future__ import annotations

from typing import Any


SCHEMA_VERSION = "1.0.0"

TOP_LEVEL_SECTIONS = (
    "schema_version",
    "product",
    "analysis",
    "provenance",
    "availability",
    "quality",
    "climatology",
    "regional_state",
    "recent_evolution",
    "spatial_structure",
    "representativeness",
    "univariate_event",
    "daily_patches",
    "spatiotemporal_activity",
    "fact_registry",
    "quality_flags",
    "limitations",
    "generation_constraints",
)

CONTEXT_STATUSES = frozenset(
    {"valid", "warning", "insufficient", "invalid", "unavailable", "not_calculated"}
)
VALIDATION_RESULTS = frozenset({"passed", "warning", "failed"})
FLAG_SEVERITIES = frozenset({"info", "warning", "error"})
REPRESENTATIVENESS_CLASSES = frozenset(
    {
        "insufficient_coverage",
        "strong_coherent",
        "strong_heterogeneous",
        "compensated_mixed",
        "patch_distributed",
        "weak_homogeneous",
        "unclassified",
    }
)

CONTROLLED_UNITS = frozenset(
    {
        "degC",
        "dimensionless",
        "fraction",
        "fraction_change",
        "percent",
        "degree_latitude",
        "degree_longitude",
        "km",
        "km2",
        "km_per_day",
        "km2_per_day",
        "km2_day",
        "degC_km2_day",
        "degC_day",
        "degC_per_degree",
        "day",
        "count",
        "index",
        "slope_degC_per_day",
    }
)


def new_context(schema_version: str = SCHEMA_VERSION) -> dict[str, Any]:
    """Return all required sections in deterministic schema order."""
    if schema_version != SCHEMA_VERSION:
        raise ValueError(
            f"Unsupported brief schema version {schema_version!r}; expected {SCHEMA_VERSION!r}"
        )
    context: dict[str, Any] = {"schema_version": schema_version}
    for section in TOP_LEVEL_SECTIONS[1:]:
        context[section] = [] if section in {"fact_registry", "quality_flags", "limitations"} else {}
    return context


def missing_top_level_sections(context: dict[str, Any]) -> list[str]:
    return [section for section in TOP_LEVEL_SECTIONS if section not in context]


def metric_reference(
    fact_id: str,
    value: Any,
    unit: str,
    status: str,
) -> dict[str, Any]:
    """Create the compact fact reference embedded in scientific sections."""
    return {"value": value, "unit": unit, "status": status, "fact_id": fact_id}
