"""Canonical metadata for Increment 3A series-bank features."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class FeatureDefinition:
    family: str
    name: str
    unit_kind: str
    requires_threshold: bool = False

    def metric_name(self, variable: str) -> str:
        return f"{variable}.{self.name}"


FEATURES: tuple[FeatureDefinition, ...] = (
    FeatureDefinition("state", "weighted_mean", "variable"),
    FeatureDefinition("state", "weighted_median", "variable"),
    FeatureDefinition("state", "spatial_minimum", "variable"),
    FeatureDefinition("state", "spatial_maximum", "variable"),
    FeatureDefinition("state", "spatial_p10", "variable"),
    FeatureDefinition("state", "spatial_p25", "variable"),
    FeatureDefinition("state", "spatial_p75", "variable"),
    FeatureDefinition("state", "spatial_p90", "variable"),
    FeatureDefinition("coverage", "valid_cell_fraction", "fraction"),
    FeatureDefinition("coverage", "weighted_valid_coverage", "fraction"),
    FeatureDefinition("heterogeneity", "weighted_standard_deviation", "variable"),
    FeatureDefinition("heterogeneity", "weighted_mad", "variable"),
    FeatureDefinition("heterogeneity", "weighted_iqr", "variable"),
    FeatureDefinition("heterogeneity", "p90_minus_p10", "variable"),
    FeatureDefinition("gradient", "latitudinal_gradient", "variable_per_degree"),
    FeatureDefinition("gradient", "longitudinal_gradient", "variable_per_degree"),
    FeatureDefinition("quality", "valid_observation_count", "count"),
    FeatureDefinition("quality", "missing_fraction", "fraction"),
    FeatureDefinition("quality", "constant_field_flag", "flag"),
)

CENTROID_FEATURES: tuple[FeatureDefinition, ...] = (
    FeatureDefinition("centroid", "warm_centroid_latitude", "degrees_north", True),
    FeatureDefinition("centroid", "warm_centroid_longitude", "degrees_east", True),
)

REQUIRED_FAMILIES = frozenset({"state", "coverage", "heterogeneity", "centroid", "gradient", "quality"})


def unit_for(variable_unit: str, kind: str) -> str:
    if kind == "variable":
        return variable_unit
    if kind == "variable_per_degree":
        return f"{variable_unit} per degree"
    return {
        "fraction": "1",
        "count": "count",
        "flag": "1",
        "degrees_north": "degrees_north",
        "degrees_east": "degrees_east",
    }[kind]
