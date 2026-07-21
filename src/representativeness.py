"""Daily spatial representativeness metrics for regional SST anomalies.

Representativeness describes whether a regional weighted mean adequately
summarizes the same day's spatial anomaly field. It is not an event detector,
and patches are never linked between dates.

The signal-to-heterogeneity ratio is ``abs(weighted mean) / (weighted SD +
epsilon)``. The standardized mean-median difference is ``abs(mean - median) /
(IQR + epsilon)``. Coexistence is ``2 * min(positive fraction, negative
fraction)``. Spatial compensation is ``1 - abs(weighted mean) / weighted mean
absolute anomaly`` and is clipped to [0, 1].
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Literal, Mapping

import numpy as np
import pandas as pd
import xarray as xr

from src.export_utils import to_json_compatible
from src.quality_control import coordinate_names, cosine_latitude_weights
from src.series_bank import weighted_median, weighted_quantile
from src.spatial_adjacency import Connectivity, label_patches

REPRESENTATIVENESS_COLUMNS = [
    "date",
    "class",
    "status",
    "triggered_rule",
    "evidence_score",
    "valid_coverage",
    "weighted_mean_anomaly",
    "weighted_median_anomaly",
    "spatial_standard_deviation",
    "weighted_iqr",
    "positive_fraction",
    "negative_fraction",
    "neutral_fraction",
    "sign_coherence",
    "signal_heterogeneity_ratio",
    "mean_median_difference",
    "dominant_patch_fraction",
    "patch_density",
    "coexistence_index",
    "spatial_compensation_index",
    "climatology_method",
    "data_mode",
]

CLASS_NAMES = (
    "insufficient_coverage",
    "strong_coherent",
    "strong_heterogeneous",
    "compensated_mixed",
    "patch_distributed",
    "weak_homogeneous",
)

ClassificationStatus = Literal["valid", "warning", "not_calculated", "invalid"]


@dataclass(frozen=True)
class RepresentativenessThresholds:
    """Configuration-backed thresholds for ordered classification rules."""

    minimum_coverage: float
    strong_signal_c: float
    weak_signal_c: float
    neutral_band_c: float
    coherent_sign_fraction: float
    heterogeneous_sd_c: float
    minimum_signal_heterogeneity_ratio: float
    dominant_patch_fraction: float
    patch_density_threshold: float
    coexistence_fraction: float
    epsilon: float

    def __post_init__(self) -> None:
        fractions = {
            "minimum_coverage": self.minimum_coverage,
            "coherent_sign_fraction": self.coherent_sign_fraction,
            "dominant_patch_fraction": self.dominant_patch_fraction,
        }
        for name, value in fractions.items():
            if not 0.0 <= value <= 1.0:
                raise ValueError(f"{name} must be between 0 and 1")
        if self.minimum_coverage <= 0:
            raise ValueError("minimum_coverage must be positive")
        if self.coherent_sign_fraction <= 0:
            raise ValueError("coherent_sign_fraction must be positive")
        if not 0.0 < self.coexistence_fraction <= 0.5:
            raise ValueError("coexistence_fraction must be greater than 0 and at most 0.5")
        non_negative = {
            "strong_signal_c": self.strong_signal_c,
            "weak_signal_c": self.weak_signal_c,
            "neutral_band_c": self.neutral_band_c,
            "heterogeneous_sd_c": self.heterogeneous_sd_c,
            "minimum_signal_heterogeneity_ratio": self.minimum_signal_heterogeneity_ratio,
            "patch_density_threshold": self.patch_density_threshold,
        }
        for name, value in non_negative.items():
            if value < 0:
                raise ValueError(f"{name} cannot be negative")
        if self.strong_signal_c < self.weak_signal_c:
            raise ValueError("strong_signal_c must be at least weak_signal_c")
        if self.strong_signal_c <= 0:
            raise ValueError("strong_signal_c must be positive")
        if self.heterogeneous_sd_c <= 0:
            raise ValueError("heterogeneous_sd_c must be positive")
        if self.minimum_signal_heterogeneity_ratio <= 0:
            raise ValueError("minimum_signal_heterogeneity_ratio must be positive")
        if self.patch_density_threshold <= 0:
            raise ValueError("patch_density_threshold must be positive")
        if self.epsilon <= 0:
            raise ValueError("epsilon must be positive")

    @classmethod
    def from_mapping(cls, values: Mapping[str, Any]) -> "RepresentativenessThresholds":
        required = {
            "minimum_coverage",
            "strong_signal_c",
            "weak_signal_c",
            "neutral_band_c",
            "coherent_sign_fraction",
            "heterogeneous_sd_c",
            "minimum_signal_heterogeneity_ratio",
            "dominant_patch_fraction",
            "patch_density_threshold",
            "coexistence_fraction",
            "epsilon",
        }
        missing = required - set(values)
        if missing:
            raise ValueError(f"Representativeness configuration is missing: {sorted(missing)}")
        return cls(**{name: float(values[name]) for name in required})

    def to_dict(self) -> dict[str, float]:
        return {
            name: float(getattr(self, name))
            for name in self.__dataclass_fields__
        }


@dataclass(frozen=True)
class RepresentativenessClassification:
    """Ordered classification plus transparent evidence and inputs."""

    classification: str
    status: ClassificationStatus
    triggered_rule: str
    input_metrics: dict[str, float]
    threshold_values: dict[str, float]
    evidence_score: float
    valid_coverage: float

    def __post_init__(self) -> None:
        if self.classification not in {*CLASS_NAMES, "unclassified"}:
            raise ValueError(f"Unknown representativeness class: {self.classification}")
        if not 0.0 <= self.evidence_score <= 1.0:
            raise ValueError("Evidence score must be between 0 and 1")

    def to_dict(self) -> dict[str, Any]:
        return {
            "class": self.classification,
            "status": self.status,
            "triggered_rule": self.triggered_rule,
            "input_metrics": self.input_metrics,
            "threshold_values": self.threshold_values,
            "evidence_score": self.evidence_score,
            "valid_coverage": self.valid_coverage,
        }

    def to_json_compatible(self) -> dict[str, Any]:
        converted = to_json_compatible(self.to_dict())
        assert isinstance(converted, dict)
        return converted


def _bounded(value: float) -> float:
    return float(np.clip(value, 0.0, 1.0)) if np.isfinite(value) else 0.0


def _mean_score(*values: float) -> float:
    return _bounded(float(np.mean([_bounded(value) for value in values])))


def classify_representativeness(
    metrics: Mapping[str, Any],
    thresholds: RepresentativenessThresholds,
) -> RepresentativenessClassification:
    """Apply ordered rules; ambiguous cases remain explicitly unclassified."""
    numeric = {
        name: float(value) if value is not None and np.isscalar(value) else np.nan
        for name, value in metrics.items()
    }
    coverage = numeric.get("weighted_valid_coverage", np.nan)
    mean = numeric.get("weighted_mean_anomaly", np.nan)
    standard_deviation = numeric.get("spatial_standard_deviation", np.nan)
    coherence = numeric.get("sign_coherence", np.nan)
    ratio = numeric.get("signal_heterogeneity_ratio", np.nan)
    positive = numeric.get("positive_fraction", np.nan)
    negative = numeric.get("negative_fraction", np.nan)
    neutral = numeric.get("neutral_fraction", np.nan)
    dominant = numeric.get("dominant_patch_fraction", np.nan)
    patch_density = numeric.get("patch_density", np.nan)
    compensation = numeric.get("spatial_compensation_index", np.nan)
    threshold_values = thresholds.to_dict()

    def result(
        classification: str,
        status: ClassificationStatus,
        rule: str,
        evidence: float,
    ) -> RepresentativenessClassification:
        return RepresentativenessClassification(
            classification=classification,
            status=status,
            triggered_rule=rule,
            input_metrics=numeric,
            threshold_values=threshold_values,
            evidence_score=_bounded(evidence),
            valid_coverage=coverage,
        )

    if not np.isfinite(coverage) or coverage < thresholds.minimum_coverage:
        evidence = 0.0 if not np.isfinite(coverage) else 1.0 - coverage / thresholds.minimum_coverage
        return result(
            "insufficient_coverage",
            "not_calculated",
            "coverage_below_minimum",
            evidence,
        )

    core = (mean, standard_deviation, coherence, ratio, positive, negative, neutral, compensation)
    if not all(np.isfinite(value) for value in core):
        return result(
            "unclassified",
            "not_calculated",
            "required_input_metric_missing",
            0.0,
        )

    absolute_mean = abs(mean)
    if (
        absolute_mean >= thresholds.strong_signal_c
        and coherence >= thresholds.coherent_sign_fraction
        and standard_deviation < thresholds.heterogeneous_sd_c
        and ratio >= thresholds.minimum_signal_heterogeneity_ratio
    ):
        evidence = _mean_score(
            absolute_mean / thresholds.strong_signal_c,
            coherence / thresholds.coherent_sign_fraction,
            ratio / thresholds.minimum_signal_heterogeneity_ratio,
            1.0 - standard_deviation / thresholds.heterogeneous_sd_c,
        )
        return result("strong_coherent", "valid", "strong_signal_and_coherent_sign", evidence)

    if (
        absolute_mean >= thresholds.strong_signal_c
        and (
            standard_deviation >= thresholds.heterogeneous_sd_c
            or ratio < thresholds.minimum_signal_heterogeneity_ratio
        )
    ):
        heterogeneity_evidence = max(
            standard_deviation / thresholds.heterogeneous_sd_c,
            thresholds.minimum_signal_heterogeneity_ratio / (ratio + thresholds.epsilon),
        )
        evidence = _mean_score(
            absolute_mean / thresholds.strong_signal_c,
            heterogeneity_evidence,
        )
        return result(
            "strong_heterogeneous",
            "valid",
            "strong_signal_with_high_heterogeneity",
            evidence,
        )

    if (
        absolute_mean <= thresholds.weak_signal_c
        and positive >= thresholds.coexistence_fraction
        and negative >= thresholds.coexistence_fraction
    ):
        evidence = _mean_score(
            1.0 - absolute_mean / max(thresholds.weak_signal_c, thresholds.epsilon),
            positive / thresholds.coexistence_fraction,
            negative / thresholds.coexistence_fraction,
            compensation,
        )
        return result(
            "compensated_mixed",
            "valid",
            "opposing_signs_with_compensated_mean",
            evidence,
        )

    if (
        np.isfinite(dominant)
        and np.isfinite(patch_density)
        and dominant <= thresholds.dominant_patch_fraction
        and patch_density >= thresholds.patch_density_threshold
        and positive + negative >= thresholds.coexistence_fraction
    ):
        dominance_evidence = 1.0 - dominant / max(thresholds.dominant_patch_fraction, thresholds.epsilon)
        evidence = _mean_score(
            dominance_evidence,
            patch_density / thresholds.patch_density_threshold,
            (positive + negative) / thresholds.coexistence_fraction,
        )
        return result(
            "patch_distributed",
            "valid",
            "distributed_non_neutral_patches",
            evidence,
        )

    if (
        absolute_mean <= thresholds.weak_signal_c
        and neutral >= thresholds.coherent_sign_fraction
        and standard_deviation < thresholds.heterogeneous_sd_c
    ):
        evidence = _mean_score(
            1.0 - absolute_mean / max(thresholds.weak_signal_c, thresholds.epsilon),
            neutral / thresholds.coherent_sign_fraction,
            1.0 - standard_deviation / thresholds.heterogeneous_sd_c,
        )
        return result(
            "weak_homogeneous",
            "valid",
            "weak_signal_and_neutral_homogeneity",
            evidence,
        )

    return result(
        "unclassified",
        "not_calculated",
        "no_ordered_rule_satisfied",
        0.0,
    )


def _patch_metrics(
    positive_mask: np.ndarray,
    negative_mask: np.ndarray,
    weights: np.ndarray,
    valid_mask: np.ndarray,
    *,
    connectivity: Connectivity,
    minimum_patch_cells: int,
) -> tuple[float, float]:
    """Return same-day sign-specific dominant fraction and patch density."""
    patch_areas: list[float] = []
    patch_count = 0
    for mask in (positive_mask, negative_mask):
        labels, count = label_patches(
            mask & valid_mask,
            connectivity=connectivity,
            minimum_patch_cells=minimum_patch_cells,
        )
        patch_count += count
        patch_areas.extend(
            float(weights[labels == label].sum())
            for label in range(1, count + 1)
        )
    total_patch_area = float(np.sum(patch_areas))
    dominant_fraction = (
        float(np.max(patch_areas) / total_patch_area)
        if patch_areas and total_patch_area > 0
        else np.nan
    )
    valid_area = float(weights[valid_mask].sum())
    density = float(patch_count / valid_area) if valid_area > 0 else np.nan
    return dominant_fraction, density


def calculate_daily_metrics(
    anomaly: xr.DataArray,
    thresholds: RepresentativenessThresholds,
    *,
    weights: xr.DataArray | np.ndarray | None = None,
    connectivity: Connectivity = "queen",
    minimum_patch_cells: int = 4,
) -> dict[str, float]:
    """Calculate area-weighted representativeness inputs for one anomaly date."""
    if not isinstance(anomaly, xr.DataArray):
        raise TypeError("Representativeness input must be an xarray.DataArray")
    latitude, longitude = coordinate_names(anomaly)
    if "time" in anomaly.dims or set(anomaly.dims) != {latitude, longitude}:
        raise ValueError("Representativeness input must be two-dimensional after date selection")
    if not np.issubdtype(anomaly.dtype, np.number):
        raise TypeError("SST anomaly input must be numeric")
    anomaly = anomaly.transpose(latitude, longitude)
    values = np.asarray(anomaly.values, dtype=float)
    if np.isinf(values).any():
        raise ValueError("SST anomaly input contains infinite values")
    if weights is None:
        weight_values = np.asarray(cosine_latitude_weights(anomaly).values, dtype=float)
    elif isinstance(weights, xr.DataArray):
        weight_values = np.asarray(
            weights.broadcast_like(anomaly).transpose(latitude, longitude).values,
            dtype=float,
        )
    else:
        weight_values = np.asarray(weights, dtype=float)
    if weight_values.shape != values.shape:
        raise ValueError("Representativeness weights are incompatible with the anomaly field")
    if not np.isfinite(weight_values).all() or bool((weight_values < 0).any()):
        raise ValueError("Representativeness weights must be finite and non-negative")
    total_weight = float(weight_values.sum())
    if total_weight <= 0:
        raise ValueError("Representativeness weights have a zero denominator")
    valid = np.isfinite(values) & (weight_values > 0)
    valid_weight = float(weight_values[valid].sum())
    coverage = valid_weight / total_weight
    empty = {
        "weighted_valid_coverage": coverage,
        "weighted_mean_anomaly": np.nan,
        "weighted_median_anomaly": np.nan,
        "spatial_standard_deviation": np.nan,
        "weighted_iqr": np.nan,
        "positive_fraction": np.nan,
        "negative_fraction": np.nan,
        "neutral_fraction": np.nan,
        "sign_coherence": np.nan,
        "signal_heterogeneity_ratio": np.nan,
        "mean_median_difference": np.nan,
        "dominant_patch_fraction": np.nan,
        "patch_density": np.nan,
        "warm_area_fraction": np.nan,
        "cold_area_fraction": np.nan,
        "coexistence_index": np.nan,
        "spatial_compensation_index": np.nan,
    }
    if valid_weight <= 0:
        return empty

    data = values[valid]
    area = weight_values[valid]
    mean = float(np.sum(data * area) / valid_weight)
    median = weighted_median(data, area)
    p25 = weighted_quantile(data, area, 0.25)
    p75 = weighted_quantile(data, area, 0.75)
    iqr = float(p75 - p25)
    standard_deviation = float(np.sqrt(np.sum(area * (data - mean) ** 2) / valid_weight))

    positive_mask = valid & (values > thresholds.neutral_band_c)
    negative_mask = valid & (values < -thresholds.neutral_band_c)
    neutral_mask = valid & ~(positive_mask | negative_mask)
    positive_fraction = float(weight_values[positive_mask].sum() / valid_weight)
    negative_fraction = float(weight_values[negative_mask].sum() / valid_weight)
    neutral_fraction = float(weight_values[neutral_mask].sum() / valid_weight)
    signs = np.where(data > thresholds.neutral_band_c, 1.0, np.where(data < -thresholds.neutral_band_c, -1.0, 0.0))
    sign_coherence = float(abs(np.sum(area * signs) / valid_weight))
    weighted_mean_absolute = float(np.sum(area * np.abs(data)) / valid_weight)
    compensation = (
        _bounded(1.0 - abs(mean) / weighted_mean_absolute)
        if weighted_mean_absolute > thresholds.epsilon
        else 0.0
    )
    dominant, density = _patch_metrics(
        positive_mask,
        negative_mask,
        weight_values,
        valid,
        connectivity=connectivity,
        minimum_patch_cells=minimum_patch_cells,
    )

    return {
        "weighted_valid_coverage": coverage,
        "weighted_mean_anomaly": mean,
        "weighted_median_anomaly": median,
        "spatial_standard_deviation": standard_deviation,
        "weighted_iqr": iqr,
        "positive_fraction": positive_fraction,
        "negative_fraction": negative_fraction,
        "neutral_fraction": neutral_fraction,
        "sign_coherence": sign_coherence,
        "signal_heterogeneity_ratio": float(abs(mean) / (standard_deviation + thresholds.epsilon)),
        "mean_median_difference": float(abs(mean - median) / (iqr + thresholds.epsilon)),
        "dominant_patch_fraction": dominant,
        "patch_density": density,
        "warm_area_fraction": float(weight_values[valid & (values >= thresholds.strong_signal_c)].sum() / valid_weight),
        "cold_area_fraction": float(weight_values[valid & (values <= -thresholds.strong_signal_c)].sum() / valid_weight),
        "coexistence_index": _bounded(2.0 * min(positive_fraction, negative_fraction)),
        "spatial_compensation_index": compensation,
    }


def build_representativeness_table(
    anomaly: xr.DataArray,
    thresholds: RepresentativenessThresholds,
    *,
    climatology_method: str | None,
    data_mode: str,
    weights: xr.DataArray | None = None,
    connectivity: Connectivity = "queen",
    minimum_patch_cells: int = 4,
) -> pd.DataFrame:
    """Build the required one-row-per-date representativeness table."""
    if "time" not in anomaly.coords or "time" not in anomaly.dims:
        raise ValueError("Representativeness anomaly cube requires a time coordinate")
    rows: list[dict[str, Any]] = []
    for index, date in enumerate(pd.DatetimeIndex(anomaly.time.values)):
        daily_weights = weights
        if daily_weights is not None and "time" in daily_weights.dims:
            daily_weights = daily_weights.isel(time=index, drop=True)
        metrics = calculate_daily_metrics(
            anomaly.isel(time=index, drop=True),
            thresholds,
            weights=daily_weights,
            connectivity=connectivity,
            minimum_patch_cells=minimum_patch_cells,
        )
        classification = classify_representativeness(metrics, thresholds)
        rows.append({
            "date": date,
            "class": classification.classification,
            "status": classification.status,
            "triggered_rule": classification.triggered_rule,
            "evidence_score": classification.evidence_score,
            "valid_coverage": metrics["weighted_valid_coverage"],
            "weighted_mean_anomaly": metrics["weighted_mean_anomaly"],
            "weighted_median_anomaly": metrics["weighted_median_anomaly"],
            "spatial_standard_deviation": metrics["spatial_standard_deviation"],
            "weighted_iqr": metrics["weighted_iqr"],
            "positive_fraction": metrics["positive_fraction"],
            "negative_fraction": metrics["negative_fraction"],
            "neutral_fraction": metrics["neutral_fraction"],
            "sign_coherence": metrics["sign_coherence"],
            "signal_heterogeneity_ratio": metrics["signal_heterogeneity_ratio"],
            "mean_median_difference": metrics["mean_median_difference"],
            "dominant_patch_fraction": metrics["dominant_patch_fraction"],
            "patch_density": metrics["patch_density"],
            "coexistence_index": metrics["coexistence_index"],
            "spatial_compensation_index": metrics["spatial_compensation_index"],
            "climatology_method": climatology_method,
            "data_mode": data_mode,
        })
    table = pd.DataFrame(rows, columns=REPRESENTATIVENESS_COLUMNS)
    table["date"] = pd.to_datetime(table["date"])
    return table.sort_values("date", ignore_index=True)
