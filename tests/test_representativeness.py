"""Deterministic representativeness metrics and ordered classification tests."""

from __future__ import annotations

import json

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.export_utils import dumps_json_safe
from src.representativeness import (
    REPRESENTATIVENESS_COLUMNS,
    RepresentativenessThresholds,
    build_representativeness_table,
    calculate_daily_metrics,
    classify_representativeness,
)


@pytest.fixture
def thresholds() -> RepresentativenessThresholds:
    return RepresentativenessThresholds(
        minimum_coverage=0.80,
        strong_signal_c=1.0,
        weak_signal_c=0.30,
        neutral_band_c=0.25,
        coherent_sign_fraction=0.75,
        heterogeneous_sd_c=0.75,
        minimum_signal_heterogeneity_ratio=1.0,
        dominant_patch_fraction=0.60,
        patch_density_threshold=0.02,
        coexistence_fraction=0.25,
        epsilon=1.0e-12,
    )


def _field(values: np.ndarray) -> xr.DataArray:
    values = np.asarray(values, dtype=float)
    return xr.DataArray(
        values,
        dims=("latitude", "longitude"),
        coords={
            "latitude": np.arange(values.shape[0], dtype=float),
            "longitude": np.arange(values.shape[1], dtype=float),
        },
        attrs={"units": "degrees_Celsius"},
        name="anomaly",
    )


def _calculate(values: np.ndarray, thresholds: RepresentativenessThresholds):
    field = _field(values)
    metrics = calculate_daily_metrics(
        field,
        thresholds,
        weights=np.ones(field.shape),
        connectivity="queen",
        minimum_patch_cells=1,
    )
    return metrics, classify_representativeness(metrics, thresholds)


def _base_metrics(**updates) -> dict[str, float]:
    metrics = {
        "weighted_valid_coverage": 1.0,
        "weighted_mean_anomaly": 0.5,
        "weighted_median_anomaly": 0.5,
        "spatial_standard_deviation": 0.2,
        "weighted_iqr": 0.2,
        "positive_fraction": 1.0,
        "negative_fraction": 0.0,
        "neutral_fraction": 0.0,
        "sign_coherence": 1.0,
        "signal_heterogeneity_ratio": 2.5,
        "mean_median_difference": 0.0,
        "dominant_patch_fraction": 1.0,
        "patch_density": 0.01,
        "warm_area_fraction": 0.0,
        "cold_area_fraction": 0.0,
        "coexistence_index": 0.0,
        "spatial_compensation_index": 0.0,
    }
    metrics.update(updates)
    return metrics


def test_insufficient_coverage(thresholds: RepresentativenessThresholds) -> None:
    values = np.array([[1.2, np.nan], [np.nan, np.nan]])
    metrics, classification = _calculate(values, thresholds)
    assert metrics["weighted_valid_coverage"] == pytest.approx(0.25)
    assert classification.classification == "insufficient_coverage"
    assert classification.status == "not_calculated"
    assert classification.triggered_rule == "coverage_below_minimum"


def test_strong_coherent_warming(thresholds: RepresentativenessThresholds) -> None:
    metrics, classification = _calculate(np.full((4, 4), 1.2), thresholds)
    assert metrics["sign_coherence"] == pytest.approx(1.0)
    assert metrics["positive_fraction"] == pytest.approx(1.0)
    assert metrics["warm_area_fraction"] == pytest.approx(1.0)
    assert classification.classification == "strong_coherent"


def test_strong_heterogeneous_warming(thresholds: RepresentativenessThresholds) -> None:
    values = np.array([[0.3, 0.3], [2.5, 2.5]])
    metrics, classification = _calculate(values, thresholds)
    assert metrics["weighted_mean_anomaly"] == pytest.approx(1.4)
    assert metrics["spatial_standard_deviation"] > thresholds.heterogeneous_sd_c
    assert classification.classification == "strong_heterogeneous"


def test_compensated_mixed_warm_and_cold(thresholds: RepresentativenessThresholds) -> None:
    metrics, classification = _calculate(np.array([[-2.0, 2.0], [-2.0, 2.0]]), thresholds)
    assert metrics["positive_fraction"] == pytest.approx(0.5)
    assert metrics["negative_fraction"] == pytest.approx(0.5)
    assert metrics["coexistence_index"] == pytest.approx(1.0)
    assert metrics["spatial_compensation_index"] == pytest.approx(1.0)
    assert classification.classification == "compensated_mixed"


def test_multiple_distributed_patches(thresholds: RepresentativenessThresholds) -> None:
    values = np.zeros((12, 12), dtype=float)
    for row in (0, 4, 8):
        for column in (0, 4, 8):
            values[row : row + 2, column : column + 2] = 0.8
    field = _field(values)
    metrics = calculate_daily_metrics(
        field,
        thresholds,
        weights=np.ones(field.shape),
        connectivity="queen",
        minimum_patch_cells=4,
    )
    classification = classify_representativeness(metrics, thresholds)
    assert metrics["positive_fraction"] == pytest.approx(0.25)
    assert metrics["dominant_patch_fraction"] == pytest.approx(1 / 9)
    assert metrics["patch_density"] == pytest.approx(9 / 144)
    assert classification.classification == "patch_distributed"


def test_weak_homogeneous_conditions(thresholds: RepresentativenessThresholds) -> None:
    metrics, classification = _calculate(np.full((4, 4), 0.1), thresholds)
    assert metrics["neutral_fraction"] == pytest.approx(1.0)
    assert classification.classification == "weak_homogeneous"


def test_ambiguous_classification_is_not_hidden(thresholds: RepresentativenessThresholds) -> None:
    metrics, classification = _calculate(np.full((4, 4), 0.5), thresholds)
    assert metrics["weighted_mean_anomaly"] == pytest.approx(0.5)
    assert classification.classification == "unclassified"
    assert classification.status == "not_calculated"
    assert classification.triggered_rule == "no_ordered_rule_satisfied"


def test_threshold_boundaries_are_deterministic(thresholds: RepresentativenessThresholds) -> None:
    coherent = classify_representativeness(_base_metrics(
        weighted_valid_coverage=thresholds.minimum_coverage,
        weighted_mean_anomaly=thresholds.strong_signal_c,
        sign_coherence=thresholds.coherent_sign_fraction,
        spatial_standard_deviation=thresholds.heterogeneous_sd_c - 0.01,
        signal_heterogeneity_ratio=thresholds.minimum_signal_heterogeneity_ratio,
    ), thresholds)
    assert coherent.classification == "strong_coherent"

    heterogeneous = classify_representativeness(_base_metrics(
        weighted_mean_anomaly=thresholds.strong_signal_c,
        spatial_standard_deviation=thresholds.heterogeneous_sd_c,
        sign_coherence=thresholds.coherent_sign_fraction,
        signal_heterogeneity_ratio=thresholds.minimum_signal_heterogeneity_ratio,
    ), thresholds)
    assert heterogeneous.classification == "strong_heterogeneous"


def test_classification_rule_order_prefers_strong_coherent(thresholds: RepresentativenessThresholds) -> None:
    classification = classify_representativeness(_base_metrics(
        weighted_mean_anomaly=1.2,
        spatial_standard_deviation=0.2,
        sign_coherence=0.9,
        signal_heterogeneity_ratio=2.0,
        positive_fraction=0.30,
        negative_fraction=0.25,
        neutral_fraction=0.45,
        spatial_compensation_index=0.8,
        dominant_patch_fraction=0.2,
        patch_density=0.1,
    ), thresholds)
    assert classification.classification == "strong_coherent"
    assert classification.triggered_rule == "strong_signal_and_coherent_sign"


def test_evidence_scores_are_bounded(thresholds: RepresentativenessThresholds) -> None:
    cases = [
        _base_metrics(weighted_valid_coverage=0.1),
        _base_metrics(weighted_mean_anomaly=2.0, spatial_standard_deviation=0.1),
        _base_metrics(weighted_mean_anomaly=0.0, positive_fraction=0.5, negative_fraction=0.5, spatial_compensation_index=1.0),
        _base_metrics(weighted_mean_anomaly=0.0, neutral_fraction=1.0, positive_fraction=0.0, spatial_standard_deviation=0.0),
    ]
    for metrics in cases:
        score = classify_representativeness(metrics, thresholds).evidence_score
        assert 0.0 <= score <= 1.0


def test_missing_patch_metrics_remain_explicit(thresholds: RepresentativenessThresholds) -> None:
    classification = classify_representativeness(_base_metrics(
        dominant_patch_fraction=np.nan,
        patch_density=np.nan,
    ), thresholds)
    assert classification.classification == "unclassified"
    assert classification.status == "not_calculated"


def test_triggered_rule_and_numpy_inputs_are_json_compatible(thresholds: RepresentativenessThresholds) -> None:
    metrics = _base_metrics(
        weighted_mean_anomaly=np.float64(1.2),
        sign_coherence=np.float32(0.9),
        spatial_standard_deviation=np.float64(0.2),
    )
    classification = classify_representativeness(metrics, thresholds)
    payload = json.loads(dumps_json_safe(classification.to_dict()))
    assert payload["triggered_rule"] == "strong_signal_and_coherent_sign"
    assert isinstance(payload["input_metrics"]["weighted_mean_anomaly"], float)
    assert 0.0 <= payload["evidence_score"] <= 1.0


def test_table_schema_and_metadata(thresholds: RepresentativenessThresholds) -> None:
    cube = xr.concat(
        [
            _field(np.full((3, 3), 1.2)).expand_dims(time=[pd.Timestamp("2020-01-01")]),
            _field(np.full((3, 3), 0.1)).expand_dims(time=[pd.Timestamp("2020-01-02")]),
        ],
        dim="time",
    )
    table = build_representativeness_table(
        cube,
        thresholds,
        climatology_method="daily_smoothed",
        data_mode="Synthetic demonstration data",
        connectivity="queen",
        minimum_patch_cells=1,
    )
    assert list(table.columns) == REPRESENTATIVENESS_COLUMNS
    assert table.date.nunique() == 2
    assert table["class"].tolist() == ["strong_coherent", "weak_homogeneous"]
    assert set(table.climatology_method) == {"daily_smoothed"}
