"""Tests for the standard analytics result model."""

import json

import numpy as np
import pytest

from src.export_utils import dumps_json_safe
from src.metric_result import MetricResult, not_calculated


def test_metric_result_validates_structural_fields() -> None:
    with pytest.raises(ValueError, match="non-empty"):
        MetricResult(metric="")
    with pytest.raises(ValueError, match="status"):
        MetricResult(metric="sst.mean", status="unknown")  # type: ignore[arg-type]
    with pytest.raises(ValueError, match="coverage"):
        MetricResult(metric="sst.mean", valid_coverage=1.2)


def test_not_calculated_uses_nan_internally_and_null_in_json() -> None:
    result = not_calculated(
        "anomaly.centroid", "No cells over threshold", unit="degrees_east",
        family="centroid", valid_coverage=0.95,
    )
    assert np.isnan(result.value)
    assert result.status == "not_calculated"
    assert result.to_json_compatible()["value"] is None


def test_nested_metric_result_serialization_is_json_compatible() -> None:
    result = MetricResult(
        metric="sst.weighted_mean",
        value=np.float32(24.5),
        n_observations=np.int64(4),
        valid_coverage=np.float64(1.0),
        unit="degrees_Celsius",
        family="state",
        metadata={"nested": {"flag": np.bool_(True), "missing": np.nan}},
    )
    parsed = json.loads(dumps_json_safe(result.to_json_compatible()))
    assert parsed["value"] == 24.5
    assert parsed["n_observations"] == 4
    assert parsed["metadata"] == {"nested": {"flag": True, "missing": None}}
