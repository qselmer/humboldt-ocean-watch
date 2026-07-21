"""Tests for univariate event intensity, development, and severity metrics."""

import json

import numpy as np
import pandas as pd

from src.event_metrics import (
    build_event_summary,
    calculate_event_metrics,
    severity_category,
)
from src.export_utils import dumps_json_safe


def _event_frame() -> pd.DataFrame:
    dates = pd.date_range("2024-01-01", periods=5)
    intensity = np.array([0.5, 1.5, 3.5, 2.5, 0.5])
    threshold = np.full(5, 20.0)
    return pd.DataFrame(
        {
            "date": dates,
            "value": threshold + intensity,
            "threshold": threshold,
            "intensity": intensity,
            "exceeds_threshold": True,
            "observed_interruption": False,
            "valid_coverage": 0.9,
        }
    )


def test_event_intensity_and_peak_metrics() -> None:
    bundle = calculate_event_metrics(
        _event_frame(), unit="degrees_Celsius", severity_bands={"strong": 1, "extreme": 3}
    )
    values = bundle.values()
    assert bundle.peak_date == pd.Timestamp("2024-01-03")
    assert values["mean_intensity"] == 1.7
    assert values["maximum_intensity"] == 3.5
    assert values["minimum_intensity"] == 0.5
    assert values["cumulative_intensity"] == 8.5
    assert values["peak_value"] == 23.5
    assert values["peak_threshold"] == 20.0
    assert values["peak_exceedance"] == 3.5
    assert bundle.severity_class == "extreme thermal exceedance"


def test_event_development_uses_actual_elapsed_days() -> None:
    frame = _event_frame().iloc[[0, 2, 4]].copy()
    bundle = calculate_event_metrics(
        frame, unit="degrees_Celsius", severity_bands={"strong": 1, "extreme": 3}
    )
    values = bundle.values()
    assert values["onset_rate"] == 1.5
    assert values["decline_rate"] == 1.5
    assert values["time_to_peak_days"] == 2
    assert values["time_from_peak_to_end_days"] == 2
    assert values["maximum_daily_intensification"] == 1.5
    assert values["maximum_daily_relaxation"] == 1.5
    assert values["number_of_turning_points"] == 1


def test_boundary_peak_rates_are_not_calculated() -> None:
    frame = _event_frame().iloc[:2].copy()
    frame["intensity"] = [2.0, 1.0]
    frame["value"] = frame.threshold + frame.intensity
    bundle = calculate_event_metrics(
        frame, unit="degrees_Celsius", severity_bands={"strong": 1, "extreme": 3}
    )
    assert bundle.results["onset_rate"].status == "not_calculated"
    assert np.isnan(bundle.results["onset_rate"].value)
    assert bundle.results["decline_rate"].status == "valid"


def test_neutral_severity_categories_and_configuration_validation() -> None:
    bands = {"strong": 1.0, "extreme": 2.0}
    assert severity_category(0.9, bands) == "moderate thermal exceedance"
    assert severity_category(1.0, bands) == "strong thermal exceedance"
    assert severity_category(2.0, bands) == "extreme thermal exceedance"


def test_event_summary_is_strict_json_compatible() -> None:
    bundle = calculate_event_metrics(
        _event_frame(), unit="degrees_Celsius", severity_bands={"strong": 1, "extreme": 3}
    )
    events = pd.DataFrame(
        [{
            "event_id": "sst-E0001",
            "start_date": np.datetime64("2024-01-01"),
            "end_date": pd.Timestamp("2024-01-05"),
            "peak_date": bundle.peak_date,
            "status": "valid",
        }]
    )
    flags = pd.DataFrame({"status": ["valid", "not_calculated"]})
    summary = build_event_summary(
        events,
        flags,
        {"sst-E0001": bundle.results},
        source_variable="regional_mean_sst",
        source_metric="sst.weighted_mean",
        threshold_type="fixed",
        threshold_description="fixed threshold 20",
        direction="above",
        climatology_method="daily_smoothed",
        data_mode="Copernicus cached data",
        temporal_discontinuities=[{"next_date": pd.Timestamp("2024-01-04")}],
        input_warnings=["sorted source dates"],
        filtered_candidate_count=np.int64(1),
        configuration={"minimum_duration_days": np.int64(5)},
    )
    parsed = json.loads(dumps_json_safe(summary))
    assert parsed["number_of_events"] == 1
    assert parsed["dates_not_calculated"] == 1
    assert parsed["official_enso_classification"] == "not provided"
    assert parsed["input_warnings"] == ["sorted source dates"]
    assert parsed["event_date_ranges"][0]["peak_date"].startswith("2024-01-03")
