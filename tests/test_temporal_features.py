"""Deterministic tests for the date-aware temporal feature engine."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from src.temporal_features import (
    FEATURE_FAMILIES,
    TEMPORAL_COLUMNS,
    build_temporal_feature_table,
    compute_temporal_features,
    select_date_window,
)


def _series(values: list[float] | np.ndarray, dates: pd.DatetimeIndex | None = None) -> pd.DataFrame:
    dates = dates if dates is not None else pd.date_range("2020-01-01", periods=len(values), freq="D")
    return pd.DataFrame({"date": dates, "value": values})


def _results(values: list[float] | np.ndarray, **kwargs):
    frame = _series(values, kwargs.pop("dates", None))
    results, _, _ = compute_temporal_features(
        frame,
        source_metric=kwargs.pop("source_metric", "anomaly.weighted_mean"),
        analysis_end_date=frame.date.max(),
        window=kwargs.pop("window", len(frame)),
        minimum_observations=kwargs.pop("minimum_observations", 5),
        spectral_minimum_observations=kwargs.pop("spectral_minimum_observations", 5),
        unit=kwargs.pop("unit", "degrees_Celsius"),
        **kwargs,
    )
    return {result.metric: result for result in results}


def test_date_windows_use_calendar_dates_and_support_all_and_user_period() -> None:
    frame = _series([1.0, 2.0, 3.0], pd.to_datetime(["2020-01-01", "2020-01-05", "2020-01-10"]))
    selected, start, end, days = select_date_window(frame, "2020-01-10", 7)
    assert selected.value.tolist() == [2.0, 3.0]
    assert (start, end, days) == (pd.Timestamp("2020-01-04"), pd.Timestamp("2020-01-10"), 7)
    selected_all, _, _, all_days = select_date_window(frame, "2020-01-10", "all")
    assert len(selected_all) == 3
    assert all_days == 10
    selected_user, _, _, user_days = select_date_window(
        frame, "2020-01-10", ("2020-01-03", "2020-01-06")
    )
    assert selected_user.value.tolist() == [2.0]
    assert user_days == 4


def test_state_magnitude_and_variability_families() -> None:
    results = _results([-2.0, -1.0, 0.0, 1.0, 2.0])
    assert results["mean"].value == pytest.approx(0.0)
    assert results["median"].value == pytest.approx(0.0)
    assert results["minimum"].value == -2.0
    assert results["maximum"].value == 2.0
    assert results["mean_absolute_value"].value == pytest.approx(1.2)
    assert results["rms"].value == pytest.approx(np.sqrt(2.0))
    assert results["positive_accumulation"].value == 3.0
    assert results["negative_accumulation"].value == 3.0
    assert results["variance"].value == pytest.approx(2.0)
    assert results["standard_deviation"].value == pytest.approx(np.sqrt(2.0))
    assert results["coefficient_of_variation"].status == "not_calculated"


def test_instability_sign_changes_and_turning_points() -> None:
    results = _results([-1.0, 1.0, -1.0, 1.0, -1.0])
    assert results["sign_change_count"].value == 4
    assert results["turning_point_count"].value == 3
    assert results["mean_absolute_successive_change"].value == 2.0
    assert results["rmssd"].value == 2.0


def test_temporal_gaps_are_not_bridged() -> None:
    dates = pd.to_datetime(["2020-01-01", "2020-01-02", "2020-01-04", "2020-01-05", "2020-01-06"])
    results = _results(
        [0.0, 1.0, 100.0, 101.0, 102.0], dates=dates,
        persistence_threshold=50.0, window="all",
    )
    assert results["maximum_absolute_jump"].value == 1.0
    assert results["state_transition_count"].value == 0.0
    assert results["longest_run_below_threshold"].value == 2.0
    assert results["longest_run_above_threshold"].value == 3.0
    assert results["spectral_variance"].status == "not_calculated"


def test_missing_values_reduce_coverage_without_bridging() -> None:
    results = _results([1.0, 2.0, np.nan, 4.0, 5.0], minimum_observations=4)
    assert results["mean"].n_observations == 4
    assert results["mean"].valid_coverage == pytest.approx(0.8)
    assert results["maximum_absolute_jump"].value == 1.0


def test_linear_trend_features() -> None:
    elapsed = np.arange(10, dtype=float)
    results = _results(3.0 + 2.0 * elapsed)
    assert results["ols_slope"].value == pytest.approx(2.0)
    assert results["ols_intercept"].value == pytest.approx(3.0)
    assert results["ols_r_squared"].value == pytest.approx(1.0)
    assert results["theil_sen_slope"].value == pytest.approx(2.0)
    assert results["theil_sen_ci_lower"].value == pytest.approx(2.0)
    assert results["theil_sen_ci_upper"].value == pytest.approx(2.0)
    assert results["mann_kendall_tau"].value == pytest.approx(1.0)
    assert results["mann_kendall_p_value"].value < 0.001
    assert results["initial_final_difference"].value == 18.0


def test_persistence_runs_and_autocorrelation() -> None:
    results = _results([1.0, 2.0, -1.0, -2.0, 3.0, 4.0], persistence_threshold=0.0)
    assert results["longest_run_above_threshold"].value == 2.0
    assert results["mean_run_length_above_threshold"].value == 2.0
    assert results["longest_run_below_threshold"].value == 2.0
    assert results["proportion_above_threshold"].value == pytest.approx(4 / 6)
    assert results["state_transition_count"].value == 2.0
    assert results["lag1_autocorrelation"].status == "valid"


def test_entropy_and_regular_spectral_features() -> None:
    values = np.sin(2.0 * np.pi * np.arange(60) / 10.0)
    results = _results(values, spectral_minimum_observations=30)
    assert results["shannon_entropy"].status == "valid"
    assert results["shannon_entropy"].value > 0
    assert results["spectral_variance"].value == pytest.approx(np.var(values))
    assert results["spectral_variance"].unit == "degrees_Celsius squared"
    assert results["dominant_period"].value == pytest.approx(10.0)
    assert results["low_frequency_power_fraction"].value + results["high_frequency_power_fraction"].value == pytest.approx(1.0)


def test_constant_series_and_insufficient_observations() -> None:
    constant = _results(
        [20.0] * 8, source_metric="sst.weighted_mean", unit="degrees_Celsius"
    )
    assert constant["mean"].status == "valid"
    assert constant["standard_deviation"].value == 0.0
    assert constant["coefficient_of_variation"].status == "not_calculated"
    assert constant["lag1_autocorrelation"].status == "not_calculated"
    assert constant["dominant_period"].status == "not_calculated"

    ratio_scale = _results(
        [1.0, 2.0, 3.0, 4.0, 5.0], source_metric="ratio.metric", unit="1",
        coefficient_of_variation_valid=True,
    )
    assert ratio_scale["coefficient_of_variation"].status == "valid"
    assert ratio_scale["coefficient_of_variation"].value == pytest.approx(np.std([1, 2, 3, 4, 5]) / 3)

    insufficient = _results([1.0, 2.0, 3.0, 4.0], minimum_observations=5)
    assert all(result.status == "not_calculated" for result in insufficient.values())
    assert insufficient["ols_slope"].unit == "degrees_Celsius per day"
    assert insufficient["dominant_period"].unit == "days"


def test_long_temporal_table_schema_and_all_families() -> None:
    dates = pd.date_range("2020-01-01", periods=8, freq="D")
    bank = pd.DataFrame({
        "date": dates,
        "metric": "anomaly.weighted_mean",
        "value": np.arange(8, dtype=float),
        "unit": "degrees_Celsius",
    })
    table = build_temporal_feature_table(
        bank,
        source_metrics=["anomaly.weighted_mean"],
        windows=[7, "all"],
        minimum_observations=5,
        instability_threshold=0.5,
        entropy_bins=5,
        spectral_minimum_observations=5,
    )
    assert list(table.columns) == TEMPORAL_COLUMNS
    assert set(table.family) == set(FEATURE_FAMILIES)
    assert set(table.source_metric) == {"anomaly.weighted_mean"}
