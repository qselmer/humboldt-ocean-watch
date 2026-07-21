"""Calendar-aware tests for univariate thermal-event identification."""

import numpy as np
import pandas as pd
import pytest

from src.event_detection import (
    DAILY_FLAG_COLUMNS,
    EVENT_COLUMNS,
    EventDetectionConfig,
    detect_univariate_events,
    prepare_event_series,
)
from src.event_thresholds import fixed_threshold


def _config(
    *,
    direction: str = "above",
    minimum_duration: int = 2,
    allowed_gap: int = 0,
    maximum_calendar_gap: int = 1,
    minimum_coverage: float = 0.8,
    comparison: str = "inclusive",
) -> EventDetectionConfig:
    return EventDetectionConfig(
        direction=direction,  # type: ignore[arg-type]
        minimum_duration_days=minimum_duration,
        allowed_gap_days=allowed_gap,
        maximum_calendar_gap_days=maximum_calendar_gap,
        minimum_valid_coverage=minimum_coverage,
        comparison=comparison,  # type: ignore[arg-type]
    )


def _detect(
    values: list[float],
    *,
    dates: pd.DatetimeIndex | None = None,
    threshold: float = 2.0,
    coverage: list[float] | None = None,
    **config_kwargs,
):
    dates = dates if dates is not None else pd.date_range("2024-01-01", periods=len(values))
    frame = pd.DataFrame(
        {
            "date": dates,
            "value": values,
            "valid_coverage": coverage if coverage is not None else 1.0,
        }
    )
    return detect_univariate_events(
        frame,
        fixed_threshold(dates.sort_values(), threshold),
        source_variable="test_anomaly",
        config=_config(**config_kwargs),
        severity_bands={"strong": 1.0, "extreme": 2.0},
        unit="degrees_Celsius",
        climatology_method="daily_smoothed",
        data_mode="Copernicus cached data",
    )


def test_unsorted_dates_are_sorted_and_duplicate_policies_are_explicit() -> None:
    frame = pd.DataFrame(
        {
            "date": ["2024-01-02", "2024-01-01", "2024-01-01"],
            "value": [3.0, 1.0, 5.0],
        }
    )
    with pytest.raises(ValueError, match="Duplicate"):
        prepare_event_series(frame)
    first = prepare_event_series(frame, duplicate_policy="first")
    last = prepare_event_series(frame, duplicate_policy="last")
    mean = prepare_event_series(frame, duplicate_policy="mean")
    assert first.date.is_monotonic_increasing
    assert first.iloc[0].value == 1.0
    assert last.iloc[0].value == 5.0
    assert mean.iloc[0].value == 3.0


def test_simple_above_threshold_event_and_output_schemas() -> None:
    result = _detect([0.0, 2.1, 3.0, 2.2, 0.0])
    assert result.events.columns.tolist() == EVENT_COLUMNS
    assert result.daily_flags.columns.tolist() == DAILY_FLAG_COLUMNS
    assert len(result.events) == 1
    event = result.events.iloc[0]
    assert event.start_date == pd.Timestamp("2024-01-02")
    assert event.end_date == pd.Timestamp("2024-01-04")
    assert event.peak_date == pd.Timestamp("2024-01-03")
    assert event.event_id == "test-anomaly-E0001"
    assert event.cumulative_intensity == pytest.approx(1.3)


def test_below_threshold_event_has_positive_relative_intensity() -> None:
    result = _detect(
        [3.0, 1.5, 0.0, 1.0, 3.0], direction="below", threshold=2.0
    )
    assert len(result.events) == 1
    assert result.events.iloc[0].maximum_intensity == 2.0
    assert result.events.iloc[0].peak_date == pd.Timestamp("2024-01-03")


def test_exact_threshold_inclusion_is_configurable() -> None:
    inclusive = _detect([2.0, 2.0], minimum_duration=2, comparison="inclusive")
    exclusive = _detect([2.0, 2.0], minimum_duration=2, comparison="exclusive")
    assert len(inclusive.events) == 1
    assert exclusive.events.empty


def test_minimum_duration_filters_short_candidate() -> None:
    result = _detect([3.0, 3.0, 0.0], minimum_duration=3)
    assert result.events.empty
    assert result.filtered_candidate_count == 1
    assert result.daily_flags.event_id.isna().all()


def test_allowed_observed_interruption_is_included_in_event() -> None:
    result = _detect([3.0, 3.0, 1.0, 3.0, 3.0], allowed_gap=1)
    assert len(result.events) == 1
    event = result.events.iloc[0]
    assert event.interruption_days == 1
    assert event.duration_observed_days == 5
    interruption = result.daily_flags.iloc[2]
    assert bool(interruption.observed_interruption)
    assert interruption.event_id == event.event_id


def test_consecutive_interruption_days_form_one_observed_interruption() -> None:
    result = _detect([3.0, 3.0, 1.0, 1.0, 3.0, 3.0], allowed_gap=2)
    event_id = result.events.iloc[0].event_id
    metric = result.metric_results[event_id]["number_of_observed_interruptions"]
    assert metric.value == 1


def test_interruption_exceeding_allowed_gap_separates_events() -> None:
    result = _detect([3.0, 3.0, 1.0, 1.0, 3.0, 3.0], allowed_gap=1)
    assert len(result.events) == 2
    assert result.events.event_id.tolist() == ["test-anomaly-E0001", "test-anomaly-E0002"]
    assert result.daily_flags.iloc[2:4].event_id.isna().all()


def test_missing_calendar_day_terminates_event_and_records_discontinuity() -> None:
    dates = pd.DatetimeIndex(["2024-01-01", "2024-01-02", "2024-01-04", "2024-01-05"])
    result = _detect([3.0] * 4, dates=dates)
    assert len(result.events) == 2
    assert result.events.duration_calendar_days.tolist() == [2, 2]
    assert result.temporal_discontinuities[0]["calendar_gap_days"] == 2
    assert result.daily_flags.iloc[2].status == "warning"


def test_irregular_but_permitted_intervals_use_calendar_duration() -> None:
    dates = pd.DatetimeIndex(["2024-01-01", "2024-01-03", "2024-01-05"])
    result = _detect(
        [3.0, 4.0, 3.0], dates=dates, maximum_calendar_gap=2, minimum_duration=3
    )
    assert len(result.events) == 1
    assert result.events.iloc[0].duration_calendar_days == 5
    assert result.events.iloc[0].duration_observed_days == 3


def test_event_can_begin_first_date_and_end_final_date() -> None:
    result = _detect([3.0, 4.0, 3.0])
    event = result.events.iloc[0]
    assert event.start_date == pd.Timestamp("2024-01-01")
    assert event.end_date == pd.Timestamp("2024-01-03")


def test_multiple_events_and_deterministic_ids() -> None:
    first = _detect([3.0, 3.0, 0.0, 3.0, 3.0])
    second = _detect([3.0, 3.0, 0.0, 3.0, 3.0])
    expected = ["test-anomaly-E0001", "test-anomaly-E0002"]
    assert first.events.event_id.tolist() == expected
    assert second.events.event_id.tolist() == expected


def test_no_event_and_constant_series_are_supported() -> None:
    no_event = _detect([1.0, 1.0, 1.0])
    constant_event = _detect([3.0, 3.0, 3.0])
    assert no_event.events.empty
    assert len(constant_event.events) == 1


def test_missing_value_and_insufficient_coverage_break_events() -> None:
    missing = _detect([3.0, 3.0, np.nan, 3.0, 3.0])
    low_coverage = _detect(
        [3.0] * 5, coverage=[1.0, 1.0, 0.5, 1.0, 1.0]
    )
    assert len(missing.events) == 2
    assert missing.daily_flags.iloc[2].status == "not_calculated"
    assert len(low_coverage.events) == 2
    assert "coverage" in str(low_coverage.daily_flags.iloc[2].reason).lower()


def test_membership_event_day_and_calendar_age_are_distinct() -> None:
    dates = pd.DatetimeIndex(["2024-01-01", "2024-01-03", "2024-01-05"])
    result = _detect(
        [3.0, 3.0, 3.0], dates=dates, maximum_calendar_gap=2, minimum_duration=3
    )
    assert result.daily_flags.event_day.tolist() == [1, 2, 3]
    assert result.daily_flags.event_age_calendar_days.tolist() == [1, 3, 5]
