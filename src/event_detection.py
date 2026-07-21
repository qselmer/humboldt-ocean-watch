"""Calendar-aware univariate thermal-event identification.

This module detects events in one numerical time series. It does not link
spatial patches and does not perform spatiotemporal event tracking.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import re
from typing import Any, Literal

import numpy as np
import pandas as pd

from src.event_metrics import EventMetricBundle, calculate_event_metrics
from src.event_thresholds import ThresholdResolution
from src.metric_result import MetricResult
from src.quality_control import DuplicatePolicy

Direction = Literal["above", "below"]
Comparison = Literal["inclusive", "exclusive"]

EVENT_COLUMNS = [
    "event_id", "source_variable", "threshold_type", "threshold_description",
    "direction", "start_date", "end_date", "peak_date",
    "duration_calendar_days", "duration_observed_days", "exceedance_days",
    "interruption_days", "valid_observation_count", "mean_valid_coverage",
    "mean_intensity", "maximum_intensity", "minimum_intensity",
    "cumulative_intensity", "intensity_standard_deviation", "p90_intensity",
    "peak_value", "peak_threshold", "peak_exceedance", "onset_rate",
    "decline_rate", "time_to_peak_days", "time_from_peak_to_end_days",
    "maximum_daily_intensification", "maximum_daily_relaxation",
    "number_of_turning_points", "number_of_observed_interruptions",
    "mean_intensity_times_duration", "severity_class", "status", "reason",
    "climatology_method", "data_mode",
]

DAILY_FLAG_COLUMNS = [
    "date", "source_variable", "value", "threshold", "exceeds_threshold",
    "observed_interruption", "event_id", "event_day",
    "event_age_calendar_days", "intensity", "valid_coverage", "status", "reason",
]


@dataclass(frozen=True)
class EventDetectionConfig:
    """Configuration controlling event membership and calendar segmentation."""

    direction: Direction
    minimum_duration_days: int
    allowed_gap_days: int
    maximum_calendar_gap_days: int
    minimum_valid_coverage: float
    comparison: Comparison

    def __post_init__(self) -> None:
        if self.direction not in {"above", "below"}:
            raise ValueError("direction must be 'above' or 'below'")
        if self.comparison not in {"inclusive", "exclusive"}:
            raise ValueError("comparison must be 'inclusive' or 'exclusive'")
        if self.minimum_duration_days < 1:
            raise ValueError("minimum_duration_days must be positive")
        if self.allowed_gap_days < 0:
            raise ValueError("allowed_gap_days cannot be negative")
        if self.maximum_calendar_gap_days < 1:
            raise ValueError("maximum_calendar_gap_days must be positive")
        if not 0.0 <= self.minimum_valid_coverage <= 1.0:
            raise ValueError("minimum_valid_coverage must be between 0 and 1")


@dataclass
class EventDetectionResult:
    events: pd.DataFrame
    daily_flags: pd.DataFrame
    metric_results: dict[str, dict[str, MetricResult]] = field(default_factory=dict)
    temporal_discontinuities: list[dict[str, Any]] = field(default_factory=list)
    input_warnings: list[str] = field(default_factory=list)
    filtered_candidate_count: int = 0

    def metric_status_counts(self) -> dict[str, int]:
        counts: dict[str, int] = {}
        for metrics in self.metric_results.values():
            for result in metrics.values():
                counts[result.status] = counts.get(result.status, 0) + 1
        return counts


def prepare_event_series(
    data: pd.Series | pd.DataFrame,
    *,
    value_column: str = "value",
    date_column: str = "date",
    coverage_column: str = "valid_coverage",
    duplicate_policy: DuplicatePolicy = "error",
) -> pd.DataFrame:
    """Normalize, sort, and resolve duplicate calendar dates explicitly."""
    if isinstance(data, pd.Series):
        if not isinstance(data.index, pd.DatetimeIndex):
            raise ValueError("Input Series requires an explicit DatetimeIndex")
        frame = pd.DataFrame({"date": data.index, "value": data.to_numpy()})
        frame["valid_coverage"] = 1.0
    elif isinstance(data, pd.DataFrame):
        frame = data.copy()
        if date_column in frame.columns:
            dates = frame[date_column]
        elif isinstance(frame.index, pd.DatetimeIndex):
            dates = frame.index
        else:
            raise ValueError("Input DataFrame requires a date column or DatetimeIndex")
        if value_column not in frame.columns:
            raise ValueError(f"Input DataFrame is missing value column {value_column!r}")
        coverage = frame[coverage_column] if coverage_column in frame.columns else 1.0
        frame = pd.DataFrame({"date": dates, "value": frame[value_column], "valid_coverage": coverage})
    else:
        raise TypeError("Event input must be a pandas Series or DataFrame")
    try:
        dates = pd.DatetimeIndex(pd.to_datetime(frame["date"], errors="raise"))
    except (TypeError, ValueError) as exc:
        raise ValueError("Event input contains unreadable dates") from exc
    if dates.tz is not None:
        dates = dates.tz_convert(None)
    frame["date"] = dates.normalize()
    frame["value"] = pd.to_numeric(frame["value"], errors="coerce").astype(float)
    frame["valid_coverage"] = pd.to_numeric(frame["valid_coverage"], errors="coerce").astype(float)
    if np.isinf(frame["value"].to_numpy()).any():
        raise ValueError("Event input contains infinite values")
    coverage_values = frame["valid_coverage"].to_numpy(dtype=float)
    if np.isinf(coverage_values).any() or bool(((coverage_values < 0) | (coverage_values > 1))[np.isfinite(coverage_values)].any()):
        raise ValueError("Valid coverage must be finite or missing and between 0 and 1")
    frame = frame.sort_values("date", kind="stable").reset_index(drop=True)
    duplicate_dates = pd.DatetimeIndex(frame.date)[pd.DatetimeIndex(frame.date).duplicated(keep=False)]
    warnings: list[str] = []
    if len(duplicate_dates):
        unique_count = int(pd.Index(duplicate_dates).nunique())
        if duplicate_policy == "error":
            raise ValueError("Duplicate event dates detected; configure first, last, or mean")
        if duplicate_policy in {"first", "last"}:
            frame = frame.drop_duplicates("date", keep=duplicate_policy).reset_index(drop=True)
        elif duplicate_policy == "mean":
            frame = frame.groupby("date", as_index=False, sort=True)[["value", "valid_coverage"]].mean()
        else:
            raise ValueError(f"Unsupported duplicate policy: {duplicate_policy!r}")
        warnings.append(
            f"Resolved {unique_count} duplicate calendar date(s) using policy {duplicate_policy}"
        )
    frame.attrs["warnings"] = warnings
    return frame[["date", "value", "valid_coverage"]]


def _event_slug(source_variable: str) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", source_variable.lower()).strip("-")
    return slug or "series"


def _comparison_description(direction: Direction, comparison: Comparison) -> str:
    return {
        ("above", "inclusive"): "value >= threshold",
        ("above", "exclusive"): "value > threshold",
        ("below", "inclusive"): "value <= threshold",
        ("below", "exclusive"): "value < threshold",
    }[(direction, comparison)]


def _exceeds(
    value: np.ndarray,
    threshold: np.ndarray,
    *,
    direction: Direction,
    comparison: Comparison,
) -> np.ndarray:
    if direction == "above":
        return value >= threshold if comparison == "inclusive" else value > threshold
    return value <= threshold if comparison == "inclusive" else value < threshold


def _intensity(value: np.ndarray, threshold: np.ndarray, direction: Direction) -> np.ndarray:
    return value - threshold if direction == "above" else threshold - value


def detect_univariate_events(
    data: pd.Series | pd.DataFrame,
    threshold: ThresholdResolution,
    *,
    source_variable: str,
    config: EventDetectionConfig,
    severity_bands: dict[str, float],
    unit: str = "1",
    duplicate_policy: DuplicatePolicy = "error",
    value_column: str = "value",
    date_column: str = "date",
    coverage_column: str = "valid_coverage",
    climatology_method: str | None = None,
    data_mode: str = "unknown",
) -> EventDetectionResult:
    """Detect threshold events without bridging disallowed calendar gaps."""
    frame = prepare_event_series(
        data,
        value_column=value_column,
        date_column=date_column,
        coverage_column=coverage_column,
        duplicate_policy=duplicate_policy,
    )
    warnings = list(frame.attrs.get("warnings", []))
    dates = pd.DatetimeIndex(frame.date)
    aligned_threshold = threshold.values.reindex(dates)
    if len(aligned_threshold) != len(frame):
        raise ValueError("Threshold could not be aligned to the prepared event series")
    values = frame.value.to_numpy(dtype=float)
    threshold_values = aligned_threshold.to_numpy(dtype=float)
    coverage = frame.valid_coverage.to_numpy(dtype=float)
    value_valid = np.isfinite(values)
    threshold_valid = np.isfinite(threshold_values)
    coverage_valid = np.isfinite(coverage) & (coverage >= config.minimum_valid_coverage)
    valid = value_valid & threshold_valid & coverage_valid
    exceeds = np.zeros(len(frame), dtype=bool)
    exceeds[valid] = _exceeds(
        values[valid],
        threshold_values[valid],
        direction=config.direction,
        comparison=config.comparison,
    )
    intensity = np.full(len(frame), np.nan, dtype=float)
    intensity[valid] = _intensity(values[valid], threshold_values[valid], config.direction)

    flags = pd.DataFrame({
        "date": dates,
        "source_variable": source_variable,
        "value": values,
        "threshold": threshold_values,
        "exceeds_threshold": pd.array(exceeds, dtype="boolean"),
        "observed_interruption": False,
        "event_id": pd.Series([None] * len(frame), dtype="object"),
        "event_day": pd.array([pd.NA] * len(frame), dtype="Int64"),
        "event_age_calendar_days": pd.array([pd.NA] * len(frame), dtype="Int64"),
        "intensity": intensity,
        "valid_coverage": coverage,
        "status": "valid",
        "reason": pd.Series([None] * len(frame), dtype="object"),
    })
    flags.loc[~value_valid, ["status", "reason"]] = [
        "not_calculated", "Source value is missing or non-finite"
    ]
    flags.loc[value_valid & ~threshold_valid, ["status", "reason"]] = [
        "not_calculated", "Active threshold is missing or non-finite"
    ]
    flags.loc[value_valid & threshold_valid & ~coverage_valid, ["status", "reason"]] = [
        "not_calculated", "Valid coverage is below the configured minimum"
    ]
    flags.loc[~valid, "exceeds_threshold"] = pd.NA

    discontinuities: list[dict[str, Any]] = []
    discontinuity_before = np.zeros(len(frame), dtype=bool)
    for position in range(1, len(frame)):
        gap_days = int((dates[position] - dates[position - 1]).days)
        if gap_days > config.maximum_calendar_gap_days:
            discontinuity_before[position] = True
            discontinuity = {
                "previous_date": dates[position - 1],
                "next_date": dates[position],
                "calendar_gap_days": gap_days,
            }
            discontinuities.append(discontinuity)
            message = (
                f"Temporal discontinuity: {gap_days}-day observation gap exceeds "
                f"maximum {config.maximum_calendar_gap_days}"
            )
            if flags.at[position, "status"] == "valid":
                flags.at[position, "status"] = "warning"
                flags.at[position, "reason"] = message
            else:
                existing = flags.at[position, "reason"]
                flags.at[position, "reason"] = f"{existing}; {message}" if existing else message

    candidates: list[list[int]] = []
    active_exceedance: list[int] = []
    committed_interruptions: list[int] = []
    pending_interruptions: list[int] = []
    filtered_candidates = 0

    def finalize_candidate() -> None:
        nonlocal active_exceedance, committed_interruptions, pending_interruptions, filtered_candidates
        if active_exceedance:
            membership = sorted([*active_exceedance, *committed_interruptions])
            # Duration counts observed member days, including a permitted
            # interruption, but never an unobserved calendar day.
            if len(membership) >= config.minimum_duration_days:
                candidates.append(membership)
            else:
                filtered_candidates += 1
        active_exceedance = []
        committed_interruptions = []
        pending_interruptions = []

    for position in range(len(frame)):
        if discontinuity_before[position] or not valid[position]:
            finalize_candidate()
            if not valid[position]:
                continue
        if exceeds[position]:
            if active_exceedance:
                committed_interruptions.extend(pending_interruptions)
                pending_interruptions = []
            active_exceedance.append(position)
        elif active_exceedance:
            pending_interruptions.append(position)
            if len(pending_interruptions) > config.allowed_gap_days:
                finalize_candidate()
    finalize_candidate()

    event_rows: list[dict[str, Any]] = []
    metric_results: dict[str, dict[str, MetricResult]] = {}
    threshold_description = (
        f"{threshold.description}; {_comparison_description(config.direction, config.comparison)}"
    )
    slug = _event_slug(source_variable)
    for event_number, membership in enumerate(candidates, start=1):
        event_id = f"{slug}-E{event_number:04d}"
        member_flags = flags.iloc[membership].copy()
        member_flags["observed_interruption"] = ~member_flags["exceeds_threshold"].astype(bool)
        bundle: EventMetricBundle = calculate_event_metrics(
            member_flags,
            unit=unit,
            severity_bands=severity_bands,
        )
        metric_results[event_id] = bundle.results
        start_date = pd.Timestamp(member_flags.date.iloc[0])
        end_date = pd.Timestamp(member_flags.date.iloc[-1])
        event_values = bundle.values()
        row = {
            "event_id": event_id,
            "source_variable": source_variable,
            "threshold_type": threshold.threshold_type,
            "threshold_description": threshold_description,
            "direction": config.direction,
            "start_date": start_date,
            "end_date": end_date,
            "peak_date": bundle.peak_date,
            "duration_calendar_days": int((end_date - start_date).days + 1),
            "duration_observed_days": len(membership),
            "exceedance_days": int(member_flags.exceeds_threshold.astype(bool).sum()),
            "interruption_days": int(member_flags.observed_interruption.sum()),
            "valid_observation_count": len(membership),
            "mean_valid_coverage": float(member_flags.valid_coverage.mean()),
            **event_values,
            "severity_class": bundle.severity_class,
            "status": "valid",
            "reason": None,
            "climatology_method": climatology_method,
            "data_mode": data_mode,
        }
        event_rows.append(row)
        for event_day, position in enumerate(membership, start=1):
            flags.at[position, "event_id"] = event_id
            flags.at[position, "event_day"] = event_day
            flags.at[position, "event_age_calendar_days"] = int(
                (dates[position] - start_date).days + 1
            )
            flags.at[position, "observed_interruption"] = bool(not exceeds[position])

    events = pd.DataFrame(event_rows, columns=EVENT_COLUMNS)
    daily_flags = flags[DAILY_FLAG_COLUMNS].copy()
    return EventDetectionResult(
        events=events,
        daily_flags=daily_flags,
        metric_results=metric_results,
        temporal_discontinuities=discontinuities,
        input_warnings=warnings,
        filtered_candidate_count=filtered_candidates,
    )
