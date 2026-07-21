"""Intensity, development, and neutral severity metrics for thermal events."""

from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from typing import Any, Mapping

import numpy as np
import pandas as pd

from src.metric_result import MetricResult, not_calculated


@dataclass(frozen=True)
class EventMetricBundle:
    """Calculated event metrics plus non-numeric event descriptors."""

    results: dict[str, MetricResult]
    peak_date: pd.Timestamp
    severity_class: str

    def values(self) -> dict[str, float]:
        return {name: float(result.value) for name, result in self.results.items()}

    def status_counts(self) -> dict[str, int]:
        statuses = pd.Series([result.status for result in self.results.values()])
        return {str(name): int(count) for name, count in statuses.value_counts().items()}


def severity_category(maximum_intensity: float, bands: Mapping[str, Any]) -> str:
    """Classify maximum relative intensity without official ENSO terminology."""
    if "strong" not in bands or "extreme" not in bands:
        raise ValueError("Severity bands require 'strong' and 'extreme' thresholds")
    strong = float(bands["strong"])
    extreme = float(bands["extreme"])
    if not np.isfinite(strong) or not np.isfinite(extreme) or strong < 0 or extreme <= strong:
        raise ValueError("Severity thresholds must be finite with 0 <= strong < extreme")
    if maximum_intensity >= extreme:
        return "extreme thermal exceedance"
    if maximum_intensity >= strong:
        return "strong thermal exceedance"
    return "moderate thermal exceedance"


def _valid_result(
    metric: str,
    value: float,
    *,
    unit: str,
    family: str,
    n: int,
    coverage: float,
) -> MetricResult:
    return MetricResult(
        metric=metric,
        value=float(value),
        status="valid",
        n_observations=n,
        valid_coverage=coverage,
        unit=unit,
        family=family,
    )


def _rate_result(
    name: str,
    numerator: float,
    elapsed_days: int,
    *,
    unit: str,
    n: int,
    coverage: float,
    reason: str,
) -> MetricResult:
    if elapsed_days <= 0:
        return not_calculated(
            name,
            reason,
            unit=f"{unit} per day",
            family="event_development",
            n_observations=n,
            valid_coverage=coverage,
        )
    return _valid_result(
        name,
        numerator / elapsed_days,
        unit=f"{unit} per day",
        family="event_development",
        n=n,
        coverage=coverage,
    )


def calculate_event_metrics(
    event: pd.DataFrame,
    *,
    unit: str,
    severity_bands: Mapping[str, Any],
) -> EventMetricBundle:
    """Calculate event metrics from observed membership rows.

    Intensity is signed relative to the active threshold, so exceedance days
    are non-negative and allowed observed interruptions may be negative.
    Onset and decline rates use the first/last observed membership values and
    actual calendar days to and from the maximum-intensity date.
    """
    required = {
        "date", "value", "threshold", "intensity", "exceeds_threshold",
        "observed_interruption", "valid_coverage",
    }
    missing = required - set(event.columns)
    if missing:
        raise ValueError(f"Event metric input is missing columns: {sorted(missing)}")
    frame = event.copy()
    frame["date"] = pd.to_datetime(frame["date"], errors="raise")
    frame = frame.sort_values("date", kind="stable")
    intensity = pd.to_numeric(frame["intensity"], errors="coerce").to_numpy(dtype=float)
    values = pd.to_numeric(frame["value"], errors="coerce").to_numpy(dtype=float)
    thresholds = pd.to_numeric(frame["threshold"], errors="coerce").to_numpy(dtype=float)
    coverage_values = pd.to_numeric(frame["valid_coverage"], errors="coerce").to_numpy(dtype=float)
    finite = np.isfinite(intensity) & np.isfinite(values) & np.isfinite(thresholds)
    if not finite.all() or frame.empty:
        raise ValueError("Event membership rows must contain finite value, threshold, and intensity")
    n = len(frame)
    coverage = float(np.nanmean(coverage_values)) if np.isfinite(coverage_values).any() else np.nan
    mean_intensity = float(np.mean(intensity))
    maximum_intensity = float(np.max(intensity))
    minimum_intensity = float(np.min(intensity))
    peak_position = int(np.argmax(intensity))
    peak_date = pd.Timestamp(frame.iloc[peak_position]["date"])
    start_date = pd.Timestamp(frame.iloc[0]["date"])
    end_date = pd.Timestamp(frame.iloc[-1]["date"])
    duration_calendar_days = int((end_date - start_date).days + 1)
    interruption_mask = frame["observed_interruption"].astype(bool).to_numpy()
    interruption_starts = interruption_mask & np.r_[True, ~interruption_mask[:-1]]
    interruption_count = int(interruption_starts.sum())
    results: dict[str, MetricResult] = {
        "mean_intensity": _valid_result("mean_intensity", mean_intensity, unit=unit, family="event_intensity", n=n, coverage=coverage),
        "maximum_intensity": _valid_result("maximum_intensity", maximum_intensity, unit=unit, family="event_intensity", n=n, coverage=coverage),
        "minimum_intensity": _valid_result("minimum_intensity", minimum_intensity, unit=unit, family="event_intensity", n=n, coverage=coverage),
        "cumulative_intensity": _valid_result("cumulative_intensity", float(np.sum(intensity)), unit=f"{unit} observed-day sum", family="event_intensity", n=n, coverage=coverage),
        "intensity_standard_deviation": _valid_result("intensity_standard_deviation", float(np.std(intensity)), unit=unit, family="event_intensity", n=n, coverage=coverage),
        "p90_intensity": _valid_result("p90_intensity", float(np.quantile(intensity, 0.90)), unit=unit, family="event_intensity", n=n, coverage=coverage),
        "peak_value": _valid_result("peak_value", values[peak_position], unit=unit, family="event_intensity", n=n, coverage=coverage),
        "peak_threshold": _valid_result("peak_threshold", thresholds[peak_position], unit=unit, family="event_intensity", n=n, coverage=coverage),
        "peak_exceedance": _valid_result("peak_exceedance", intensity[peak_position], unit=unit, family="event_intensity", n=n, coverage=coverage),
        "time_to_peak_days": _valid_result("time_to_peak_days", float((peak_date - start_date).days), unit="days", family="event_development", n=n, coverage=coverage),
        "time_from_peak_to_end_days": _valid_result("time_from_peak_to_end_days", float((end_date - peak_date).days), unit="days", family="event_development", n=n, coverage=coverage),
        "number_of_observed_interruptions": _valid_result("number_of_observed_interruptions", float(interruption_count), unit="count", family="event_development", n=n, coverage=coverage),
        "mean_intensity_times_duration": _valid_result("mean_intensity_times_duration", mean_intensity * duration_calendar_days, unit=f"{unit} calendar-days", family="event_severity", n=n, coverage=coverage),
    }
    results["onset_rate"] = _rate_result(
        "onset_rate",
        maximum_intensity - intensity[0],
        int((peak_date - start_date).days),
        unit=unit,
        n=n,
        coverage=coverage,
        reason="Onset rate requires the peak to occur after the event start",
    )
    results["decline_rate"] = _rate_result(
        "decline_rate",
        maximum_intensity - intensity[-1],
        int((end_date - peak_date).days),
        unit=unit,
        n=n,
        coverage=coverage,
        reason="Decline rate requires the event to continue after the peak",
    )

    if n >= 2:
        elapsed = np.diff(frame["date"].to_numpy(dtype="datetime64[ns]")).astype("timedelta64[D]").astype(int)
        if bool((elapsed <= 0).any()):
            raise ValueError("Event dates must be unique and increasing")
        rates = np.diff(intensity) / elapsed
        results["maximum_daily_intensification"] = _valid_result(
            "maximum_daily_intensification", float(max(np.max(rates), 0.0)),
            unit=f"{unit} per day", family="event_development", n=n, coverage=coverage,
        )
        results["maximum_daily_relaxation"] = _valid_result(
            "maximum_daily_relaxation", float(max(np.max(-rates), 0.0)),
            unit=f"{unit} per day", family="event_development", n=n, coverage=coverage,
        )
        nonzero_signs = np.sign(rates)
        nonzero_signs = nonzero_signs[nonzero_signs != 0]
        turning_points = int(np.sum(nonzero_signs[1:] != nonzero_signs[:-1])) if nonzero_signs.size > 1 else 0
        results["number_of_turning_points"] = _valid_result(
            "number_of_turning_points", float(turning_points), unit="count",
            family="event_development", n=n, coverage=coverage,
        )
    else:
        for name, result_unit in (
            ("maximum_daily_intensification", f"{unit} per day"),
            ("maximum_daily_relaxation", f"{unit} per day"),
            ("number_of_turning_points", "count"),
        ):
            results[name] = not_calculated(
                name,
                "At least two observed event days are required",
                unit=result_unit,
                family="event_development",
                n_observations=n,
                valid_coverage=coverage,
            )
    severity = severity_category(maximum_intensity, severity_bands)
    return EventMetricBundle(results=results, peak_date=peak_date, severity_class=severity)


def build_event_summary(
    events: pd.DataFrame,
    daily_flags: pd.DataFrame,
    metric_results: Mapping[str, Mapping[str, MetricResult]],
    *,
    source_variable: str,
    source_metric: str,
    threshold_type: str,
    threshold_description: str,
    direction: str,
    climatology_method: str | None,
    data_mode: str,
    temporal_discontinuities: list[dict[str, Any]],
    input_warnings: list[str],
    filtered_candidate_count: int,
    configuration: Mapping[str, Any],
) -> dict[str, Any]:
    """Create a strict-JSON-compatible operational event summary payload."""
    metric_status_counts: dict[str, int] = {}
    for event_metrics in metric_results.values():
        for result in event_metrics.values():
            metric_status_counts[result.status] = metric_status_counts.get(result.status, 0) + 1
    event_status_counts = (
        {str(name): int(count) for name, count in events.status.value_counts(dropna=False).items()}
        if "status" in events else {}
    )
    daily_status_counts = (
        {str(name): int(count) for name, count in daily_flags.status.value_counts(dropna=False).items()}
        if "status" in daily_flags else {}
    )
    date_ranges = [
        {
            "event_id": row.event_id,
            "start_date": row.start_date,
            "end_date": row.end_date,
            "peak_date": row.peak_date,
        }
        for row in events.itertuples(index=False)
    ]
    return {
        "title": "Humboldt Ocean Watch univariate thermal-event summary",
        "experimental_product": True,
        "official_enso_classification": "not provided",
        "event_tracking_type": "univariate temporal detection only",
        "source_variable": source_variable,
        "source_metric": source_metric,
        "threshold_type": threshold_type,
        "threshold_description": threshold_description,
        "direction": direction,
        "climatology_method": climatology_method,
        "data_mode": data_mode,
        "number_of_events": len(events),
        "event_date_ranges": date_ranges,
        "event_status_counts": event_status_counts,
        "daily_flag_status_counts": daily_status_counts,
        "metric_result_status_counts": metric_status_counts,
        "dates_not_calculated": int((daily_flags.status == "not_calculated").sum()),
        "temporal_discontinuities": temporal_discontinuities,
        "input_warnings": input_warnings,
        "filtered_candidate_count": filtered_candidate_count,
        "configuration": dict(configuration),
        "created_at": datetime.now(UTC),
    }
