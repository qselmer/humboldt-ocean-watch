"""Build cached univariate thermal events, daily flags, and JSON summary."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import COPERNICUS_CACHED_MODE
from src.event_detection import (
    EventDetectionConfig,
    detect_univariate_events,
    prepare_event_series,
)
from src.event_metrics import build_event_summary
from src.event_thresholds import ThresholdResolution, resolve_threshold
from src.export_utils import dumps_json_safe
from src.utils import configure_logging, load_config, resolve_project_path

LOGGER = logging.getLogger(__name__)

SOURCE_ALIASES: dict[str, tuple[str, str, float, str]] = {
    "regional_mean_sst": ("series_bank", "sst.weighted_mean", 1.0, "degrees_Celsius"),
    "regional_mean_sst_anomaly": ("series_bank", "anomaly.weighted_mean", 1.0, "degrees_Celsius"),
    "standardized_anomaly": ("series_bank", "zscore.weighted_mean", 1.0, "1"),
    "warm_area_fraction": ("daily_metrics", "area_anomaly_ge_2c_percent", 0.01, "1"),
}


@dataclass(frozen=True)
class SourceSeries:
    frame: pd.DataFrame
    source_variable: str
    source_metric: str
    unit: str
    climatology_method: str | None
    data_mode: str


@dataclass(frozen=True)
class BuildArtifacts:
    events_path: Path
    daily_flags_path: Path
    summary_path: Path
    events: pd.DataFrame
    daily_flags: pd.DataFrame
    summary: dict[str, Any]


def _metadata_value(frame: pd.DataFrame, column: str, default: Any = None) -> Any:
    if column not in frame:
        return default
    values = frame[column].dropna().unique()
    if len(values) > 1:
        raise ValueError(f"Cached source contains inconsistent {column} values")
    return values[0] if len(values) else default


def load_source_series(
    *,
    series_bank_path: Path,
    daily_metrics_path: Path,
    source_variable: str,
    source_metric: str | None,
) -> SourceSeries:
    """Load one explicit cached source series without network access."""
    if not series_bank_path.exists():
        raise FileNotFoundError(f"Cached series bank not found: {series_bank_path}")
    bank = pd.read_parquet(series_bank_path)
    required_bank = {"date", "metric", "value", "valid_coverage", "status"}
    missing = required_bank - set(bank.columns)
    if missing:
        raise ValueError(f"Series bank is missing columns: {sorted(missing)}")
    alias = SOURCE_ALIASES.get(source_variable)
    effective_metric = source_metric or (alias[1] if alias else None)
    if not effective_metric:
        raise ValueError("An explicit source metric is required for an unrecognized source variable")

    bank_selection = bank[bank.metric == effective_metric].copy()
    if not bank_selection.empty:
        bank_selection["date"] = pd.to_datetime(bank_selection.date, errors="raise")
        bank_selection["value"] = pd.to_numeric(bank_selection.value, errors="coerce")
        unavailable = ~bank_selection.status.isin(["valid", "warning"])
        bank_selection.loc[unavailable, "value"] = np.nan
        unit = str(_metadata_value(bank_selection, "unit", "1"))
        frame = bank_selection[["date", "value", "valid_coverage"]].copy()
        return SourceSeries(
            frame=frame,
            source_variable=source_variable,
            source_metric=effective_metric,
            unit=unit,
            climatology_method=_metadata_value(bank_selection, "climatology_method"),
            data_mode=str(_metadata_value(bank_selection, "data_mode", "unknown")),
        )

    if alias is None or alias[0] != "daily_metrics" or effective_metric != alias[1]:
        raise ValueError(f"Source metric {effective_metric!r} is not present in the cached series bank")
    if not daily_metrics_path.exists():
        raise FileNotFoundError(f"Cached daily metrics not found: {daily_metrics_path}")
    daily = pd.read_parquet(daily_metrics_path)
    if effective_metric not in daily or "date" not in daily:
        raise ValueError(f"Daily metrics do not contain source column {effective_metric!r}")
    coverage = (
        pd.to_numeric(daily["valid_data_coverage_percent"], errors="coerce") / 100.0
        if "valid_data_coverage_percent" in daily
        else pd.Series(1.0, index=daily.index)
    )
    scale = alias[2]
    frame = pd.DataFrame({
        "date": pd.to_datetime(daily.date, errors="raise"),
        "value": pd.to_numeric(daily[effective_metric], errors="coerce") * scale,
        "valid_coverage": coverage,
    })
    return SourceSeries(
        frame=frame,
        source_variable=source_variable,
        source_metric=effective_metric,
        unit=alias[3],
        climatology_method=_metadata_value(daily, "climatology_method"),
        data_mode=str(_metadata_value(bank, "data_mode", "unknown")),
    )


def _temporary_path(destination: Path, suffix: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.stem}-", suffix=suffix, dir=destination.parent,
    )
    os.close(descriptor)
    return Path(name)


def write_outputs_atomically(
    events: pd.DataFrame,
    daily_flags: pd.DataFrame,
    summary: dict[str, Any],
    *,
    events_path: Path,
    daily_flags_path: Path,
    summary_path: Path,
) -> None:
    """Write and validate every temporary product before replacing outputs."""
    temporary_events = _temporary_path(events_path, ".parquet")
    temporary_flags = _temporary_path(daily_flags_path, ".parquet")
    temporary_summary = _temporary_path(summary_path, ".json")
    temporary = [temporary_events, temporary_flags, temporary_summary]
    try:
        events.to_parquet(temporary_events, index=False)
        daily_flags.to_parquet(temporary_flags, index=False)
        temporary_summary.write_text(dumps_json_safe(summary) + "\n", encoding="utf-8")
        if pd.read_parquet(temporary_events).columns.tolist() != events.columns.tolist():
            raise RuntimeError("Temporary event table failed schema validation")
        if pd.read_parquet(temporary_flags).columns.tolist() != daily_flags.columns.tolist():
            raise RuntimeError("Temporary daily-flag table failed schema validation")
        json.loads(temporary_summary.read_text(encoding="utf-8"))
        temporary_events.replace(events_path)
        temporary_flags.replace(daily_flags_path)
        temporary_summary.replace(summary_path)
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


def _threshold_resolution(
    source: SourceSeries,
    *,
    threshold_type: str,
    fixed_threshold: float | None,
    percentile: float,
    climatology_path: Path,
    climatological_variable: str,
    minimum_coverage: float,
) -> tuple[ThresholdResolution, xr.Dataset | None]:
    values = pd.Series(
        source.frame.value.to_numpy(dtype=float),
        index=pd.DatetimeIndex(source.frame.date),
    )
    climatology: xr.Dataset | None = None
    if threshold_type == "daily_climatological":
        if source.source_metric != "sst.weighted_mean":
            raise ValueError(
                "Daily climatological P10/P90 thresholds are compatible only with regional mean SST"
            )
        if not climatology_path.exists():
            raise FileNotFoundError(
                f"Smoothed daily climatology is required and was not found: {climatology_path}"
            )
        climatology = xr.open_dataset(climatology_path)
        if (
            source.data_mode == COPERNICUS_CACHED_MODE
            and str(climatology.attrs.get("climatology_method", "daily_smoothed")) != "daily_smoothed"
        ):
            climatology.close()
            raise ValueError("Real cached SST cannot use a synthetic or non-daily climatology")
    eligible = (
        np.isfinite(source.frame.value.to_numpy(dtype=float))
        & np.isfinite(source.frame.valid_coverage.to_numpy(dtype=float))
        & (source.frame.valid_coverage.to_numpy(dtype=float) >= minimum_coverage)
    )
    try:
        resolution = resolve_threshold(
            values,
            threshold_type,  # type: ignore[arg-type]
            fixed_value=fixed_threshold,
            percentile=percentile,
            climatology=climatology,
            climatological_variable=climatological_variable,
            valid_mask=eligible,
        )
    except Exception:
        if climatology is not None:
            climatology.close()
        raise
    return resolution, climatology


def build(
    *,
    config_path: Path,
    series_bank_path: Path | None,
    daily_metrics_path: Path | None,
    climatology_input: Path | None,
    source_variable: str | None,
    source_metric: str | None,
    threshold_type: str | None,
    fixed_threshold: float | None,
    direction: str | None,
    minimum_duration: int | None,
    allowed_gap: int | None,
    events_output: Path | None,
    daily_flags_output: Path | None,
    summary_output: Path | None,
    overwrite: bool,
) -> BuildArtifacts:
    config = load_config(resolve_project_path(config_path))
    configure_logging(config["logging"]["level"])
    event_config = config["event_detection"]
    configured_source = str(event_config["source_variable"])
    effective_source = source_variable or configured_source
    if source_metric is not None:
        effective_metric = source_metric
    elif source_variable is not None and source_variable != configured_source:
        effective_metric = SOURCE_ALIASES.get(effective_source, (None, None, None, None))[1]
    else:
        effective_metric = str(event_config["source_metric"])
    bank_path = resolve_project_path(series_bank_path or config["series_bank"]["output"])
    metrics_path = resolve_project_path(daily_metrics_path or config["data"]["metrics_path"])
    source = load_source_series(
        series_bank_path=bank_path,
        daily_metrics_path=metrics_path,
        source_variable=effective_source,
        source_metric=effective_metric,
    )
    prepared_source = prepare_event_series(
        source.frame,
        duplicate_policy=config["quality_control"]["duplicate_policy"],
    )
    preparation_warnings = list(prepared_source.attrs.get("warnings", []))
    source = SourceSeries(
        frame=prepared_source,
        source_variable=source.source_variable,
        source_metric=source.source_metric,
        unit=source.unit,
        climatology_method=source.climatology_method,
        data_mode=source.data_mode,
    )
    effective_threshold_type = threshold_type or str(event_config["threshold_type"])
    effective_direction = direction or str(event_config["direction"])
    effective_fixed = (
        fixed_threshold
        if fixed_threshold is not None
        else float(event_config["fixed_anomaly_threshold_c"])
    )
    detection_config = EventDetectionConfig(
        direction=effective_direction,  # type: ignore[arg-type]
        minimum_duration_days=(
            minimum_duration if minimum_duration is not None else int(event_config["minimum_duration_days"])
        ),
        allowed_gap_days=allowed_gap if allowed_gap is not None else int(event_config["allowed_gap_days"]),
        maximum_calendar_gap_days=int(event_config["maximum_calendar_gap_days"]),
        minimum_valid_coverage=float(event_config["minimum_valid_coverage"]),
        comparison=str(event_config["comparison"]),  # type: ignore[arg-type]
    )
    climatology_path = resolve_project_path(
        climatology_input or config["climatology"]["daily_path"]
    )
    LOGGER.info(
        "Resolving %s threshold for %s (%s)",
        effective_threshold_type,
        effective_source,
        source.source_metric,
    )
    resolution, climatology = _threshold_resolution(
        source,
        threshold_type=effective_threshold_type,
        fixed_threshold=effective_fixed,
        percentile=float(event_config["percentile"]),
        climatology_path=climatology_path,
        climatological_variable=str(event_config["climatological_threshold_variable"]),
        minimum_coverage=detection_config.minimum_valid_coverage,
    )
    try:
        result = detect_univariate_events(
            source.frame,
            resolution,
            source_variable=source.source_variable,
            config=detection_config,
            severity_bands=dict(event_config["severity_bands"]),
            unit=source.unit,
            duplicate_policy=config["quality_control"]["duplicate_policy"],
            climatology_method=source.climatology_method,
            data_mode=source.data_mode,
        )
    finally:
        if climatology is not None:
            climatology.close()
    comparison_text = {
        ("above", "inclusive"): "value >= threshold",
        ("above", "exclusive"): "value > threshold",
        ("below", "inclusive"): "value <= threshold",
        ("below", "exclusive"): "value < threshold",
    }[(detection_config.direction, detection_config.comparison)]
    description = f"{resolution.description}; {comparison_text}"
    summary = build_event_summary(
        result.events,
        result.daily_flags,
        result.metric_results,
        source_variable=source.source_variable,
        source_metric=source.source_metric,
        threshold_type=resolution.threshold_type,
        threshold_description=description,
        direction=detection_config.direction,
        climatology_method=source.climatology_method,
        data_mode=source.data_mode,
        temporal_discontinuities=result.temporal_discontinuities,
        input_warnings=[*preparation_warnings, *result.input_warnings],
        filtered_candidate_count=result.filtered_candidate_count,
        configuration={
            **event_config,
            "direction": detection_config.direction,
            "threshold_type": resolution.threshold_type,
            "minimum_duration_days": detection_config.minimum_duration_days,
            "allowed_gap_days": detection_config.allowed_gap_days,
        },
    )
    events_path = resolve_project_path(events_output or event_config["events_output"])
    flags_path = resolve_project_path(daily_flags_output or event_config["daily_flags_output"])
    json_path = resolve_project_path(summary_output or event_config["summary_output"])
    existing = [path for path in (events_path, flags_path, json_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Event output exists; use --overwrite: " + ", ".join(map(str, existing))
        )
    LOGGER.info("Writing %d detected event(s) and %d daily flag rows", len(result.events), len(result.daily_flags))
    write_outputs_atomically(
        result.events,
        result.daily_flags,
        summary,
        events_path=events_path,
        daily_flags_path=flags_path,
        summary_path=json_path,
    )
    return BuildArtifacts(
        events_path=events_path,
        daily_flags_path=flags_path,
        summary_path=json_path,
        events=result.events,
        daily_flags=result.daily_flags,
        summary=summary,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-bank", type=Path)
    parser.add_argument("--daily-metrics", type=Path)
    parser.add_argument("--climatology-input", type=Path)
    parser.add_argument("--source-variable")
    parser.add_argument("--source-metric")
    parser.add_argument(
        "--threshold-type",
        choices=["fixed", "global_percentile", "daily_climatological", "custom_time_varying"],
    )
    parser.add_argument("--fixed-threshold", type=float)
    parser.add_argument("--direction", choices=["above", "below"])
    parser.add_argument("--minimum-duration", type=int)
    parser.add_argument("--allowed-gap", type=int)
    parser.add_argument("--events-output", type=Path)
    parser.add_argument("--daily-flags-output", type=Path)
    parser.add_argument("--summary-output", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = build(
        config_path=args.config,
        series_bank_path=args.series_bank,
        daily_metrics_path=args.daily_metrics,
        climatology_input=args.climatology_input,
        source_variable=args.source_variable,
        source_metric=args.source_metric,
        threshold_type=args.threshold_type,
        fixed_threshold=args.fixed_threshold,
        direction=args.direction,
        minimum_duration=args.minimum_duration,
        allowed_gap=args.allowed_gap,
        events_output=args.events_output,
        daily_flags_output=args.daily_flags_output,
        summary_output=args.summary_output,
        overwrite=args.overwrite,
    )
    print(f"Events: {artifacts.events_path} {artifacts.events.shape}")
    print(f"Daily flags: {artifacts.daily_flags_path} {artifacts.daily_flags.shape}")
    print(f"Summary: {artifacts.summary_path}")
    print(f"Source series: {artifacts.summary['source_variable']} ({artifacts.summary['source_metric']})")
    print(f"Threshold: {artifacts.summary['threshold_description']}")
    print(f"Events detected: {artifacts.summary['number_of_events']}")
    print(f"Event date ranges: {artifacts.summary['event_date_ranges']}")
    print(f"Metric result statuses: {artifacts.summary['metric_result_status_counts']}")


if __name__ == "__main__":
    main()
