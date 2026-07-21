"""Cached integration tests for the Increment 4A event builder."""

from copy import deepcopy
import json
from pathlib import Path

import pandas as pd
import pytest
import yaml

from scripts.build_univariate_events import build
from src.data_loader import COPERNICUS_CACHED_MODE
from src.event_detection import DAILY_FLAG_COLUMNS, EVENT_COLUMNS
from src.utils import load_config


def _write_bank(path: Path, metric: str = "anomaly.weighted_mean") -> None:
    dates = pd.date_range("2024-01-01", periods=8)
    values = [0.0, 2.5, 3.0, 2.8, 0.0, 2.2, 2.3, 0.0]
    pd.DataFrame(
        {
            "date": dates,
            "family": "state",
            "metric": metric,
            "value": values,
            "unit": "degrees_Celsius",
            "status": "valid",
            "reason": None,
            "n_observations": 4,
            "valid_coverage": 1.0,
            "window_days": None,
            "climatology_method": "daily_smoothed",
            "data_mode": COPERNICUS_CACHED_MODE,
        }
    ).to_parquet(path, index=False)


def _config(tmp_path: Path) -> Path:
    config = deepcopy(load_config())
    config["event_detection"].update(
        {
            "source_variable": "regional_mean_sst_anomaly",
            "source_metric": "anomaly.weighted_mean",
            "threshold_type": "fixed",
            "fixed_anomaly_threshold_c": 2.0,
            "minimum_duration_days": 2,
            "allowed_gap_days": 0,
            "events_output": str(tmp_path / "events.parquet"),
            "daily_flags_output": str(tmp_path / "flags.parquet"),
            "summary_output": str(tmp_path / "summary.json"),
        }
    )
    config["series_bank"]["output"] = str(tmp_path / "bank.parquet")
    config["data"]["metrics_path"] = str(tmp_path / "unused_metrics.parquet")
    path = tmp_path / "config.yaml"
    path.write_text(yaml.safe_dump(config, sort_keys=False), encoding="utf-8")
    return path


def _run(tmp_path: Path, **overrides):
    arguments = {
        "config_path": _config(tmp_path),
        "series_bank_path": None,
        "daily_metrics_path": None,
        "climatology_input": None,
        "source_variable": None,
        "source_metric": None,
        "threshold_type": None,
        "fixed_threshold": None,
        "direction": None,
        "minimum_duration": None,
        "allowed_gap": None,
        "events_output": None,
        "daily_flags_output": None,
        "summary_output": None,
        "overwrite": False,
    }
    arguments.update(overrides)
    return build(**arguments)


def test_cached_builder_writes_all_output_schemas_and_json(tmp_path: Path) -> None:
    _write_bank(tmp_path / "bank.parquet")
    artifacts = _run(tmp_path)
    assert artifacts.events_path.exists()
    assert artifacts.daily_flags_path.exists()
    assert artifacts.summary_path.exists()
    assert artifacts.events.columns.tolist() == EVENT_COLUMNS
    assert artifacts.daily_flags.columns.tolist() == DAILY_FLAG_COLUMNS
    assert artifacts.events.shape[0] == 2
    assert artifacts.daily_flags.shape == (8, len(DAILY_FLAG_COLUMNS))
    summary = json.loads(artifacts.summary_path.read_text(encoding="utf-8"))
    assert summary["number_of_events"] == 2
    assert summary["threshold_type"] == "fixed"
    assert summary["source_metric"] == "anomaly.weighted_mean"
    assert summary["event_tracking_type"] == "univariate temporal detection only"


def test_builder_supports_any_explicit_numerical_bank_metric(tmp_path: Path) -> None:
    _write_bank(tmp_path / "bank.parquet", metric="custom.numeric_metric")
    artifacts = _run(
        tmp_path,
        source_variable="custom_index",
        source_metric="custom.numeric_metric",
        threshold_type="fixed",
        fixed_threshold=2.0,
    )
    assert artifacts.summary["source_variable"] == "custom_index"
    assert artifacts.summary["source_metric"] == "custom.numeric_metric"


def test_existing_outputs_are_preserved_when_processing_fails(tmp_path: Path) -> None:
    _write_bank(tmp_path / "bank.parquet")
    events = tmp_path / "events.parquet"
    flags = tmp_path / "flags.parquet"
    summary = tmp_path / "summary.json"
    events.write_bytes(b"existing events")
    flags.write_bytes(b"existing flags")
    summary.write_text("existing summary", encoding="utf-8")

    with pytest.raises(ValueError, match="compatible only with regional mean SST"):
        _run(tmp_path, threshold_type="daily_climatological", overwrite=True)

    assert events.read_bytes() == b"existing events"
    assert flags.read_bytes() == b"existing flags"
    assert summary.read_text(encoding="utf-8") == "existing summary"


def test_builder_requires_overwrite_for_existing_products(tmp_path: Path) -> None:
    _write_bank(tmp_path / "bank.parquet")
    _run(tmp_path)
    with pytest.raises(FileExistsError, match="--overwrite"):
        _run(tmp_path)
