"""Build the cached Niño 1+2 quality-control report and daily series bank."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.daily_climatology import match_climatology, select_climatology
from src.data_loader import (
    COPERNICUS_CACHED_MODE,
    SYNTHETIC_DEMO_MODE,
    load_active_sst_dataset,
    load_sst_dataset,
    subset_nino12,
)
from src.quality_control import save_qc_report, validate_input
from src.series_bank import build_series_bank
from src.utils import configure_logging, load_config, resolve_project_path

LOGGER = logging.getLogger(__name__)


def _load_sst(config: dict, override: Path | None) -> tuple[xr.Dataset, str, Path]:
    data = config["data"]
    region = config["region"]
    if override is not None:
        source = resolve_project_path(override)
        dataset = load_sst_dataset(source, aliases=data["sst_aliases"], create_demo_if_missing=False)
        regional = subset_nino12(dataset)
        assert isinstance(regional, xr.Dataset)
        mode = SYNTHETIC_DEMO_MODE if source == resolve_project_path(data["demo_path"]) else COPERNICUS_CACHED_MODE
        regional.attrs["data_mode"] = mode
        return regional, mode, source
    dataset, mode = load_active_sst_dataset(
        resolve_project_path(data["live_path"]),
        resolve_project_path(data["demo_path"]),
        aliases=data["sst_aliases"],
        demo_options={
            "longitude": tuple(region["longitude"]),
            "latitude": tuple(region["latitude"]),
            "days": data["demo_days"],
            "seed": data["demo_seed"],
        },
    )
    return dataset, mode, Path(str(dataset.attrs.get("source_path", data["live_path"])))


def build(config_path: Path, sst_input: Path | None, climatology_input: Path | None,
          output: Path | None, qc_output: Path | None, overwrite: bool) -> tuple[Path, Path, int, int]:
    config = load_config(resolve_project_path(config_path))
    configure_logging(config["logging"]["level"])
    qc_config = config["quality_control"]
    bank_config = config["series_bank"]
    dataset, data_mode, source = _load_sst(config, sst_input)
    cleaned, weights, report = validate_input(
        dataset,
        variable="sst",
        expected_frequency=qc_config["expected_frequency"],
        duplicate_policy=qc_config["duplicate_policy"],
        maximum_gap_days=qc_config["maximum_gap_days"],
        allow_irregular_intervals=qc_config["allow_irregular_intervals"],
        fail_on_unknown_units=qc_config["fail_on_unknown_units"],
        minimum_temporal_observations=qc_config["minimum_temporal_observations"],
        minimum_valid_coverage=qc_config["minimum_valid_coverage"],
        source_file=source,
        data_mode=data_mode,
    )
    qc_path = resolve_project_path(qc_output or Path("outputs/analytics/qc_report.json"))
    save_qc_report(report, qc_path)
    if report.errors:
        raise RuntimeError("Quality-control validation failed: " + "; ".join(report.errors))

    climate = config["climatology"]
    daily_path = resolve_project_path(climatology_input or climate["daily_path"])
    selection = select_climatology(
        daily_path,
        resolve_project_path(climate["monthly_path"]),
        primary_method=climate["primary_method"],
        fallback_method=climate["fallback_method"],
        allow_monthly_fallback=climate["allow_monthly_fallback"],
        data_mode=data_mode,
    )
    fields = xr.Dataset({"sst": cleaned})
    climatology = selection.dataset
    if climatology is not None:
        mean, std, _ = match_climatology(climatology, cleaned.time)
        anomaly = (cleaned - mean).rename("anomaly")
        anomaly.attrs.update(units="degrees_Celsius", long_name="SST anomaly")
        zscore = xr.where(np.isfinite(std) & (std > 0), anomaly / std, np.nan).rename("zscore")
        zscore.attrs.update(units="1", long_name="standardized SST anomaly")
        fields = fields.assign(anomaly=anomaly, zscore=zscore)
    elif data_mode == COPERNICUS_CACHED_MODE:
        LOGGER.warning("No compatible real climatology; anomaly and z-score series are omitted")

    bank = build_series_bank(
        fields,
        variables=bank_config["variables"],
        weights=weights,
        minimum_valid_coverage=qc_config["minimum_valid_coverage"],
        warm_anomaly_threshold_c=bank_config["warm_anomaly_threshold_c"],
        climatology_method=selection.method,
        data_mode=data_mode,
    )
    output_path = resolve_project_path(output or bank_config["output"])
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Series-bank output exists; use --overwrite: {output_path}")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(prefix=f".{output_path.stem}-", suffix=".parquet", dir=output_path.parent)
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        bank.to_parquet(temporary, index=False)
        temporary.replace(output_path)
    finally:
        temporary.unlink(missing_ok=True)
    return output_path, qc_path, int(bank.metric.nunique()), int(bank.date.nunique())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--sst-input", type=Path)
    parser.add_argument("--climatology-input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--qc-output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output, qc_output, metrics, dates = build(
        args.config, args.sst_input, args.climatology_input,
        args.output, args.qc_output, args.overwrite,
    )
    print(f"Series bank: {output}")
    print(f"QC report: {qc_output}")
    print(f"Unique metrics: {metrics}")
    print(f"Dates: {dates}")


if __name__ == "__main__":
    main()
