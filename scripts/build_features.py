"""Build cached temporal and daily spatial feature-engine outputs."""

from __future__ import annotations

import argparse
import logging
import os
import sys
import tempfile
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_series_bank import _load_sst
from src.daily_climatology import match_climatology, select_climatology
from src.data_loader import COPERNICUS_CACHED_MODE
from src.spatial_features import build_spatial_feature_table
from src.temporal_features import build_temporal_feature_table
from src.utils import configure_logging, load_config, resolve_project_path

LOGGER = logging.getLogger(__name__)


def _atomic_parquet(frame: pd.DataFrame, destination: Path) -> None:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.stem}-", suffix=".parquet", dir=destination.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)


def _analysis_fields(
    config: dict,
    *,
    sst_input: Path | None,
    climatology_input: Path | None,
) -> xr.Dataset:
    dataset, data_mode, _ = _load_sst(config, sst_input)
    fields = xr.Dataset({"sst": dataset["sst"]})
    climate = config["climatology"]
    selection = select_climatology(
        resolve_project_path(climatology_input or climate["daily_path"]),
        resolve_project_path(climate["monthly_path"]),
        primary_method=climate["primary_method"],
        fallback_method=climate["fallback_method"],
        allow_monthly_fallback=climate["allow_monthly_fallback"],
        data_mode=data_mode,
    )
    if selection.dataset is not None:
        mean, standard_deviation, _ = match_climatology(selection.dataset, dataset.time)
        anomaly = (dataset.sst - mean).rename("anomaly")
        anomaly.attrs.update(units="degrees_Celsius", long_name="SST anomaly")
        zscore = xr.where(
            np.isfinite(standard_deviation) & (standard_deviation > 0),
            anomaly / standard_deviation,
            np.nan,
        ).rename("zscore")
        zscore.attrs.update(units="1", long_name="standardized SST anomaly")
        fields = fields.assign(anomaly=anomaly, zscore=zscore)
    elif data_mode == COPERNICUS_CACHED_MODE:
        LOGGER.warning("No compatible real climatology; anomaly and z-score features are omitted")
    return fields


def build(
    *,
    config_path: Path,
    series_bank_path: Path | None,
    sst_input: Path | None,
    climatology_input: Path | None,
    temporal_output: Path | None,
    spatial_output: Path | None,
    overwrite: bool,
) -> tuple[Path, Path, pd.DataFrame, pd.DataFrame]:
    config = load_config(resolve_project_path(config_path))
    configure_logging(config["logging"]["level"])
    temporal_config = config["temporal_features"]
    spatial_config = config["spatial_features"]
    temporal_path = resolve_project_path(temporal_output or temporal_config["output"])
    spatial_path = resolve_project_path(spatial_output or spatial_config["output"])
    existing = [str(path) for path in (temporal_path, spatial_path) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError("Feature output exists; use --overwrite: " + ", ".join(existing))

    bank_path = resolve_project_path(series_bank_path or config["series_bank"]["output"])
    if not bank_path.exists():
        raise FileNotFoundError(
            f"Cached series bank not found: {bank_path}. Run scripts/build_series_bank.py first."
        )
    bank = pd.read_parquet(bank_path)
    required_bank_columns = {"date", "metric", "value", "unit"}
    missing_columns = required_bank_columns - set(bank.columns)
    if missing_columns:
        raise ValueError(f"Series bank is missing columns: {sorted(missing_columns)}")
    bank["date"] = pd.to_datetime(bank["date"], errors="raise")
    source_metrics = [
        f"{variable}.weighted_mean"
        for variable in config["series_bank"]["variables"]
        if f"{variable}.weighted_mean" in set(bank.metric)
    ]
    if not source_metrics:
        raise ValueError("Series bank has no configured weighted-mean source metrics")
    windows: list[int | str] = [int(value) for value in temporal_config["windows_days"]]
    windows.append("all")
    temporal = build_temporal_feature_table(
        bank,
        source_metrics=source_metrics,
        windows=windows,
        minimum_observations=int(temporal_config["minimum_observations"]),
        instability_threshold=float(temporal_config["instability_threshold_c"]),
        entropy_bins=int(temporal_config["entropy_bins"]),
        spectral_minimum_observations=int(temporal_config["spectral_minimum_observations"]),
    )

    fields = _analysis_fields(
        config, sst_input=sst_input, climatology_input=climatology_input,
    )
    spatial = build_spatial_feature_table(
        fields,
        variables=config["series_bank"]["variables"],
        threshold=float(spatial_config["threshold_c"]),
        minimum_valid_coverage=float(spatial_config["minimum_valid_coverage"]),
        connectivity=spatial_config["connectivity"],
        minimum_patch_cells=int(spatial_config["minimum_patch_cells"]),
        moran_weights=spatial_config["moran_weights"],
        local_window_cells=int(spatial_config["local_window_cells"]),
    )
    _atomic_parquet(temporal, temporal_path)
    _atomic_parquet(spatial, spatial_path)
    return temporal_path, spatial_path, temporal, spatial


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--series-bank", type=Path)
    parser.add_argument("--sst-input", type=Path)
    parser.add_argument("--climatology-input", type=Path)
    parser.add_argument("--temporal-output", type=Path)
    parser.add_argument("--spatial-output", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    temporal_path, spatial_path, temporal, spatial = build(
        config_path=args.config,
        series_bank_path=args.series_bank,
        sst_input=args.sst_input,
        climatology_input=args.climatology_input,
        temporal_output=args.temporal_output,
        spatial_output=args.spatial_output,
        overwrite=args.overwrite,
    )
    print(f"Temporal features: {temporal_path} {temporal.shape}")
    print(f"Spatial features: {spatial_path} {spatial.shape}")
    print(f"Temporal status counts: {temporal.status.value_counts(dropna=False).to_dict()}")
    print(f"Spatial status counts: {spatial.status.value_counts(dropna=False).to_dict()}")


if __name__ == "__main__":
    main()
