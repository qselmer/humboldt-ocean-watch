"""Build daily SST-anomaly spatial representativeness from cached data."""

from __future__ import annotations

import argparse
import os
import sys
import tempfile
from pathlib import Path

import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_series_bank import _load_sst
from src.daily_climatology import match_climatology, select_climatology
from src.quality_control import cosine_latitude_weights
from src.representativeness import (
    RepresentativenessThresholds,
    build_representativeness_table,
)
from src.utils import configure_logging, load_config, resolve_project_path


def _atomic_parquet(table: pd.DataFrame, output: Path) -> None:
    output.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{output.stem}-", suffix=".parquet", dir=output.parent,
    )
    os.close(descriptor)
    temporary = Path(temporary_name)
    try:
        table.to_parquet(temporary, index=False)
        temporary.replace(output)
    finally:
        temporary.unlink(missing_ok=True)


def build(
    *,
    config_path: Path,
    sst_input: Path | None,
    climatology_input: Path | None,
    output: Path | None,
    overwrite: bool,
) -> tuple[Path, pd.DataFrame]:
    """Build the table without any network access or remote API calls."""
    config = load_config(resolve_project_path(config_path))
    configure_logging(config["logging"]["level"])
    representativeness_config = config["representativeness"]
    thresholds = RepresentativenessThresholds.from_mapping(representativeness_config)
    output_path = resolve_project_path(output or representativeness_config["output"])
    if output_path.exists() and not overwrite:
        raise FileExistsError(f"Representativeness output exists; use --overwrite: {output_path}")

    dataset, data_mode, _ = _load_sst(config, sst_input)
    climatology_config = config["climatology"]
    selection = select_climatology(
        resolve_project_path(climatology_input or climatology_config["daily_path"]),
        resolve_project_path(climatology_config["monthly_path"]),
        primary_method=climatology_config["primary_method"],
        fallback_method=climatology_config["fallback_method"],
        allow_monthly_fallback=climatology_config["allow_monthly_fallback"],
        data_mode=data_mode,
    )
    if selection.dataset is None or selection.method is None:
        raise RuntimeError(
            "No compatible climatology is available; SST-anomaly representativeness cannot be calculated"
        )
    climatological_mean, _, _ = match_climatology(selection.dataset, dataset.time)
    anomaly = (dataset.sst - climatological_mean).rename("anomaly")
    anomaly.attrs.update(units="degrees_Celsius", long_name="SST anomaly")
    weights = cosine_latitude_weights(anomaly)
    spatial_config = config["spatial_features"]
    table = build_representativeness_table(
        anomaly,
        thresholds,
        climatology_method=selection.method,
        data_mode=data_mode,
        weights=weights,
        connectivity=spatial_config["connectivity"],
        minimum_patch_cells=int(spatial_config["minimum_patch_cells"]),
    )
    _atomic_parquet(table, output_path)
    return output_path, table


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--sst-input", type=Path)
    parser.add_argument("--climatology-input", type=Path)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    output, table = build(
        config_path=args.config,
        sst_input=args.sst_input,
        climatology_input=args.climatology_input,
        output=args.output,
        overwrite=args.overwrite,
    )
    print(f"Representativeness: {output} {table.shape}")
    print(f"Class counts: {table['class'].value_counts(dropna=False).to_dict()}")
    print(f"Dates not calculated: {int((table.status == 'not_calculated').sum())}")


if __name__ == "__main__":
    main()
