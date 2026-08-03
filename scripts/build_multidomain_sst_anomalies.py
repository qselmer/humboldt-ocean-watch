"""Build multidomain SST anomaly products exclusively from local prepared files."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable

import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_multidomain_sst_snapshot import _netcdf_safe, _promote_atomically
from src.multidomain_climatology import load_multidomain_climatologies
from src.multidomain_sst import load_multidomain_sst
from src.multidomain_sst_anomalies import calculate_multidomain_sst_anomalies
from src.sst_regional_climatology import (
    REGIONAL_CLIMATOLOGY_COLUMNS,
    build_nino_regional_climatology,
)
from src.sst_regional_indices import INDEX_COLUMNS, calculate_nino_region_sst
from src.utils import load_config, resolve_project_path


DEFAULT_DOMAINS = ("pacific_context", "humboldt_coastal")


def _paths(config: dict[str, Any], directory: Path | None) -> dict[str, Path]:
    outputs = config["sst"]["anomaly_outputs"]
    climate_output = config["sst"]["regional_indices"]["climatology_output"]
    if directory is None:
        processed = resolve_project_path(outputs["processed_directory"])
        return {
            "status": resolve_project_path(outputs["status"]),
            "regional_climatology": resolve_project_path(climate_output),
            "regional_anomalies": resolve_project_path(outputs["regional_indices"]),
            "pacific_context": processed / "pacific_context_latest_anomaly.nc",
            "humboldt_coastal": processed / "humboldt_coastal_latest_anomaly.nc",
        }
    return {
        "status": directory / "multidomain_anomaly_status.json",
        "regional_climatology": directory / "nino_region_climatology.parquet",
        "regional_anomalies": directory / "nino_region_sst_anomaly.parquet",
        "pacific_context": directory / "processed" / "pacific_context_latest_anomaly.nc",
        "humboldt_coastal": directory / "processed" / "humboldt_coastal_latest_anomaly.nc",
    }


def _json(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def build_multidomain_anomalies(
    config: dict[str, Any],
    *,
    domains: tuple[str, ...] = DEFAULT_DOMAINS,
    allow_demo: bool = False,
    overwrite: bool = False,
    dry_run: bool = False,
    output_directory: Path | None = None,
) -> dict[str, Any]:
    """Build a failure-isolated, atomically published local anomaly bundle."""
    if not domains or len(set(domains)) != len(domains):
        raise ValueError("--domains must contain unique configured domain IDs")
    paths = _paths(config, output_directory)
    sst = load_multidomain_sst(config, domains, allow_demo=allow_demo)
    climatologies = load_multidomain_climatologies(config, sst, domains)
    anomalies = calculate_multidomain_sst_anomalies(
        config, sst, climatologies, domains
    )
    plan = {
        "dry_run": dry_run,
        "offline": True,
        "network_access": False,
        "domains": {
            domain_id: {
                "sst_status": sst[domain_id].status,
                "sst_mode": sst[domain_id].source_mode,
                "climatology": climatologies[domain_id].to_status_dict(),
                "anomaly": anomalies[domain_id].to_status_dict(),
            }
            for domain_id in domains
        },
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    if dry_run:
        return plan

    pacific_climate = climatologies.get("pacific_context")
    pacific_sst = sst.get("pacific_context")
    if (
        pacific_climate is not None
        and pacific_climate.dataset is not None
        and pacific_climate.compatibility.compatible
    ):
        regional_climatology = build_nino_regional_climatology(
            pacific_climate.dataset, config
        )
    else:
        regional_climatology = pd.DataFrame(columns=REGIONAL_CLIMATOLOGY_COLUMNS)
    if pacific_sst is not None and pacific_sst.dataset is not None:
        regional_anomalies = calculate_nino_region_sst(
            pacific_sst.dataset,
            config,
            source_domain="pacific_context",
            source_mode=pacific_sst.source_mode,
            regional_climatology=regional_climatology,
        )
    else:
        regional_anomalies = pd.DataFrame(columns=INDEX_COLUMNS)

    created = pd.Timestamp.now(tz="UTC").isoformat()
    payload = {
        "validation": "complete",
        "created_at_utc": created,
        "offline": True,
        "network_access": False,
        "domains": plan["domains"],
        "regional_climatology_rows": int(len(regional_climatology)),
        "regional_anomaly_rows": int(len(regional_anomalies)),
        "regional_anomaly_status": (
            {}
            if regional_anomalies.empty
            else {
                str(region_id): sorted(group.anomaly_status.astype(str).unique().tolist())
                for region_id, group in regional_anomalies.groupby("region_id", sort=True)
            }
        ),
        "official_enso_classification": False,
        "oni_calculated": False,
        "warnings": [
            warning
            for result in anomalies.values()
            for warning in result.warnings
        ],
        "errors": [
            error for result in anomalies.values() for error in result.errors
        ],
    }
    writers: list[tuple[Path, Callable[[Path], None]]] = [
        (paths["status"], lambda path: path.write_text(_json(payload), encoding="utf-8")),
        (
            paths["regional_climatology"],
            lambda path: regional_climatology.to_parquet(path, index=False),
        ),
        (
            paths["regional_anomalies"],
            lambda path: regional_anomalies.to_parquet(path, index=False),
        ),
    ]
    for domain_id in domains:
        result = anomalies[domain_id]
        if result.dataset is None:
            continue
        latest = _netcdf_safe(result.dataset.isel(time=[-1]))
        writers.append(
            (paths[domain_id], lambda path, data=latest: data.to_netcdf(path))
        )
    _promote_atomically(writers, overwrite=overwrite)
    return {**payload, "outputs": {key: str(value) for key, value in paths.items()}}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build offline multidomain SST anomalies.")
    parser.add_argument("--domains", nargs="+", choices=DEFAULT_DOMAINS, default=list(DEFAULT_DOMAINS))
    parser.add_argument("--allow-demo", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--output-directory", type=Path)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    report = build_multidomain_anomalies(
        load_config(),
        domains=tuple(args.domains),
        allow_demo=args.allow_demo,
        overwrite=args.overwrite,
        dry_run=args.dry_run,
        output_directory=args.output_directory,
    )
    print(_json(report), end="")
    return report


if __name__ == "__main__":
    main()
