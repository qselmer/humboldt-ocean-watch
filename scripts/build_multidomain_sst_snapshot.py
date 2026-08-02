"""Build validated multidomain SST products from local files only."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from typing import Any, Callable
from uuid import uuid4

import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.multidomain_sst import SSTDomainResult, load_multidomain_sst
from src.sst_domains import load_sst_domain_specs
from src.sst_regional_indices import INDEX_COLUMNS, calculate_nino_region_sst
from src.utils import load_config, resolve_project_path


DEFAULT_DOMAINS = ("pacific_context", "humboldt_coastal")


def _output_paths(config: dict[str, Any], directory: Path | None) -> dict[str, Path]:
    outputs = config["sst"]["outputs"]
    index_output = config["sst"]["regional_indices"]["output"]
    if directory is None:
        processed = resolve_project_path(outputs["processed_directory"])
        return {
            "status": resolve_project_path(outputs["status"]),
            "snapshot": resolve_project_path(outputs["snapshot"]),
            "indices": resolve_project_path(index_output),
            "pacific_context": processed / "pacific_context_latest.nc",
            "humboldt_coastal": processed / "humboldt_coastal_latest.nc",
        }
    return {
        "status": directory / "multidomain_sst_status.json",
        "snapshot": directory / "multidomain_sst_snapshot.json",
        "indices": directory / "nino_region_sst.parquet",
        "pacific_context": directory / "processed" / "pacific_context_latest.nc",
        "humboldt_coastal": directory / "processed" / "humboldt_coastal_latest.nc",
    }


def _json_text(payload: Any) -> str:
    return json.dumps(payload, indent=2, ensure_ascii=False, allow_nan=False) + "\n"


def _netcdf_safe(dataset: xr.Dataset) -> xr.Dataset:
    result = dataset.copy()
    result.attrs = {
        key: (str(value).lower() if isinstance(value, bool) else value)
        for key, value in result.attrs.items()
    }
    for variable in result.variables:
        result[variable].attrs = {
            key: (str(value).lower() if isinstance(value, bool) else value)
            for key, value in result[variable].attrs.items()
        }
    return result


def _promote_atomically(
    writers: list[tuple[Path, Callable[[Path], None]]], *, overwrite: bool
) -> None:
    existing = [target for target, _ in writers if target.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Products already exist; use --overwrite: " + ", ".join(map(str, existing))
        )
    token = uuid4().hex
    staged: list[tuple[Path, Path]] = []
    backups: list[tuple[Path, Path]] = []
    promoted: list[Path] = []
    try:
        for target, writer in writers:
            target.parent.mkdir(parents=True, exist_ok=True)
            suffix = target.suffix or ".tmp"
            temporary = target.with_name(f".{target.stem}.{token}.tmp{suffix}")
            writer(temporary)
            if not temporary.exists() or temporary.stat().st_size == 0:
                raise OSError(f"Staged product is empty: {temporary}")
            staged.append((temporary, target))
        for _, target in staged:
            if target.exists():
                backup = target.with_name(f".{target.name}.{token}.backup")
                target.replace(backup)
                backups.append((backup, target))
        for temporary, target in staged:
            temporary.replace(target)
            promoted.append(target)
    except Exception:
        for target in reversed(promoted):
            if target.exists():
                target.unlink()
        for backup, target in backups:
            if backup.exists():
                backup.replace(target)
        raise
    finally:
        for temporary, _ in staged:
            if temporary.exists():
                temporary.unlink()
        for backup, _ in backups:
            if backup.exists():
                backup.unlink()


def _domain_summary(results: dict[str, SSTDomainResult]) -> dict[str, Any]:
    return {domain_id: result.to_status_dict() for domain_id, result in results.items()}


def build_multidomain_snapshot(
    config: dict[str, Any],
    *,
    domains: tuple[str, ...] = DEFAULT_DOMAINS,
    allow_demo: bool = False,
    overwrite: bool = False,
    output_directory: Path | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Validate all stages in memory, then promote the complete product bundle."""
    if not domains or len(set(domains)) != len(domains):
        raise ValueError("--domains must contain one or more unique domain IDs")
    specs = load_sst_domain_specs(config)
    paths = _output_paths(config, output_directory)
    plan = {
        "dry_run": dry_run,
        "network_access": False,
        "domains": {
            domain_id: {
                "live_path": str(resolve_project_path(specs.domains[domain_id].live_path)),
                "demo_path": str(resolve_project_path(specs.domains[domain_id].demo_path)),
                "bounds": specs.domains[domain_id].bounds.to_dict(),
                "target_resolution_degrees": specs.domains[domain_id].target_resolution_degrees,
                "allow_demo": allow_demo,
            }
            for domain_id in domains
        },
        "outputs": {key: str(value) for key, value in paths.items()},
    }
    results = load_multidomain_sst(config, domains, allow_demo=allow_demo)
    plan["statuses"] = {
        domain_id: result.status for domain_id, result in results.items()
    }
    if dry_run:
        return plan
    unavailable = {
        domain_id: result
        for domain_id, result in results.items()
        if result.dataset is None or result.status not in {"available_live", "available_demo"}
    }
    if unavailable:
        details = "; ".join(
            f"{key}: {value.status} ({'; '.join(value.errors or value.warnings)})"
            for key, value in unavailable.items()
        )
        raise RuntimeError(f"Multidomain SST validation failed before writing: {details}")
    pacific = results.get("pacific_context")
    if pacific is not None:
        if pacific.source_dataset is None:
            raise RuntimeError("Pacific context source grid is unavailable for regional SST")
        indices = calculate_nino_region_sst(
            pacific.source_dataset,
            config,
            source_domain="pacific_context",
            source_mode=pacific.source_mode,
        )
        if indices.empty:
            raise RuntimeError("Regional SST calculation returned no rows")
    else:
        indices = pd.DataFrame(columns=INDEX_COLUMNS)

    domain_status = _domain_summary(results)
    created = pd.Timestamp.now(tz="UTC").isoformat()
    status_payload = {
        "validation_status": "valid",
        "created_at_utc": created,
        "offline": True,
        "domains": domain_status,
        "warnings": [warning for result in results.values() for warning in result.warnings],
        "errors": [],
    }
    region_coverage = {
        region_id: {
            "minimum": float(group.valid_coverage.min()),
            "maximum": float(group.valid_coverage.max()),
            "anomaly_status": sorted(group.anomaly_status.unique().tolist()),
        }
        for region_id, group in indices.groupby("region_id", sort=True)
    }
    snapshot_payload = {
        "validation_status": "valid",
        "created_at_utc": created,
        "domains_loaded": list(results),
        "source_modes": {
            key: value.source_mode for key, value in results.items()
        },
        "latest_dates": {
            key: value.latest_date for key, value in results.items()
        },
        "domains": domain_status,
        "nino_region_coverage": region_coverage,
        "regional_index_rows": int(len(indices)),
        "anomalies_calculated": False,
        "anomaly_note": "No compatible Niño 3.4 or Niño 3 climatology is configured.",
        "official_enso_classification": False,
        "warnings": status_payload["warnings"],
        "errors": [],
    }

    writers: list[tuple[Path, Callable[[Path], None]]] = [
        (paths["status"], lambda path: path.write_text(_json_text(status_payload), encoding="utf-8")),
        (paths["snapshot"], lambda path: path.write_text(_json_text(snapshot_payload), encoding="utf-8")),
        (paths["indices"], lambda path: indices.to_parquet(path, index=False)),
    ]
    for domain_id in domains:
        result = results[domain_id]
        assert result.dataset is not None
        latest = _netcdf_safe(result.dataset.isel(time=[-1]))
        writers.append((paths[domain_id], lambda path, data=latest: data.to_netcdf(path)))
    _promote_atomically(writers, overwrite=overwrite)
    return {
        **snapshot_payload,
        "outputs": {key: str(value) for key, value in paths.items()},
    }


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build local multidomain SST snapshots.")
    parser.add_argument("--domains", nargs="+", choices=DEFAULT_DOMAINS, default=list(DEFAULT_DOMAINS))
    parser.add_argument("--allow-demo", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict[str, Any]:
    args = parse_args(argv)
    report = build_multidomain_snapshot(
        load_config(),
        domains=tuple(args.domains),
        allow_demo=args.allow_demo,
        overwrite=args.overwrite,
        output_directory=args.output_directory,
        dry_run=args.dry_run,
    )
    print(_json_text(report), end="")
    return report


if __name__ == "__main__":
    main()
