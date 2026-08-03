"""Resumable tile-wise production builder for observed OSTIA climatologies."""

from __future__ import annotations

from datetime import datetime, timezone
import ctypes
import json
from pathlib import Path
import tempfile
import time
from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from src.climatology_build_plan import ClimatologyBuildPlan, DomainBuildPlan
from src.climatology_checkpoints import (
    CheckpointContract, atomic_write_checkpoint, checkpoint_metadata, sha256_file,
    validate_checkpoint,
)
from src.daily_climatology import build_daily_statistics, validate_daily_climatology
from src.data_loader import find_sst_variable, normalize_spatial_coordinates, normalize_sst
from src.geography import GeographicBounds
from src.real_regional_climatology import (
    build_regional_daily_climatology, merge_regional_partials, regional_partial_series,
)
from src.sst_aggregation import aggregate_sst_to_target_resolution
from src.utils import PROJECT_ROOT, resolve_project_path

SOFTWARE_REVISION = "increment-6c2b1-local"


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def project_relative(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        raise ValueError("Build artifacts must remain inside the project workspace")


def observed_peak_memory_bytes() -> int | None:
    """Return OS peak working set without adding a runtime dependency."""
    if not hasattr(ctypes, "windll"):
        return None
    class ProcessMemoryCounters(ctypes.Structure):
        _fields_ = [
            ("cb", ctypes.c_ulong), ("PageFaultCount", ctypes.c_ulong),
            ("PeakWorkingSetSize", ctypes.c_size_t), ("WorkingSetSize", ctypes.c_size_t),
            ("QuotaPeakPagedPoolUsage", ctypes.c_size_t), ("QuotaPagedPoolUsage", ctypes.c_size_t),
            ("QuotaPeakNonPagedPoolUsage", ctypes.c_size_t), ("QuotaNonPagedPoolUsage", ctypes.c_size_t),
            ("PagefileUsage", ctypes.c_size_t), ("PeakPagefileUsage", ctypes.c_size_t),
        ]
    counters = ProcessMemoryCounters()
    counters.cb = ctypes.sizeof(counters)
    kernel32 = ctypes.windll.kernel32
    kernel32.GetCurrentProcess.restype = ctypes.c_void_p
    process = kernel32.GetCurrentProcess()
    function = getattr(kernel32, "K32GetProcessMemoryInfo", None)
    if function is None:
        function = ctypes.windll.psapi.GetProcessMemoryInfo
    function.argtypes = [
        ctypes.c_void_p, ctypes.POINTER(ProcessMemoryCounters), ctypes.c_ulong,
    ]
    function.restype = ctypes.c_int
    ok = function(process, ctypes.byref(counters), counters.cb)
    return int(counters.PeakWorkingSetSize) if ok else None


def atomic_json(path: Path, payload: Mapping[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        mode="w", encoding="utf-8", prefix=f".{path.stem}-", suffix=".json",
        dir=path.parent, delete=False,
    ) as stream:
        json.dump(payload, stream, indent=2, sort_keys=True)
        stream.write("\n")
        temporary = Path(stream.name)
    temporary.replace(path)


def atomic_parquet(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile(
        prefix=f".{path.stem}-", suffix=".parquet", dir=path.parent, delete=False
    ) as stream:
        temporary = Path(stream.name)
    try:
        frame.to_parquet(temporary, index=False)
        temporary.replace(path)
    finally:
        temporary.unlink(missing_ok=True)


def _resolution(coordinate: xr.DataArray) -> float:
    values = np.asarray(coordinate.values, dtype=float)
    if len(values) < 2:
        raise ValueError("At least two coordinate values are required")
    result = float(np.median(np.diff(values)))
    if result <= 0 or not np.allclose(np.diff(values), result, atol=0.001, rtol=0.0):
        raise ValueError("Source coordinates must form a regular increasing grid")
    return result


def _bounds(dataset: xr.Dataset) -> tuple[float, float, float, float]:
    return (
        float(dataset.longitude.min()), float(dataset.longitude.max()),
        float(dataset.latitude.min()), float(dataset.latitude.max()),
    )


def _coverage(dataset: xr.Dataset) -> float:
    return float(dataset.sst.count() / dataset.sst.size) if dataset.sst.size else 0.0


def _encoding(dataset: xr.Dataset, level: int) -> dict[str, dict[str, Any]]:
    return {
        name: {"zlib": True, "complevel": level, "shuffle": True}
        for name in dataset.data_vars
    }


def prepare_daily_checkpoint(
    source: xr.Dataset, domain: DomainBuildPlan, config: Mapping[str, Any],
) -> tuple[xr.Dataset, float, bool]:
    """Normalize one tile-year and aggregate only when the domain requires it."""
    normalized = normalize_sst(source, find_sst_variable(source))
    normalized = normalize_spatial_coordinates(normalized)[["sst"]].sortby("time")
    source_resolution = _resolution(normalized.latitude)
    if not np.isclose(source_resolution, domain.source_resolution_degrees, atol=0.001):
        raise ValueError(
            f"Source resolution {source_resolution:g} differs from planned {domain.source_resolution_degrees:g}"
        )
    if domain.target_resolution_degrees > source_resolution + 0.001:
        result, aggregation = aggregate_sst_to_target_resolution(
            normalized, domain.target_resolution_degrees,
            minimum_valid_fraction=float(config["real_climatology_build"]["minimum_valid_coverage"]),
        )
        return result[["sst"]], source_resolution, aggregation.aggregation_applied
    if domain.target_resolution_degrees < source_resolution - 0.001:
        raise ValueError("Production climatology never upsamples SST")
    return normalized, source_resolution, False


def _daily_contract(domain: DomainBuildPlan, tile_id: str, year: int, source_version: str | None) -> CheckpointContract:
    build_dataset = "METOFFICE-GLO-SST-L4-REP-OBS-SST"
    return CheckpointContract(
        domain_id=domain.domain_id, tile_id=tile_id, processing_stage="daily_target_grid",
        source_dataset=build_dataset, source_variable="analysed_sst",
        target_resolution=domain.target_resolution_degrees,
        checkpoint_year=year, source_version=source_version,
    )


def _climate_contract(domain: DomainBuildPlan, tile_id: str, source_version: str | None) -> CheckpointContract:
    return CheckpointContract(
        domain_id=domain.domain_id, tile_id=tile_id, processing_stage="tile_daily_climatology",
        source_dataset="METOFFICE-GLO-SST-L4-REP-OBS-SST", source_variable="analysed_sst",
        target_resolution=domain.target_resolution_degrees, source_version=source_version,
    )


def build_tile_climatology(
    checkpoints: Sequence[Path], output: Path, contract: CheckpointContract,
    domain: DomainBuildPlan, config: Mapping[str, Any], *, pilot: bool,
) -> Path:
    opened = xr.open_mfdataset(checkpoints, combine="by_coords", chunks={"time": -1})
    try:
        result = build_daily_statistics(
            opened.sst.chunk({"time": -1}), sampling_half_window_days=5,
            smoothing_window_days=31, calculate_percentiles=True,
        )
        result.attrs.update(
            checkpoint_metadata(
                contract, requested_bounds=domain.bounds, effective_bounds=_bounds(opened),
                source_resolution=domain.source_resolution_degrees,
                valid_coverage=_coverage(opened),
                time_range=(f"{domain.start_year}-01-01", f"{domain.end_year}-12-31"),
                aggregation_applied=domain.target_resolution_degrees > domain.source_resolution_degrees,
                created_at=_utc_now(), software_revision=SOFTWARE_REVISION,
            )
        )
        result.attrs.update(
            climatology_source_mode="observed_reprocessed",
            scientific_use="pipeline_validation_only" if pilot else "experimental_monitoring",
            represents_observed_climatology="true",
            represents_full_reference_climatology="false" if pilot else "true",
            pilot_build="true" if pilot else "false",
            reference_start=domain.start_year, reference_end=domain.end_year,
            reference_period=f"{domain.start_year}-{domain.end_year}",
            climatology_method="daily_smoothed",
            source_product_family="ostia",
            sampling_half_window_days=5, smoothing_window_days=31,
            leap_day_method="stable_366_feb29_bin60_mar01_bin61",
            validation_status="valid",
        )
        atomic_write_checkpoint(
            result, output, contract, encoding=_encoding(result, int(config["real_climatology_build"]["compression_level"])),
            extra_validator=validate_daily_climatology,
        )
    finally:
        opened.close()
    return output


def assemble_domain_climatology(
    tile_paths: Sequence[Path], output: Path, domain: DomainBuildPlan,
    config: Mapping[str, Any], *, pilot: bool, source_version: str | None,
) -> Path:
    if len(tile_paths) != domain.number_of_tiles or len(set(tile_paths)) != len(tile_paths):
        raise ValueError("Expected one unique validated climatology file per planned tile")
    datasets: list[xr.Dataset] = []
    for tile, path in zip(domain.tiles, tile_paths):
        contract = _climate_contract(domain, tile.tile_id, source_version)
        validation = validate_checkpoint(path, contract)
        if not validation.valid:
            raise ValueError(f"Invalid tile climatology {tile.tile_id}: {validation.reason}")
        datasets.append(xr.load_dataset(path))
    try:
        assembled = xr.combine_by_coords(datasets, combine_attrs="override").sortby(["latitude", "longitude"])
        errors = validate_daily_climatology(assembled)
        if errors:
            raise ValueError("Assembled climatology is invalid: " + "; ".join(errors))
        lat = np.asarray(assembled.latitude.values)
        lon = np.asarray(assembled.longitude.values)
        if len(np.unique(lat)) != len(lat) or len(np.unique(lon)) != len(lon):
            raise ValueError("Assembled climatology has duplicate spatial coordinates")
        tolerance = 0.001
        for name, values in (("latitude", lat), ("longitude", lon)):
            differences = np.diff(values)
            if np.any(differences <= 0) or np.any(differences > domain.target_resolution_degrees + tolerance):
                raise ValueError(f"Assembled climatology has an unexpected {name} gap")
        final_contract = CheckpointContract(
            domain_id=domain.domain_id, tile_id="assembled", processing_stage="assembled_pilot" if pilot else "assembled_final",
            source_dataset="METOFFICE-GLO-SST-L4-REP-OBS-SST", source_variable="analysed_sst",
            target_resolution=domain.target_resolution_degrees, source_version=source_version,
        )
        assembled.attrs.update(
            domain_id=domain.domain_id,
            geography_id=domain.domain_id,
            source_resolution=domain.source_resolution_degrees,
            target_resolution=domain.target_resolution_degrees,
            build_manifest=str(Path(domain.resume_path) / "manifests" / "build_manifest.json"),
            validation_status="valid",
            scientific_use="pipeline_validation_only" if pilot else "experimental_monitoring",
            represents_full_reference_climatology="false" if pilot else "true",
            pilot_build="true" if pilot else "false",
        )
        assembled.attrs.update({
            "tile_id": "assembled", "processing_stage": final_contract.processing_stage,
            "source_dataset": final_contract.source_dataset, "source_variable": final_contract.source_variable,
        })
        atomic_write_checkpoint(
            assembled, output, final_contract,
            encoding=_encoding(assembled, int(config["real_climatology_build"]["compression_level"])),
            extra_validator=validate_daily_climatology,
        )
    finally:
        for dataset in datasets:
            dataset.close()
    return output


def execute_build(
    plan: ClimatologyBuildPlan, config: Mapping[str, Any], adapter: Any,
    *, resume: bool, source_version: str | None = None,
) -> dict[str, Any]:
    """Execute an already-approved plan. The caller owns preflight/confirmation gates."""
    if plan.status != "valid":
        raise RuntimeError("resource_limit_exceeded: no download was started")
    if plan.pilot and (plan.start_year, plan.end_year) != (1991, 1992):
        raise RuntimeError("Pilot is hard-limited to 1991-1992")
    build = config["real_climatology_build"]
    status_path = resolve_project_path(build["build_status_report"])
    previous_report: dict[str, Any] = {}
    if plan.pilot and status_path.exists():
        try:
            candidate = json.loads(status_path.read_text(encoding="utf-8"))
            if candidate.get("pilot_build") is True:
                previous_report = candidate
        except (OSError, ValueError, TypeError):
            previous_report = {}
    started = time.perf_counter()
    total_downloaded = 0
    domain_reports: dict[str, Any] = {}
    all_partials: list[pd.DataFrame] = []
    geography = None
    for domain in plan.domains:
        if plan.pilot and domain.number_of_tiles > int(build["pilot"]["maximum_tiles"]):
            raise RuntimeError("Pilot tile limit exceeded")
        output = resolve_project_path(domain.pilot_output_path if plan.pilot else domain.output_path)
        if plan.pilot and "data\\climatology\\pilot" not in str(output).lower().replace("/", "\\"):
            raise RuntimeError("Pilot attempted to publish outside pilot paths")
        tile_outputs: list[Path] = []
        root = resolve_project_path(domain.resume_path)
        manifest_path = root / "manifests" / "build_manifest.json"
        previous_manifest: dict[str, Any] = {}
        if manifest_path.exists():
            try:
                previous_manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
            except (OSError, ValueError, TypeError):
                previous_manifest = {}
        reused = 0
        download_seconds = 0.0
        domain_downloaded = 0
        manifest_checkpoints: list[dict[str, Any]] = []
        domain_start = time.perf_counter()
        for tile in domain.tiles:
            daily_paths: list[Path] = []
            for year in range(domain.start_year, domain.end_year + 1):
                contract = _daily_contract(domain, tile.tile_id, year, source_version)
                daily = root / "daily" / tile.tile_id / f"{year}.nc"
                validation = validate_checkpoint(daily, contract)
                if resume and validation.valid:
                    reused += 1
                    daily_paths.append(daily)
                    manifest_checkpoints.append({
                        "tile_id": tile.tile_id, "year": year, "path": project_relative(daily),
                        "checksum": validation.checksum, "resume": "reused_valid",
                    })
                    continue
                raw = root / "source" / tile.tile_id / f"{year}.nc"
                before = time.perf_counter()
                adapter.subset_to_netcdf(
                    dataset_id=plan.dataset_id, variable=plan.source_variable,
                    bounds=GeographicBounds(*tile.bounds),
                    start_datetime=f"{year}-01-01T00:00:00", end_datetime=f"{year}-12-31T23:59:59",
                    target=raw, mode="pilot" if plan.pilot else "full",
                    compression_level=int(build["compression_level"]),
                )
                download_seconds += time.perf_counter() - before
                downloaded_bytes = raw.stat().st_size
                total_downloaded += downloaded_bytes
                domain_downloaded += downloaded_bytes
                try:
                    with xr.open_dataset(raw) as source:
                        normalized = normalize_spatial_coordinates(
                            normalize_sst(source, find_sst_variable(source))
                        )[["sst"]]
                        if domain.domain_id == "pacific_context" and not plan.pilot:
                            partial = regional_partial_series(
                                normalized, config, tile_id=tile.tile_id, checkpoint_year=year,
                            )
                            if not partial.empty:
                                all_partials.append(partial)
                        prepared, source_resolution, aggregated = prepare_daily_checkpoint(source, domain, config)
                        metadata = checkpoint_metadata(
                            contract, requested_bounds=tile.bounds, effective_bounds=_bounds(prepared),
                            source_resolution=source_resolution, valid_coverage=_coverage(prepared),
                            time_range=(f"{year}-01-01", f"{year}-12-31"),
                            aggregation_applied=aggregated, created_at=_utc_now(),
                            software_revision=SOFTWARE_REVISION,
                        )
                        prepared.attrs.update(metadata)
                        atomic_write_checkpoint(
                            prepared, daily, contract,
                            encoding=_encoding(prepared, int(build["compression_level"])),
                        )
                        manifest_checkpoints.append({
                            "tile_id": tile.tile_id, "year": year, "path": project_relative(daily),
                            "checksum": sha256_file(daily), "resume": "downloaded_and_built",
                        })
                finally:
                    raw.unlink(missing_ok=True)
                daily_paths.append(daily)
            climate = root / "tile_climatologies" / f"{tile.tile_id}.nc"
            climate_contract = _climate_contract(domain, tile.tile_id, source_version)
            validation = validate_checkpoint(climate, climate_contract)
            if not (resume and validation.valid):
                build_tile_climatology(
                    daily_paths, climate, climate_contract, domain, config,
                    pilot=plan.pilot,
                )
            else:
                reused += 1
            tile_outputs.append(climate)
        assemble_domain_climatology(
            tile_outputs, output, domain, config, pilot=plan.pilot, source_version=source_version,
        )
        if plan.pilot and domain.domain_id == "pacific_context":
            from src.geography import load_geography_registry
            geography = load_geography_registry(config)
            for region_id, region in geography.standard_regions.items():
                for year in range(domain.start_year, domain.end_year + 1):
                    partial_path = root / "regional_indices" / "source_series" / f"{region_id}_{year}.parquet"
                    expected_columns = {
                        "date", "region_id", "weighted_sum", "valid_weight",
                        "region_total_weight", "valid_cell_count", "region_total_cell_count",
                    }
                    if resume and partial_path.exists():
                        candidate = pd.read_parquet(partial_path)
                        if (
                            expected_columns <= set(candidate)
                            and set(candidate.region_id) == {region_id}
                            and set(pd.DatetimeIndex(candidate.date).year) == {year}
                            and bool((candidate.valid_weight <= candidate.total_weight + 1.0e-6).all())
                            and bool((candidate.valid_cell_count <= candidate.region_total_cell_count).all())
                        ):
                            all_partials.append(candidate)
                            reused += 1
                            continue
                    raw = root / "regional_indices" / "source" / region_id / f"{year}.nc"
                    before = time.perf_counter()
                    adapter.subset_to_netcdf(
                        dataset_id=plan.dataset_id, variable=plan.source_variable,
                        bounds=region.bounds,
                        start_datetime=f"{year}-01-01T00:00:00",
                        end_datetime=f"{year}-12-31T23:59:59", target=raw, mode="pilot",
                        compression_level=int(build["compression_level"]),
                    )
                    download_seconds += time.perf_counter() - before
                    downloaded_bytes = raw.stat().st_size
                    total_downloaded += downloaded_bytes
                    domain_downloaded += downloaded_bytes
                    try:
                        with xr.open_dataset(raw) as source:
                            normalized = normalize_spatial_coordinates(
                                normalize_sst(source, find_sst_variable(source))
                            )[["sst"]]
                            partial = regional_partial_series(
                                normalized, config, tile_id=f"regional_{region_id}",
                                checkpoint_year=year, region_ids=(region_id,),
                            )
                        if partial.empty:
                            raise RuntimeError(f"Regional pilot subset is empty for {region_id} {year}")
                        atomic_parquet(partial_path, partial)
                        all_partials.append(partial)
                    finally:
                        raw.unlink(missing_ok=True)
        previous_domain_download = int(previous_manifest.get(
            "cumulative_downloaded_bytes", previous_manifest.get("downloaded_bytes", 0)
        ))
        previous_domain_duration = float(previous_manifest.get(
            "cumulative_duration_seconds", previous_manifest.get("duration_seconds", 0.0)
        ))
        domain_duration = time.perf_counter() - domain_start
        atomic_json(manifest_path, {
            "domain_id": domain.domain_id, "dataset_id": plan.dataset_id,
            "source_variable": plan.source_variable, "source_version": source_version,
            "execution_mode": "pilot" if plan.pilot else "full",
            "reference_period": [domain.start_year, domain.end_year],
            "download_strategy": "yearly_spatial_tile", "network_source": "Copernicus Marine",
            "run_downloaded_bytes": domain_downloaded,
            "cumulative_downloaded_bytes": previous_domain_download + domain_downloaded,
            "download_seconds": download_seconds,
            "duration_seconds": domain_duration,
            "cumulative_duration_seconds": previous_domain_duration + domain_duration,
            "checkpoints": manifest_checkpoints, "assembled_output": project_relative(output),
            "assembled_checksum": sha256_file(output), "created_at": _utc_now(),
        })
        domain_reports[domain.domain_id] = {
            "status": "valid", "output": project_relative(output), "tiles": domain.number_of_tiles,
            "resume_reused_checkpoints": reused,
            "calculation_seconds": domain_duration,
            "download_seconds": download_seconds,
            "downloaded_bytes": domain_downloaded,
            "cumulative_downloaded_bytes": previous_domain_download + domain_downloaded,
            "checkpoint_bytes": sum(path.stat().st_size for path in resolve_project_path(domain.resume_path).glob("daily/*/*.nc")),
            "build_manifest": project_relative(manifest_path),
            "estimated_peak_memory_bytes": domain.storage.peak_memory_bytes,
        }
    regional_path = resolve_project_path(
        build["pilot_regional_output"] if plan.pilot else build["regional_output"]
    )
    regional_status = "unavailable_no_intersecting_pacific_tile"
    if all_partials:
        merged = merge_regional_partials(
            pd.concat(all_partials, ignore_index=True),
            source_resolution=float(build["source_resolution_degrees"]), dataset_id=plan.dataset_id,
        )
        regional = build_regional_daily_climatology(merged)
        series_path = resolve_project_path("data/climatology/pilot/work/pacific_context/regional_indices/daily_series.parquet") if plan.pilot else resolve_project_path("data/climatology/work/pacific_context/regional_indices/daily_series.parquet")
        atomic_parquet(series_path, merged)
        atomic_parquet(regional_path, regional)
        complete_regions = set(merged.loc[merged.valid_coverage >= float(build["minimum_valid_coverage"]), "region_id"])
        regional_status = (
            "valid_pipeline_validation_only"
            if plan.pilot and complete_regions == {"nino34", "nino3", "nino12"}
            else "pilot_partial_spatial_coverage" if plan.pilot else "valid"
        )
    duration = time.perf_counter() - started
    previous_download = int(previous_report.get(
        "cumulative_bytes_downloaded", previous_report.get("bytes_downloaded", 0)
    ))
    previous_duration = float(previous_report.get(
        "cumulative_duration_seconds", previous_report.get("duration_seconds", 0.0)
    ))
    current_peak = observed_peak_memory_bytes()
    prior_peak = previous_report.get("observed_peak_memory_bytes")
    observed_peak = max(
        value for value in (current_peak, int(prior_peak) if prior_peak is not None else None)
        if value is not None
    ) if current_peak is not None or prior_peak is not None else None
    report = {
        "schema_version": "1.0.0", "build_status": "pilot_complete" if plan.pilot else "complete",
        "pilot_status": regional_status if plan.pilot else "not_applicable",
        "pilot_build": plan.pilot, "full_build_executed": not plan.pilot,
        "final_climatologies_available": not plan.pilot,
        "dataset_id": plan.dataset_id, "source_variable": plan.source_variable,
        "reference_period_planned": [1991, 2020],
        "reference_period_built": [plan.start_year, plan.end_year],
        "bytes_downloaded": total_downloaded,
        "cumulative_bytes_downloaded": previous_download + total_downloaded,
        "duration_seconds": duration,
        "cumulative_duration_seconds": previous_duration + duration,
        "observed_peak_memory_bytes": observed_peak,
        "peak_memory_measurement": "operating_system_peak_working_set",
        "domains": domain_reports, "regional_climatology": project_relative(regional_path),
        "regional_status": regional_status, "last_validation": _utc_now(),
        "warnings": [
            "Pilot outputs are pipeline validation only and are not an operational climatology"
        ] if plan.pilot else [],
    }
    atomic_json(status_path, report)
    if plan.pilot:
        atomic_json(resolve_project_path(build["pilot_report"]), report)
    return report
