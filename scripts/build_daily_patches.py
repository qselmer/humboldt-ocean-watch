"""Build daily local thermal-patch labels and metrics from cached SST data."""

from __future__ import annotations

import argparse
import json
import logging
import os
import sys
import tempfile
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_series_bank import _load_sst
from src.daily_climatology import calendar_dayofyear, match_climatology
from src.daily_diagnosis import available_dates
from src.export_utils import dumps_json_safe
from src.patch_detection import (
    PatchDetectionConfig,
    identify_patches,
    threshold_definition,
)
from src.patch_metrics import (
    DAILY_PATCH_SUMMARY_COLUMNS,
    PATCH_COLUMNS,
    characterize_daily_patches,
    daily_summary_frame,
)
from src.patch_validation import CachedClimatologySelection, open_cached_climatology
from src.utils import configure_logging, load_config, resolve_project_path

LOGGER = logging.getLogger(__name__)

SOURCE_ALIASES = {
    "anomaly": "anomaly",
    "sst_anomaly": "anomaly",
    "zscore": "zscore",
    "standardized_anomaly": "zscore",
    "sst": "sst",
    "sea_surface_temperature": "sst",
}


@dataclass(frozen=True)
class PatchBuildArtifacts:
    patches_path: Path
    daily_summary_path: Path
    labels_path: Path | None
    json_summary_path: Path
    patches: pd.DataFrame
    daily_summary: pd.DataFrame
    labels: xr.Dataset
    summary: dict[str, Any]


def _temporary_path(destination: Path, suffix: str) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.stem}-", suffix=suffix, dir=destination.parent
    )
    os.close(descriptor)
    return Path(name)


def _canonical_source(source_variable: str) -> str:
    try:
        return SOURCE_ALIASES[source_variable]
    except KeyError as exc:
        raise ValueError(
            "source_variable must be anomaly, zscore/standardized_anomaly, or sst"
        ) from exc


def _validate_definition(source_variable: str, threshold_type: str) -> None:
    if threshold_type == "fixed" and source_variable != "anomaly":
        raise ValueError("Fixed thermal patches require source_variable='anomaly'")
    if threshold_type == "standardized" and source_variable != "zscore":
        raise ValueError("Standardized thermal patches require source_variable='zscore'")
    if threshold_type == "daily_climatological" and source_variable != "sst":
        raise ValueError("Daily climatological percentile patches require source_variable='sst'")


def _match_date_fields(
    dataset: xr.Dataset,
    climatology: xr.Dataset,
    position: int,
    *,
    source_variable: str,
    threshold_type: str,
    scalar_threshold: float,
    climatological_threshold_variable: str,
) -> tuple[xr.DataArray, float | xr.DataArray]:
    """Calculate only the selected date's source and active threshold fields."""
    current = dataset.sst.isel(time=position, drop=True).load()
    selected_time = xr.DataArray(
        [dataset.time.values[position]], dims="time", coords={"time": [dataset.time.values[position]]}
    )
    mean, standard_deviation, _ = match_climatology(climatology, selected_time)
    mean = mean.isel(time=0, drop=True).load()
    standard_deviation = standard_deviation.isel(time=0, drop=True).load()
    try:
        current, mean, standard_deviation = xr.align(
            current, mean, standard_deviation, join="exact"
        )
    except ValueError as exc:
        raise ValueError("SST and climatology spatial coordinates are incompatible") from exc
    anomaly = (current - mean).rename("anomaly")
    anomaly.attrs.update(units="degrees_Celsius", long_name="daily SST anomaly")
    zscore = xr.where(
        np.isfinite(standard_deviation) & (standard_deviation > 0),
        anomaly / standard_deviation,
        np.nan,
    ).rename("zscore")
    zscore.attrs.update(units="1", long_name="standardized daily SST anomaly")
    if threshold_type == "daily_climatological":
        if climatological_threshold_variable not in climatology:
            raise ValueError(
                f"Daily climatology lacks threshold variable {climatological_threshold_variable!r}"
            )
        stable_day = int(calendar_dayofyear(selected_time).item())
        threshold = climatology[climatological_threshold_variable].sel(
            climatological_day=stable_day
        ).load()
        try:
            current, threshold = xr.align(current, threshold, join="exact")
        except ValueError as exc:
            raise ValueError("SST and daily climatological threshold coordinates are incompatible") from exc
        return current.rename("sst"), threshold.rename("threshold")
    source = anomaly if source_variable == "anomaly" else zscore
    return source, scalar_threshold


def _label_dataset(
    dataset: xr.Dataset,
    patch_ids: np.ndarray,
    threshold_masks: np.ndarray,
    valid_masks: np.ndarray,
    *,
    definition: str,
    config: PatchDetectionConfig,
    climatology_method: str,
    data_mode: str,
    minimum_patch_area_km2: float,
) -> xr.Dataset:
    labels = xr.Dataset(
        {
            "patch_id": (
                ("time", "latitude", "longitude"), patch_ids.astype(np.int32, copy=False)
            ),
            "threshold_mask": (
                ("time", "latitude", "longitude"), threshold_masks.astype(np.uint8, copy=False)
            ),
            "valid_ocean_mask": (
                ("time", "latitude", "longitude"), valid_masks.astype(np.uint8, copy=False)
            ),
        },
        coords={
            "time": dataset.time.values,
            "latitude": dataset.latitude.values,
            "longitude": dataset.longitude.values,
        },
        attrs={
            "title": "Humboldt Ocean Watch daily local thermal patch labels",
            "threshold_definition": definition,
            "direction": config.direction,
            "connectivity": config.connectivity,
            "minimum_patch_cells": config.minimum_patch_cells,
            "minimum_patch_area_km2": minimum_patch_area_km2,
            "climatology_method": climatology_method,
            "data_mode": data_mode,
            "experimental_product": "true",
            "patch_id_scope": "local to each date; labels are not persistent track IDs",
            "spatiotemporal_tracking": "not performed",
        },
    )
    labels.patch_id.attrs.update(
        long_name="local daily thermal patch identifier",
        flag_values="0 means no retained patch; positive integers are date-local labels",
        patch_id_scope="local to each date; not a persistent track identifier",
    )
    labels.threshold_mask.attrs.update(
        long_name="unfiltered valid-ocean threshold exceedance mask",
        flag_values="0, 1",
    )
    labels.valid_ocean_mask.attrs.update(
        long_name="date-specific valid ocean mask",
        flag_values="0, 1",
    )
    return labels


def _validate_outputs(
    patches_path: Path,
    daily_path: Path,
    labels_path: Path | None,
    summary_path: Path,
    *,
    expected_dates: int,
) -> None:
    if pd.read_parquet(patches_path).columns.tolist() != PATCH_COLUMNS:
        raise RuntimeError("Temporary patch-level table failed schema validation")
    daily = pd.read_parquet(daily_path)
    if daily.columns.tolist() != DAILY_PATCH_SUMMARY_COLUMNS or len(daily) != expected_dates:
        raise RuntimeError("Temporary daily patch summary failed schema/date validation")
    if labels_path is not None:
        with xr.open_dataset(labels_path) as labels:
            if set(labels.data_vars) != {"patch_id", "threshold_mask", "valid_ocean_mask"}:
                raise RuntimeError("Temporary patch-label cube failed variable validation")
            if labels.sizes.get("time") != expected_dates:
                raise RuntimeError("Temporary patch-label cube failed time validation")
    json.loads(summary_path.read_text(encoding="utf-8"))


def write_outputs_atomically(
    patches: pd.DataFrame,
    daily_summary: pd.DataFrame,
    labels: xr.Dataset,
    summary: dict[str, Any],
    *,
    patches_path: Path,
    daily_summary_path: Path,
    labels_path: Path | None,
    json_summary_path: Path,
    compression_level: int,
) -> dict[str, int]:
    """Validate all temporary products before replacing any requested output."""
    temporary_patches = _temporary_path(patches_path, ".parquet")
    temporary_daily = _temporary_path(daily_summary_path, ".parquet")
    temporary_json = _temporary_path(json_summary_path, ".json")
    temporary_labels = _temporary_path(labels_path, ".nc") if labels_path is not None else None
    temporary = [temporary_patches, temporary_daily, temporary_json]
    if temporary_labels is not None:
        temporary.append(temporary_labels)
    try:
        patches.to_parquet(temporary_patches, index=False)
        daily_summary.to_parquet(temporary_daily, index=False)
        if temporary_labels is not None:
            encoding = {
                "patch_id": {
                    "dtype": "int32", "zlib": True, "complevel": compression_level,
                    "chunksizes": (1, labels.sizes["latitude"], labels.sizes["longitude"]),
                },
                "threshold_mask": {
                    "dtype": "uint8", "zlib": True, "complevel": compression_level,
                    "chunksizes": (1, labels.sizes["latitude"], labels.sizes["longitude"]),
                },
                "valid_ocean_mask": {
                    "dtype": "uint8", "zlib": True, "complevel": compression_level,
                    "chunksizes": (1, labels.sizes["latitude"], labels.sizes["longitude"]),
                },
            }
            labels.to_netcdf(temporary_labels, engine="netcdf4", encoding=encoding)
        output_sizes = {
            "patches_parquet": temporary_patches.stat().st_size,
            "daily_summary_parquet": temporary_daily.stat().st_size,
            "labels_netcdf": temporary_labels.stat().st_size if temporary_labels else 0,
        }
        summary_to_write = {**summary, "approximate_output_sizes_bytes": output_sizes}
        temporary_json.write_text(dumps_json_safe(summary_to_write) + "\n", encoding="utf-8")
        _validate_outputs(
            temporary_patches,
            temporary_daily,
            temporary_labels,
            temporary_json,
            expected_dates=len(daily_summary),
        )
        temporary_patches.replace(patches_path)
        temporary_daily.replace(daily_summary_path)
        if temporary_labels is not None and labels_path is not None:
            temporary_labels.replace(labels_path)
        temporary_json.replace(json_summary_path)
        return output_sizes
    finally:
        for path in temporary:
            path.unlink(missing_ok=True)


def build(
    *,
    config_path: Path,
    sst_input: Path | None,
    climatology_input: Path | None,
    source_variable: str | None,
    threshold_type: str | None,
    fixed_threshold: float | None,
    direction: str | None,
    connectivity: str | None,
    minimum_patch_cells: int | None,
    minimum_patch_area_km2: float | None,
    patches_output: Path | None,
    daily_summary_output: Path | None,
    labels_output: Path | None,
    json_summary_output: Path | None,
    overwrite: bool,
) -> PatchBuildArtifacts:
    started = perf_counter()
    config_values = load_config(resolve_project_path(config_path))
    configure_logging(config_values["logging"]["level"])
    patch_values = config_values["patch_detection"]
    effective_threshold_type = threshold_type or str(patch_values["threshold_type"])
    if source_variable is not None:
        effective_source = _canonical_source(source_variable)
    elif threshold_type is not None and threshold_type != patch_values["threshold_type"]:
        effective_source = {
            "fixed": "anomaly",
            "standardized": "zscore",
            "daily_climatological": "sst",
        }[effective_threshold_type]
    else:
        effective_source = _canonical_source(str(patch_values["source_variable"]))
    _validate_definition(effective_source, effective_threshold_type)
    effective_direction = direction or str(patch_values["direction"])
    effective_connectivity = connectivity or str(patch_values["connectivity"])
    effective_minimum_cells = (
        minimum_patch_cells
        if minimum_patch_cells is not None
        else int(patch_values["minimum_patch_cells"])
    )
    effective_minimum_area = (
        minimum_patch_area_km2
        if minimum_patch_area_km2 is not None
        else float(patch_values["minimum_patch_area_km2"])
    )
    if effective_threshold_type == "standardized":
        scalar_threshold = (
            fixed_threshold
            if fixed_threshold is not None
            else float(patch_values["standardized_threshold"])
        )
    else:
        scalar_threshold = (
            fixed_threshold
            if fixed_threshold is not None
            else float(patch_values["fixed_threshold"])
        )
    detection_config = PatchDetectionConfig(
        source_variable=effective_source,
        direction=effective_direction,  # type: ignore[arg-type]
        threshold_type=effective_threshold_type,  # type: ignore[arg-type]
        comparison=str(patch_values["comparison"]),  # type: ignore[arg-type]
        connectivity=effective_connectivity,  # type: ignore[arg-type]
        minimum_patch_cells=effective_minimum_cells,
        minimum_patch_area_km2=effective_minimum_area,
        remove_boundary_only_artifacts=bool(patch_values["remove_boundary_only_artifacts"]),
    )
    compression_level = int(patch_values["label_compression_level"])
    if not 0 <= compression_level <= 9:
        raise ValueError("label_compression_level must be between 0 and 9")
    patches_path = resolve_project_path(patches_output or patch_values["patches_output"])
    daily_path = resolve_project_path(daily_summary_output or patch_values["daily_summary_output"])
    save_labels = bool(patch_values["save_label_cube"])
    labels_path = (
        resolve_project_path(labels_output or patch_values["labels_output"])
        if save_labels
        else None
    )
    json_path = resolve_project_path(json_summary_output or patch_values["json_summary_output"])
    destinations = [patches_path, daily_path, json_path]
    if labels_path is not None:
        destinations.append(labels_path)
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Daily patch output exists; use --overwrite: " + ", ".join(map(str, existing))
        )

    dataset, data_mode, source_path = _load_sst(config_values, sst_input)
    dates = available_dates(dataset)
    climate_values = config_values["climatology"]
    daily_climatology_path = resolve_project_path(
        climatology_input or climate_values["daily_path"]
    )
    selection: CachedClimatologySelection = open_cached_climatology(
        daily_climatology_path,
        resolve_project_path(climate_values["monthly_path"]),
        data_mode=data_mode,
        require_daily_threshold=effective_threshold_type == "daily_climatological",
        allow_monthly_fallback=bool(climate_values["allow_monthly_fallback"]),
        threshold_variable=str(patch_values["climatological_threshold_variable"]),
    )
    if fixed_threshold is not None and effective_threshold_type == "daily_climatological":
        selection.close()
        raise ValueError("--fixed-threshold cannot be used with daily_climatological patches")
    definition = threshold_definition(
        detection_config,
        threshold_name=str(patch_values["climatological_threshold_variable"]),
        scalar_threshold=scalar_threshold,
    )
    LOGGER.info(
        "Processing %d cached dates using %s and %s connectivity",
        len(dates), definition, detection_config.connectivity,
    )
    ocean_domain = dataset.sst.notnull().any("time").load()
    patch_ids = np.zeros(
        (len(dates), dataset.sizes["latitude"], dataset.sizes["longitude"]),
        dtype=np.int32,
    )
    threshold_masks = np.zeros_like(patch_ids, dtype=np.uint8)
    valid_masks = np.zeros_like(patch_ids, dtype=np.uint8)
    patch_frames: list[pd.DataFrame] = []
    daily_rows: list[dict[str, Any]] = []
    warnings = list(selection.warnings)
    try:
        for position, date in enumerate(dates):
            source_field, active_threshold = _match_date_fields(
                dataset,
                selection.dataset,
                position,
                source_variable=effective_source,
                threshold_type=effective_threshold_type,
                scalar_threshold=scalar_threshold,
                climatological_threshold_variable=str(
                    patch_values["climatological_threshold_variable"]
                ),
            )
            detection = identify_patches(
                source_field,
                active_threshold,
                config=detection_config,
                ocean_mask=ocean_domain,
            )
            patches, daily = characterize_daily_patches(
                detection,
                date=date,
                config=detection_config,
                climatology_method=selection.method,
                data_mode=data_mode,
                ocean_domain_mask=ocean_domain,
            )
            if not patches.empty:
                patch_frames.append(patches)
            daily_rows.append(daily)
            patch_ids[position] = detection.patch_id.values
            threshold_masks[position] = detection.threshold_mask.values.astype(np.uint8)
            valid_masks[position] = detection.valid_ocean_mask.values.astype(np.uint8)
            warnings.extend(detection.warnings)
            if daily.get("status") == "warning" and daily.get("reason"):
                warnings.append(str(daily["reason"]))
            if (position + 1) % 25 == 0 or position + 1 == len(dates):
                LOGGER.info("Processed %d/%d dates", position + 1, len(dates))
    finally:
        selection.close()
    patches = (
        pd.concat(patch_frames, ignore_index=True)
        if patch_frames
        else pd.DataFrame(columns=PATCH_COLUMNS)
    )
    patches = patches[PATCH_COLUMNS]
    daily_summary = daily_summary_frame(daily_rows)
    labels = _label_dataset(
        dataset,
        patch_ids,
        threshold_masks,
        valid_masks,
        definition=definition,
        config=detection_config,
        climatology_method=selection.method,
        data_mode=data_mode,
        minimum_patch_area_km2=effective_minimum_area,
    )
    largest_patch = None
    if not patches.empty:
        largest_row = patches.loc[patches.area_km2.idxmax()]
        largest_patch = {
            "date": largest_row.date,
            "patch_id": int(largest_row.patch_id),
            "area_km2": float(largest_row.area_km2),
        }
    elapsed = perf_counter() - started
    unique_warnings = list(dict.fromkeys(str(value) for value in warnings if value))
    summary: dict[str, Any] = {
        "title": "Humboldt Ocean Watch daily spatial thermal-patch build summary",
        "experimental_product": True,
        "official_enso_classification": "not provided",
        "spatiotemporal_tracking": "not performed",
        "input_file": str(source_path),
        "source_variable": effective_source,
        "threshold_type": effective_threshold_type,
        "threshold_definition": definition,
        "date_range": {"start": dates[0], "end": dates[-1]},
        "number_of_dates": len(dates),
        "number_of_dates_with_patches": int((daily_summary.patch_count > 0).sum()),
        "total_patches_detected": len(patches),
        "largest_patch_observed": largest_patch,
        "maximum_daily_patch_count": int(daily_summary.patch_count.max()),
        "connectivity": detection_config.connectivity,
        "filtering_parameters": {
            "minimum_patch_cells": detection_config.minimum_patch_cells,
            "minimum_patch_area_km2": detection_config.minimum_patch_area_km2,
            "remove_boundary_only_artifacts": detection_config.remove_boundary_only_artifacts,
        },
        "climatology_method": selection.method,
        "climatology_file": str(selection.path),
        "climatology_fallback_used": selection.fallback_used,
        "data_mode": data_mode,
        "output_paths": {
            "patches": str(patches_path),
            "daily_summary": str(daily_path),
            "labels": str(labels_path) if labels_path is not None else None,
            "json_summary": str(json_path),
        },
        "processing_time_seconds": elapsed,
        "overall_status": (
            "not_calculated" if not len(patches) else "warning" if unique_warnings else "valid"
        ),
        "warnings": unique_warnings,
    }
    output_sizes = write_outputs_atomically(
        patches,
        daily_summary,
        labels,
        summary,
        patches_path=patches_path,
        daily_summary_path=daily_path,
        labels_path=labels_path,
        json_summary_path=json_path,
        compression_level=compression_level,
    )
    summary["approximate_output_sizes_bytes"] = output_sizes
    LOGGER.info(
        "Completed daily patch build in %.2f seconds; output sizes %s",
        elapsed, output_sizes,
    )
    return PatchBuildArtifacts(
        patches_path=patches_path,
        daily_summary_path=daily_path,
        labels_path=labels_path,
        json_summary_path=json_path,
        patches=patches,
        daily_summary=daily_summary,
        labels=labels,
        summary=summary,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sst-input", type=Path)
    parser.add_argument("--climatology-input", type=Path)
    parser.add_argument("--source-variable")
    parser.add_argument(
        "--threshold-type", choices=["fixed", "standardized", "daily_climatological"]
    )
    parser.add_argument("--fixed-threshold", type=float)
    parser.add_argument("--direction", choices=["above", "below"])
    parser.add_argument("--connectivity", choices=["rook", "queen"])
    parser.add_argument("--minimum-patch-cells", type=int)
    parser.add_argument("--minimum-patch-area-km2", type=float)
    parser.add_argument("--patches-output", type=Path)
    parser.add_argument("--daily-summary-output", type=Path)
    parser.add_argument("--labels-output", type=Path)
    parser.add_argument("--json-summary-output", type=Path)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = build(
        config_path=args.config,
        sst_input=args.sst_input,
        climatology_input=args.climatology_input,
        source_variable=args.source_variable,
        threshold_type=args.threshold_type,
        fixed_threshold=args.fixed_threshold,
        direction=args.direction,
        connectivity=args.connectivity,
        minimum_patch_cells=args.minimum_patch_cells,
        minimum_patch_area_km2=args.minimum_patch_area_km2,
        patches_output=args.patches_output,
        daily_summary_output=args.daily_summary_output,
        labels_output=args.labels_output,
        json_summary_output=args.json_summary_output,
        overwrite=args.overwrite,
    )
    print(f"Patch table: {artifacts.patches_path} {artifacts.patches.shape}")
    print(f"Daily summary: {artifacts.daily_summary_path} {artifacts.daily_summary.shape}")
    print(f"Label cube: {artifacts.labels_path} {dict(artifacts.labels.sizes)}")
    print(f"Build summary: {artifacts.json_summary_path}")
    print(f"Threshold: {artifacts.summary['threshold_definition']}")
    print(f"Total patches: {artifacts.summary['total_patches_detected']}")
    print(f"Dates with patches: {artifacts.summary['number_of_dates_with_patches']}")
    print(f"Maximum daily patch count: {artifacts.summary['maximum_daily_patch_count']}")
    print(f"Processing time seconds: {artifacts.summary['processing_time_seconds']:.2f}")


if __name__ == "__main__":
    main()
