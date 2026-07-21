"""Build offline spatiotemporal patch lineages, tracks, and event families."""

from __future__ import annotations

import argparse
import json
import logging
import sys
from dataclasses import dataclass
from pathlib import Path
from time import perf_counter
from typing import Any
from uuid import uuid4

import numpy as np
import pandas as pd
import xarray as xr

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.export_utils import dumps_json_safe
from src.patch_linking import PatchTrackingConfig, link_date_pair, tracking_config
from src.patch_tracking import (
    EDGE_COLUMNS,
    NODE_EVENT_CODES,
    OBSERVATION_COLUMNS,
    build_lineage,
    validate_lineage_dag,
)
from src.track_metrics import (
    FAMILY_COLUMNS,
    TRACK_COLUMNS,
    calculate_family_metrics,
    calculate_track_metrics,
)
from src.track_validation import ValidatedTrackingInputs, load_and_validate_tracking_inputs
from src.utils import configure_logging, load_config, resolve_project_path

LOGGER = logging.getLogger("humboldt_ocean_watch.track_builder")


TRACK_MAPPING_COLUMNS = ["track_numeric_id", "track_id", "start_date", "end_date"]
FAMILY_MAPPING_COLUMNS = [
    "event_family_numeric_id", "event_family_id", "start_date", "end_date"
]


@dataclass(frozen=True)
class TrackingArtifacts:
    observations_path: Path
    edges_path: Path
    tracks_path: Path
    families_path: Path
    track_labels_path: Path
    track_mapping_path: Path
    family_mapping_path: Path
    json_summary_path: Path
    observations: pd.DataFrame
    edges: pd.DataFrame
    tracks: pd.DataFrame
    families: pd.DataFrame
    track_labels: xr.Dataset
    track_mapping: pd.DataFrame
    family_mapping: pd.DataFrame
    summary: dict[str, Any]


def _temporary_path(destination: Path) -> Path:
    return destination.with_name(
        f".{destination.stem}.{uuid4().hex}.tmp{destination.suffix}"
    )


def _tracking_paths(
    values: dict[str, Any],
    **overrides: Path | None,
) -> dict[str, Path]:
    names = {
        "observations": "observations_output",
        "edges": "edges_output",
        "tracks": "tracks_output",
        "families": "families_output",
        "track_labels": "track_labels_output",
        "track_mapping": "track_mapping_output",
        "family_mapping": "family_mapping_output",
        "json_summary": "json_summary_output",
    }
    return {
        name: resolve_project_path(overrides.get(name) or values[key])
        for name, key in names.items()
    }


def calculate_all_links(
    validated: ValidatedTrackingInputs,
    config: PatchTrackingConfig,
) -> pd.DataFrame:
    """Calculate accepted edges between adjacent available dates only."""
    if not config.preserve_all_candidate_edges:
        raise ValueError(
            "preserve_all_candidate_edges must be true so splits and merges are retained"
        )
    labels = validated.labels.patch_id
    latitude_name = validated.geometry.latitude_name
    longitude_name = validated.geometry.longitude_name
    labels = labels.transpose("time", latitude_name, longitude_name)
    areas = np.asarray(validated.geometry.area_km2.values, dtype=float)
    groups = {
        pd.Timestamp(date).normalize(): group.copy()
        for date, group in validated.patches.groupby("date", sort=False)
    }
    edge_frames: list[pd.DataFrame] = []
    for position in range(len(validated.dates) - 1):
        predecessor_date = validated.dates[position]
        successor_date = validated.dates[position + 1]
        elapsed = int((successor_date - predecessor_date).days)
        if elapsed > config.maximum_calendar_gap_days and not (
            config.allow_gap_bridging and elapsed <= config.maximum_bridge_gap_days
        ):
            LOGGER.warning(
                "Terminating active tracks across %d-day gap ending %s",
                elapsed,
                successor_date.date().isoformat(),
            )
            continue
        predecessor_patches = groups.get(predecessor_date, validated.patches.iloc[0:0])
        successor_patches = groups.get(successor_date, validated.patches.iloc[0:0])
        if predecessor_patches.empty or successor_patches.empty:
            continue
        pair_edges = link_date_pair(
            predecessor_patches,
            successor_patches,
            np.asarray(labels.isel(time=position).values),
            np.asarray(labels.isel(time=position + 1).values),
            areas,
            predecessor_date=predecessor_date,
            successor_date=successor_date,
            config=config,
        )
        if not pair_edges.empty:
            edge_frames.append(pair_edges)
    if not edge_frames:
        from src.patch_linking import PAIR_METRIC_COLUMNS

        return pd.DataFrame(columns=PAIR_METRIC_COLUMNS)
    return pd.concat(edge_frames, ignore_index=True)


def _mapping_tables(
    observations: pd.DataFrame,
    tracks: pd.DataFrame,
    families: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    track_dates = tracks.set_index("track_id")[["start_date", "end_date"]]
    track_ids = sorted(observations.track_id.unique())
    track_mapping = pd.DataFrame([
        {
            "track_numeric_id": number,
            "track_id": identifier,
            "start_date": track_dates.loc[identifier, "start_date"],
            "end_date": track_dates.loc[identifier, "end_date"],
        }
        for number, identifier in enumerate(track_ids, start=1)
    ], columns=TRACK_MAPPING_COLUMNS)
    family_dates = families.set_index("event_family_id")[["start_date", "end_date"]]
    family_ids = sorted(observations.event_family_id.unique())
    family_mapping = pd.DataFrame([
        {
            "event_family_numeric_id": number,
            "event_family_id": identifier,
            "start_date": family_dates.loc[identifier, "start_date"],
            "end_date": family_dates.loc[identifier, "end_date"],
        }
        for number, identifier in enumerate(family_ids, start=1)
    ], columns=FAMILY_MAPPING_COLUMNS)
    return track_mapping, family_mapping


def build_track_label_cube(
    validated: ValidatedTrackingInputs,
    observations: pd.DataFrame,
    track_mapping: pd.DataFrame,
    family_mapping: pd.DataFrame,
) -> xr.Dataset:
    """Map string track/family IDs to deterministic compressed integer fields."""
    latitude_name = validated.geometry.latitude_name
    longitude_name = validated.geometry.longitude_name
    local = validated.labels.patch_id.transpose(
        "time", latitude_name, longitude_name
    ).astype(np.int32)
    if latitude_name != "latitude" or longitude_name != "longitude":
        local = local.rename({latitude_name: "latitude", longitude_name: "longitude"})
    local_values = np.asarray(local.values, dtype=np.int32)
    track_values = np.zeros(local_values.shape, dtype=np.int32)
    family_values = np.zeros(local_values.shape, dtype=np.int32)
    event_values = np.zeros(local_values.shape, dtype=np.uint8)
    track_codes = dict(zip(track_mapping.track_id, track_mapping.track_numeric_id, strict=True))
    family_codes = dict(zip(
        family_mapping.event_family_id,
        family_mapping.event_family_numeric_id,
        strict=True,
    ))
    date_positions = {date: position for position, date in enumerate(validated.dates)}
    for row in observations.itertuples(index=False):
        position = date_positions[pd.Timestamp(row.date).normalize()]
        cells = local_values[position] == int(row.local_patch_id)
        if not cells.any():
            raise ValueError("Tracked observation has no corresponding local label cells")
        track_values[position][cells] = int(track_codes[row.track_id])
        family_values[position][cells] = int(family_codes[row.event_family_id])
        event_values[position][cells] = NODE_EVENT_CODES[str(row.node_event_type)]
    if not np.array_equal(local_values > 0, track_values > 0):
        raise ValueError("Every local patch cell must receive one persistent track label")
    coords = {name: local.coords[name] for name in ("time", "latitude", "longitude")}
    data_vars: dict[str, Any] = {
        "local_patch_id": (("time", "latitude", "longitude"), local_values),
        "track_numeric_id": (("time", "latitude", "longitude"), track_values),
        "event_family_numeric_id": (("time", "latitude", "longitude"), family_values),
        "node_event_code": (("time", "latitude", "longitude"), event_values),
    }
    if "valid_ocean_mask" in validated.labels:
        valid = validated.labels.valid_ocean_mask.transpose(
            "time", validated.geometry.latitude_name, validated.geometry.longitude_name
        )
        if validated.geometry.latitude_name != "latitude" or validated.geometry.longitude_name != "longitude":
            valid = valid.rename({
                validated.geometry.latitude_name: "latitude",
                validated.geometry.longitude_name: "longitude",
            })
        data_vars["valid_ocean_mask"] = (
            ("time", "latitude", "longitude"),
            np.asarray(valid.values, dtype=np.uint8),
        )
    dataset = xr.Dataset(
        data_vars,
        coords=coords,
        attrs={
            "title": "Humboldt Ocean Watch spatiotemporal thermal-patch track labels",
            "local_patch_id_scope": "date-local daily patch label; not persistent",
            "track_id_mapping": "outputs/analytics/track_id_mapping.parquet",
            "event_family_id_mapping": "outputs/analytics/event_family_id_mapping.parquet",
            "node_event_code_mapping": dumps_json_safe(NODE_EVENT_CODES, indent=None),
            "climatology_method": validated.labels.attrs.get("climatology_method", "unknown"),
            "data_mode": validated.labels.attrs.get("data_mode", "unknown"),
            "experimental_product": "true",
            "official_enso_classification": "not provided",
        },
    )
    dataset.local_patch_id.attrs.update({
        "long_name": "date-local daily patch identifier", "flag_values": "0 means no patch"
    })
    dataset.track_numeric_id.attrs.update({
        "long_name": "persistent non-branching track numeric identifier",
        "mapping_table": "track_id_mapping.parquet", "flag_values": "0 means no tracked patch",
    })
    dataset.event_family_numeric_id.attrs.update({
        "long_name": "persistent weak-lineage-component numeric identifier",
        "mapping_table": "event_family_id_mapping.parquet", "flag_values": "0 means no tracked patch",
    })
    dataset.node_event_code.attrs.update({
        "long_name": "lineage node-event classification code",
        "code_mapping": dumps_json_safe(NODE_EVENT_CODES, indent=None),
    })
    return dataset


def _validate_output_files(
    temporary: dict[str, Path],
    *,
    expected_observations: int,
    expected_dates: int,
) -> None:
    observations = pd.read_parquet(temporary["observations"])
    edges = pd.read_parquet(temporary["edges"])
    tracks = pd.read_parquet(temporary["tracks"])
    families = pd.read_parquet(temporary["families"])
    track_mapping = pd.read_parquet(temporary["track_mapping"])
    family_mapping = pd.read_parquet(temporary["family_mapping"])
    if observations.columns.tolist() != OBSERVATION_COLUMNS or len(observations) != expected_observations:
        raise ValueError("Temporary patch-observation output failed schema validation")
    if edges.columns.tolist() != EDGE_COLUMNS:
        raise ValueError("Temporary lineage-edge output failed schema validation")
    if tracks.columns.tolist() != TRACK_COLUMNS or families.columns.tolist() != FAMILY_COLUMNS:
        raise ValueError("Temporary track or family output failed schema validation")
    if track_mapping.columns.tolist() != TRACK_MAPPING_COLUMNS or family_mapping.columns.tolist() != FAMILY_MAPPING_COLUMNS:
        raise ValueError("Temporary ID mapping output failed schema validation")
    if observations.track_id.isna().any() or observations.event_family_id.isna().any():
        raise ValueError("Temporary observations contain unassigned IDs")
    validate_lineage_dag(observations.node_id, edges)
    with xr.open_dataset(temporary["track_labels"]) as labels:
        if dict(labels.sizes).get("time") != expected_dates:
            raise ValueError("Temporary persistent label cube has an invalid time dimension")
        required = {
            "local_patch_id", "track_numeric_id", "event_family_numeric_id",
            "node_event_code",
        }
        if not required.issubset(labels.data_vars):
            raise ValueError("Temporary persistent label cube is missing required variables")
        if not np.array_equal(labels.local_patch_id.values > 0, labels.track_numeric_id.values > 0):
            raise ValueError("Temporary persistent labels do not cover all patch cells")
    parsed = json.loads(temporary["json_summary"].read_text(encoding="utf-8"))
    if parsed.get("overall_status") not in {"valid", "warning", "not_calculated"}:
        raise ValueError("Temporary JSON tracking summary is invalid")


def write_outputs_atomically(
    artifacts: TrackingArtifacts,
    destinations: dict[str, Path],
    *,
    compression_level: int,
) -> dict[str, int]:
    """Validate all temporary products before replacing any destination."""
    for path in destinations.values():
        path.parent.mkdir(parents=True, exist_ok=True)
    temporary = {name: _temporary_path(path) for name, path in destinations.items()}
    try:
        artifacts.observations.to_parquet(temporary["observations"], index=False)
        artifacts.edges.to_parquet(temporary["edges"], index=False)
        artifacts.tracks.to_parquet(temporary["tracks"], index=False)
        artifacts.families.to_parquet(temporary["families"], index=False)
        artifacts.track_mapping.to_parquet(temporary["track_mapping"], index=False)
        artifacts.family_mapping.to_parquet(temporary["family_mapping"], index=False)
        spatial_shape = (
            artifacts.track_labels.sizes["latitude"],
            artifacts.track_labels.sizes["longitude"],
        )
        encoding = {
            name: {
                "dtype": "uint8" if name in {"node_event_code", "valid_ocean_mask"} else "int32",
                "zlib": True,
                "complevel": compression_level,
                "chunksizes": (1, *spatial_shape),
            }
            for name in artifacts.track_labels.data_vars
        }
        artifacts.track_labels.to_netcdf(
            temporary["track_labels"], engine="netcdf4", encoding=encoding
        )
        provisional_sizes = {
            name: path.stat().st_size
            for name, path in temporary.items()
            if name != "json_summary"
        }
        summary = {
            **artifacts.summary,
            "approximate_output_sizes_bytes": provisional_sizes,
        }
        temporary["json_summary"].write_text(
            dumps_json_safe(summary) + "\n", encoding="utf-8"
        )
        _validate_output_files(
            temporary,
            expected_observations=len(artifacts.observations),
            expected_dates=artifacts.track_labels.sizes["time"],
        )
        sizes = {name: path.stat().st_size for name, path in temporary.items()}
        for name, destination in destinations.items():
            temporary[name].replace(destination)
        return sizes
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def build(
    *,
    config_path: str | Path = PROJECT_ROOT / "config.yaml",
    daily_patches: str | Path | None = None,
    daily_labels: str | Path | None = None,
    daily_summary: str | Path | None = None,
    observations_output: str | Path | None = None,
    edges_output: str | Path | None = None,
    tracks_output: str | Path | None = None,
    families_output: str | Path | None = None,
    track_labels_output: str | Path | None = None,
    track_mapping_output: str | Path | None = None,
    family_mapping_output: str | Path | None = None,
    json_summary_output: str | Path | None = None,
    overwrite: bool = False,
) -> TrackingArtifacts:
    """Run the complete cached-only spatiotemporal tracking build."""
    started = perf_counter()
    config_values = load_config(config_path)
    configure_logging(str(config_values.get("logging", {}).get("level", "INFO")))
    if "patch_tracking" not in config_values:
        raise ValueError("Configuration is missing patch_tracking")
    tracking_values = config_values["patch_tracking"]
    link_config = tracking_config(tracking_values)
    compression_level = int(tracking_values["label_compression_level"])
    if not 0 <= compression_level <= 9:
        raise ValueError("label_compression_level must be between zero and nine")
    if not bool(tracking_values["save_track_label_cube"]):
        raise ValueError("save_track_label_cube must be true for Increment 4C outputs")
    patch_values = config_values["patch_detection"]
    input_paths = {
        "daily_patches": resolve_project_path(daily_patches or patch_values["patches_output"]),
        "daily_labels": resolve_project_path(daily_labels or patch_values["labels_output"]),
        "daily_summary": resolve_project_path(daily_summary or patch_values["daily_summary_output"]),
    }
    destinations = _tracking_paths(
        tracking_values,
        observations=Path(observations_output) if observations_output else None,
        edges=Path(edges_output) if edges_output else None,
        tracks=Path(tracks_output) if tracks_output else None,
        families=Path(families_output) if families_output else None,
        track_labels=Path(track_labels_output) if track_labels_output else None,
        track_mapping=Path(track_mapping_output) if track_mapping_output else None,
        family_mapping=Path(family_mapping_output) if family_mapping_output else None,
        json_summary=Path(json_summary_output) if json_summary_output else None,
    )
    existing = [path for path in destinations.values() if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Spatiotemporal tracking output exists; use --overwrite: "
            + ", ".join(str(path) for path in existing)
        )

    validated = load_and_validate_tracking_inputs(
        input_paths["daily_patches"], input_paths["daily_labels"], input_paths["daily_summary"]
    )
    LOGGER.info(
        "Tracking %d patch observations across %d cached dates",
        len(validated.patches), len(validated.dates),
    )
    pair_edges = calculate_all_links(validated, link_config)
    observations, edges = build_lineage(validated.patches, pair_edges)
    tracks = calculate_track_metrics(observations)
    families = calculate_family_metrics(observations)
    track_mapping, family_mapping = _mapping_tables(observations, tracks, families)
    label_cube = build_track_label_cube(
        validated, observations, track_mapping, family_mapping
    )
    elapsed = perf_counter() - started
    warnings = list(validated.warnings)
    warning_observations = int((observations.status == "warning").sum()) if not observations.empty else 0
    if warning_observations:
        warnings.append(
            f"{warning_observations} patch observation(s) inherit daily-patch geometry "
            "warnings; see observation status and reason"
        )
    warning_tracks = int((tracks.status == "warning").sum()) if not tracks.empty else 0
    if warning_tracks:
        warnings.append(
            f"{warning_tracks} track(s) contain undefined one-day or zero-displacement "
            "movement/area-change metrics; see track status and reason"
        )
    warning_families = int((families.status == "warning").sum()) if not families.empty else 0
    if warning_families:
        warnings.append(
            f"{warning_families} one-day event family/families have undefined movement "
            "rates; see family status and reason"
        )
    split_count = int((observations.node_event_type == "split_parent").sum())
    merge_count = int((observations.node_event_type == "merge_child").sum())
    complex_count = int((observations.node_event_type == "complex_branch").sum())
    isolated_count = int((observations.node_event_type == "isolated_single_day").sum())
    longest_track = None
    if not tracks.empty:
        row = tracks.sort_values(
            ["duration_calendar_days", "track_id"], ascending=[False, True]
        ).iloc[0]
        longest_track = {
            "track_id": row.track_id,
            "duration_calendar_days": int(row.duration_calendar_days),
            "observed_days": int(row.observed_days),
        }
    longest_family = None
    if not families.empty:
        row = families.sort_values(
            ["duration_calendar_days", "event_family_id"], ascending=[False, True]
        ).iloc[0]
        longest_family = {
            "event_family_id": row.event_family_id,
            "duration_calendar_days": int(row.duration_calendar_days),
            "observed_patch_days": int(row.observed_patch_days),
        }
    summary: dict[str, Any] = {
        "title": "Humboldt Ocean Watch spatiotemporal thermal-patch tracking summary",
        "experimental_product": True,
        "official_enso_classification": "not provided",
        "input_paths": {name: str(path) for name, path in input_paths.items()},
        "date_range": {
            "start": validated.dates[0] if len(validated.dates) else None,
            "end": validated.dates[-1] if len(validated.dates) else None,
        },
        "number_of_dates": len(validated.dates),
        "total_patch_observations": len(observations),
        "accepted_lineage_edges": len(edges),
        "continuation_edges": int((edges.lineage_relation == "continuation").sum()),
        "split_events": split_count,
        "merge_events": merge_count,
        "complex_events": complex_count,
        "isolated_patches": isolated_count,
        "total_tracks": len(tracks),
        "total_event_families": len(families),
        "longest_track": longest_track,
        "longest_event_family": longest_family,
        "largest_track_area_km2": float(tracks.maximum_area_km2.max()) if not tracks.empty else None,
        "maximum_family_area_km2": float(families.maximum_total_daily_area_km2.max()) if not families.empty else None,
        "maximum_speed_km_per_day": float(tracks.maximum_speed_km_per_day.max()) if not tracks.empty and tracks.maximum_speed_km_per_day.notna().any() else None,
        "configuration": tracking_values,
        "output_paths": {name: str(path) for name, path in destinations.items()},
        "overall_status": "not_calculated" if observations.empty else "warning" if warnings else "valid",
        "warnings": list(dict.fromkeys(warnings)),
        "processing_time_seconds": elapsed,
    }
    artifacts = TrackingArtifacts(
        observations_path=destinations["observations"],
        edges_path=destinations["edges"],
        tracks_path=destinations["tracks"],
        families_path=destinations["families"],
        track_labels_path=destinations["track_labels"],
        track_mapping_path=destinations["track_mapping"],
        family_mapping_path=destinations["family_mapping"],
        json_summary_path=destinations["json_summary"],
        observations=observations,
        edges=edges,
        tracks=tracks,
        families=families,
        track_labels=label_cube,
        track_mapping=track_mapping,
        family_mapping=family_mapping,
        summary=summary,
    )
    sizes = write_outputs_atomically(
        artifacts, destinations, compression_level=compression_level
    )
    summary["approximate_output_sizes_bytes"] = sizes
    LOGGER.info(
        "Completed tracking in %.2f seconds: %d edges, %d tracks, %d families",
        elapsed, len(edges), len(tracks), len(families),
    )
    return artifacts


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--daily-patches", type=Path)
    parser.add_argument("--daily-labels", type=Path)
    parser.add_argument("--daily-summary", type=Path)
    parser.add_argument("--observations-output", type=Path)
    parser.add_argument("--edges-output", type=Path)
    parser.add_argument("--tracks-output", type=Path)
    parser.add_argument("--families-output", type=Path)
    parser.add_argument("--track-labels-output", type=Path)
    parser.add_argument("--track-mapping-output", type=Path)
    parser.add_argument("--family-mapping-output", type=Path)
    parser.add_argument("--json-summary-output", type=Path)
    parser.add_argument("--config", type=Path, default=PROJECT_ROOT / "config.yaml")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    arguments = parse_args()
    artifacts = build(
        config_path=arguments.config,
        daily_patches=arguments.daily_patches,
        daily_labels=arguments.daily_labels,
        daily_summary=arguments.daily_summary,
        observations_output=arguments.observations_output,
        edges_output=arguments.edges_output,
        tracks_output=arguments.tracks_output,
        families_output=arguments.families_output,
        track_labels_output=arguments.track_labels_output,
        track_mapping_output=arguments.track_mapping_output,
        family_mapping_output=arguments.family_mapping_output,
        json_summary_output=arguments.json_summary_output,
        overwrite=arguments.overwrite,
    )
    print(f"Patch observations: {artifacts.observations_path} {artifacts.observations.shape}")
    print(f"Lineage edges: {artifacts.edges_path} {artifacts.edges.shape}")
    print(f"Tracks: {artifacts.tracks_path} {artifacts.tracks.shape}")
    print(f"Event families: {artifacts.families_path} {artifacts.families.shape}")
    print(f"Persistent labels: {artifacts.track_labels_path} {dict(artifacts.track_labels.sizes)}")
    print(f"Accepted links: {artifacts.summary['accepted_lineage_edges']}")
    print(f"Splits / merges / complex: {artifacts.summary['split_events']} / {artifacts.summary['merge_events']} / {artifacts.summary['complex_events']}")
    print(f"Tracks / families: {artifacts.summary['total_tracks']} / {artifacts.summary['total_event_families']}")
    print(f"Processing time seconds: {artifacts.summary['processing_time_seconds']:.2f}")


if __name__ == "__main__":
    main()
