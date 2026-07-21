"""Cached, failure-tolerant loaders for Increment 4 analytical products.

The dashboard treats every Increment 4 output as optional presentation data.
Structural problems are reported through :class:`ProductState` instead of
bringing down the complete Streamlit application.  NetCDF slices are loaded
eagerly inside a context manager so no file handle remains open in the cache.
"""

from __future__ import annotations

from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Generic, TypeVar

import pandas as pd
import streamlit as st
import xarray as xr


T = TypeVar("T")


@dataclass(frozen=True)
class ProductState(Generic[T]):
    """Availability and validation state for one cached dashboard product."""

    value: T | None
    path: Path
    status: str
    reason: str | None = None

    @property
    def available(self) -> bool:
        return self.status in {"available", "empty"} and self.value is not None

    @property
    def empty(self) -> bool:
        return self.status == "empty"


EVENT_COLUMNS = {
    "event_id", "source_variable", "threshold_type", "direction", "start_date",
    "end_date", "peak_date", "duration_calendar_days", "mean_intensity",
    "maximum_intensity", "cumulative_intensity", "severity_class", "status",
}
FLAG_COLUMNS = {
    "date", "source_variable", "value", "threshold", "exceeds_threshold",
    "event_id", "event_day", "intensity", "valid_coverage", "status",
}
PATCH_COLUMNS = {
    "date", "patch_id", "source_variable", "threshold_type", "direction",
    "area_km2", "centroid_latitude", "centroid_longitude", "mean_source_value",
    "maximum_source_value", "mean_exceedance", "maximum_exceedance", "status",
}
PATCH_SUMMARY_COLUMNS = {
    "date", "patch_count", "total_patch_area_km2", "largest_patch_area_km2",
    "threshold_area_fraction", "valid_coverage", "status",
}
OBSERVATION_COLUMNS = {
    "node_id", "date", "local_patch_id", "track_id", "event_family_id",
    "node_event_type", "area_km2", "centroid_latitude", "centroid_longitude",
    "mean_exceedance", "maximum_exceedance", "predecessor_count",
    "successor_count", "status",
}
EDGE_COLUMNS = {
    "predecessor_node_id", "successor_node_id", "predecessor_date",
    "successor_date", "elapsed_days", "area_weighted_iou", "link_score",
    "link_basis", "lineage_relation", "is_continuation_backbone", "status",
}
TRACK_COLUMNS = {
    "track_id", "event_family_id", "start_date", "end_date", "observed_days",
    "duration_calendar_days", "maximum_area_km2", "cumulative_severity",
    "trajectory_length_km", "mean_speed_km_per_day", "maximum_speed_km_per_day",
    "status",
}
FAMILY_COLUMNS = {
    "event_family_id", "start_date", "end_date", "duration_calendar_days",
    "track_count", "unique_patch_observation_count", "split_count", "merge_count",
    "complex_branch_count", "maximum_total_daily_area_km2",
    "family_area_days_km2_days", "cumulative_severity", "status",
}
TRACK_MAPPING_COLUMNS = {"track_numeric_id", "track_id"}
FAMILY_MAPPING_COLUMNS = {"event_family_numeric_id", "event_family_id"}


TABLE_SPECS: dict[str, tuple[str, set[str], tuple[str, ...]]] = {
    "univariate_events": ("event_detection.events_output", EVENT_COLUMNS, ("start_date", "end_date", "peak_date")),
    "univariate_flags": ("event_detection.daily_flags_output", FLAG_COLUMNS, ("date",)),
    "daily_patches": ("patch_detection.patches_output", PATCH_COLUMNS, ("date",)),
    "daily_patch_summary": ("patch_detection.daily_summary_output", PATCH_SUMMARY_COLUMNS, ("date",)),
    "patch_observations": ("patch_tracking.observations_output", OBSERVATION_COLUMNS, ("date",)),
    "lineage_edges": ("patch_tracking.edges_output", EDGE_COLUMNS, ("predecessor_date", "successor_date")),
    "tracks": ("patch_tracking.tracks_output", TRACK_COLUMNS, ("start_date", "end_date")),
    "event_families": ("patch_tracking.families_output", FAMILY_COLUMNS, ("start_date", "end_date")),
    "track_mapping": ("patch_tracking.track_mapping_output", TRACK_MAPPING_COLUMNS, ("start_date", "end_date")),
    "family_mapping": ("patch_tracking.family_mapping_output", FAMILY_MAPPING_COLUMNS, ("start_date", "end_date")),
}


@dataclass(frozen=True)
class EventProducts:
    """All tabular and JSON products used by the Thermal events tab."""

    univariate_events: ProductState[pd.DataFrame]
    univariate_flags: ProductState[pd.DataFrame]
    daily_patches: ProductState[pd.DataFrame]
    daily_patch_summary: ProductState[pd.DataFrame]
    patch_observations: ProductState[pd.DataFrame]
    lineage_edges: ProductState[pd.DataFrame]
    tracks: ProductState[pd.DataFrame]
    event_families: ProductState[pd.DataFrame]
    track_mapping: ProductState[pd.DataFrame]
    family_mapping: ProductState[pd.DataFrame]
    tracking_summary: ProductState[dict[str, Any]]

    def table(self, name: str) -> pd.DataFrame:
        state = getattr(self, name)
        return state.value.copy() if state.value is not None else pd.DataFrame()


def _file_signature(path: Path) -> tuple[str, int, int]:
    resolved = path.resolve()
    stat = resolved.stat()
    return str(resolved), stat.st_mtime_ns, stat.st_size


@st.cache_data(show_spinner=False, max_entries=24)
def _cached_parquet(
    path: str,
    modified_ns: int,
    size_bytes: int,
) -> pd.DataFrame:
    del modified_ns, size_bytes
    return pd.read_parquet(path)


@st.cache_data(show_spinner=False, max_entries=8)
def _cached_json(path: str, modified_ns: int, size_bytes: int) -> dict[str, Any]:
    del modified_ns, size_bytes
    return json.loads(Path(path).read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False, max_entries=32)
def _cached_netcdf_date(
    path: str,
    modified_ns: int,
    size_bytes: int,
    selected_date: str,
    required_variables: tuple[str, ...],
) -> xr.Dataset:
    """Read one exact date and close the source file before returning."""
    del modified_ns, size_bytes
    with xr.open_dataset(path) as source:
        missing = set(required_variables) - set(source.data_vars)
        if missing:
            raise ValueError(f"NetCDF is missing variables: {sorted(missing)}")
        if "time" not in source.coords:
            raise ValueError("NetCDF is missing the required time coordinate")
        times = pd.DatetimeIndex(pd.to_datetime(source.time.values, errors="coerce")).normalize()
        target = pd.Timestamp(selected_date).normalize()
        matches = [index for index, value in enumerate(times) if value == target]
        if not matches:
            raise KeyError(f"Date {target.date().isoformat()} is unavailable in {path}")
        return source[list(required_variables)].isel(time=matches[0]).load()


def load_table_product(
    path: str | Path,
    *,
    required_columns: set[str],
    date_columns: tuple[str, ...] = (),
) -> ProductState[pd.DataFrame]:
    """Load and validate a table without propagating expected product errors."""
    product_path = Path(path)
    if not product_path.exists():
        return ProductState(None, product_path, "unavailable", "Cached product does not exist")
    try:
        frame = _cached_parquet(*_file_signature(product_path))
    except Exception as exc:
        return ProductState(None, product_path, "invalid", f"Could not read Parquet product: {exc}")
    missing = required_columns - set(frame.columns)
    if missing:
        return ProductState(None, product_path, "invalid", f"Missing required columns: {sorted(missing)}")
    frame = frame.copy()
    for column in date_columns:
        if column in frame.columns:
            frame[column] = pd.to_datetime(frame[column], errors="coerce")
            if len(frame) and frame[column].isna().any():
                return ProductState(None, product_path, "invalid", f"Column {column!r} contains unreadable dates")
    return ProductState(frame, product_path, "empty" if frame.empty else "available")


def load_json_product(path: str | Path) -> ProductState[dict[str, Any]]:
    product_path = Path(path)
    if not product_path.exists():
        return ProductState(None, product_path, "unavailable", "Cached product does not exist")
    try:
        payload = _cached_json(*_file_signature(product_path))
    except Exception as exc:
        return ProductState(None, product_path, "invalid", f"Could not read JSON product: {exc}")
    if not isinstance(payload, dict):
        return ProductState(None, product_path, "invalid", "JSON product must contain an object")
    return ProductState(payload, product_path, "empty" if not payload else "available")


def load_netcdf_date(
    path: str | Path,
    selected_date: str | pd.Timestamp,
    *,
    required_variables: tuple[str, ...],
) -> ProductState[xr.Dataset]:
    product_path = Path(path)
    if not product_path.exists():
        return ProductState(None, product_path, "unavailable", "Cached product does not exist")
    try:
        dataset = _cached_netcdf_date(
            *_file_signature(product_path),
            pd.Timestamp(selected_date).date().isoformat(),
            tuple(required_variables),
        )
    except KeyError as exc:
        return ProductState(None, product_path, "unavailable", str(exc).strip("'"))
    except Exception as exc:
        return ProductState(None, product_path, "invalid", f"Could not read NetCDF product: {exc}")
    return ProductState(dataset, product_path, "available")


def _configured_path(config: dict[str, Any], dotted_key: str, root: Path) -> Path:
    section, key = dotted_key.split(".", 1)
    return root / Path(config[section][key])


def load_event_products(config: dict[str, Any], *, root: str | Path = ".") -> EventProducts:
    """Load every small cached product; unavailable products remain isolated."""
    project_root = Path(root)
    states: dict[str, ProductState[pd.DataFrame]] = {}
    for name, (dotted_key, columns, date_columns) in TABLE_SPECS.items():
        states[name] = load_table_product(
            _configured_path(config, dotted_key, project_root),
            required_columns=columns,
            date_columns=date_columns,
        )
    tracking_summary = load_json_product(
        _configured_path(config, "patch_tracking.json_summary_output", project_root)
    )
    return EventProducts(**states, tracking_summary=tracking_summary)


def product_messages(products: EventProducts) -> list[tuple[str, str, str]]:
    """Return label/status/reason triples for unavailable or invalid products."""
    messages: list[tuple[str, str, str]] = []
    for name in EventProducts.__dataclass_fields__:
        state = getattr(products, name)
        if state.status not in {"available", "empty"}:
            messages.append((name.replace("_", " "), state.status, state.reason or "Unknown error"))
    return messages
