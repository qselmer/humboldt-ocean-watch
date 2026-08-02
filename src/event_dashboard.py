"""Pure view models for the Increment 4D Streamlit interface.

This module only filters and summarizes cached Increment 4 products.  It does
not rerun event detection, daily patch identification, or patch tracking.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
import xarray as xr

from src.event_charts import (
    lineage_chart,
    plot_track_trajectory,
    temporal_metric_chart,
    univariate_event_timeline,
)
from src.event_data_loader import EventProducts, load_netcdf_date, product_messages
from src.event_maps import plot_daily_patch_map, plot_family_map, plot_track_map
from src.event_tables import (
    EDGE_DISPLAY_COLUMNS,
    EVENT_DISPLAY_COLUMNS,
    OBSERVATION_DISPLAY_COLUMNS,
    TRACK_DISPLAY_COLUMNS,
    filter_catalogue,
    prepare_family_tables,
    prepare_patch_table,
    rows_for_date,
    selected_record,
    stable_table,
)
from src.event_ui_utils import dataframe_to_iso_csv, format_fraction, format_scalar, record_to_json
from src.help_content import event_concept_hierarchy, filter_description, metric_tooltip
from src.i18n import format_identifier, human_label, internal_id_caption, tr
from src.interpretation_text import interpret_thermal_event
from src.patch_linking import haversine_distance_km


@dataclass(frozen=True)
class EventOverview:
    active_event: bool
    event_id: str | None
    event_day: int | None
    event_duration_days: int | None
    current_intensity: float | None
    cumulative_event_intensity: float | None
    patch_count: int
    total_patch_area_km2: float
    largest_patch_area_km2: float
    active_track_count: int
    active_family_count: int
    representativeness_class: str | None
    valid_coverage: float | None


def _native_float(value: Any) -> float | None:
    numeric = pd.to_numeric(pd.Series([value]), errors="coerce").iloc[0]
    return float(numeric) if pd.notna(numeric) and np.isfinite(float(numeric)) else None


def _native_int(value: Any) -> int | None:
    numeric = _native_float(value)
    return int(numeric) if numeric is not None else None


def prepare_event_overview(
    products: EventProducts,
    selected_date: str | pd.Timestamp,
    representativeness: pd.DataFrame | None = None,
) -> EventOverview:
    """Build current-state metrics with explicit zero/Not-active states."""
    flags = rows_for_date(products.table("univariate_flags"), selected_date)
    active_flag = flags.loc[flags.get("event_id", pd.Series(index=flags.index, dtype=object)).notna()]
    event_id = str(active_flag.iloc[0].event_id) if not active_flag.empty else None
    events = products.table("univariate_events")
    event_row = (
        events.loc[events.event_id.astype(str) == event_id]
        if event_id and not events.empty and "event_id" in events
        else events.iloc[0:0]
    )
    patch_summary = rows_for_date(products.table("daily_patch_summary"), selected_date)
    patch_row = patch_summary.iloc[0] if not patch_summary.empty else None
    observations = rows_for_date(products.table("patch_observations"), selected_date)
    representative = rows_for_date(representativeness, selected_date) if representativeness is not None else pd.DataFrame()
    representative_row = representative.iloc[0] if not representative.empty else None
    flag_row = active_flag.iloc[0] if not active_flag.empty else None
    return EventOverview(
        active_event=event_id is not None,
        event_id=event_id,
        event_day=_native_int(flag_row.get("event_day")) if flag_row is not None else None,
        event_duration_days=_native_int(event_row.iloc[0].get("duration_calendar_days")) if not event_row.empty else None,
        current_intensity=_native_float(flag_row.get("intensity")) if flag_row is not None else None,
        cumulative_event_intensity=_native_float(event_row.iloc[0].get("cumulative_intensity")) if not event_row.empty else None,
        patch_count=_native_int(patch_row.get("patch_count")) if patch_row is not None else 0,
        total_patch_area_km2=_native_float(patch_row.get("total_patch_area_km2")) if patch_row is not None else 0.0,
        largest_patch_area_km2=_native_float(patch_row.get("largest_patch_area_km2")) if patch_row is not None else 0.0,
        active_track_count=int(observations.track_id.dropna().nunique()) if "track_id" in observations else 0,
        active_family_count=int(observations.event_family_id.dropna().nunique()) if "event_family_id" in observations else 0,
        representativeness_class=(str(representative_row.get("class")) if representative_row is not None and pd.notna(representative_row.get("class")) else None),
        valid_coverage=(
            _native_float(patch_row.get("valid_coverage")) if patch_row is not None
            else (_native_float(representative_row.get("valid_coverage")) if representative_row is not None else None)
        ),
    )


def consistent_track_family_selection(
    tracks: pd.DataFrame,
    *,
    requested_track: str | None,
    requested_family: str | None,
) -> tuple[str | None, str | None, list[str]]:
    """Resolve valid IDs while keeping the family selector authoritative."""
    if tracks.empty:
        return None, None, []
    frame = tracks.dropna(subset=["track_id", "event_family_id"]).copy()
    frame["track_id"] = frame.track_id.astype(str)
    frame["event_family_id"] = frame.event_family_id.astype(str)
    families = set(frame.event_family_id)
    family = requested_family if requested_family in families else None
    candidates = frame.loc[frame.event_family_id == family] if family else frame
    options = sorted(candidates.track_id.unique().tolist())
    track = requested_track if requested_track in options else None
    if track:
        family = str(frame.loc[frame.track_id == track, "event_family_id"].iloc[0])
    return track, family, options


def prepare_track_daily_series(
    observations: pd.DataFrame,
    edges: pd.DataFrame,
    track_id: str,
) -> pd.DataFrame:
    """Prepare display-only daily track values from cached observations/edges."""
    track = observations.loc[observations.track_id.astype(str) == str(track_id)].copy()
    if track.empty:
        return track
    track["date"] = pd.to_datetime(track.date, errors="coerce")
    track = track.dropna(subset=["date"]).sort_values("date", kind="mergesort").reset_index(drop=True)
    track["speed_km_per_day"] = np.nan
    track["area_change_km2_per_day"] = np.nan
    for index in range(1, len(track)):
        elapsed = (track.loc[index, "date"] - track.loc[index - 1, "date"]).days
        if elapsed <= 0:
            continue
        track.loc[index, "area_change_km2_per_day"] = (
            float(track.loc[index, "area_km2"]) - float(track.loc[index - 1, "area_km2"])
        ) / elapsed
        predecessor = str(track.loc[index - 1, "node_id"])
        successor = str(track.loc[index, "node_id"])
        matching = edges.loc[
            (edges.predecessor_node_id.astype(str) == predecessor)
            & (edges.successor_node_id.astype(str) == successor)
        ] if not edges.empty else pd.DataFrame()
        if not matching.empty and "centroid_distance_km" in matching:
            distance = _native_float(matching.iloc[0].centroid_distance_km)
        else:
            distance = haversine_distance_km(
                float(track.loc[index - 1, "centroid_latitude"]),
                float(track.loc[index - 1, "centroid_longitude"]),
                float(track.loc[index, "centroid_latitude"]),
                float(track.loc[index, "centroid_longitude"]),
            )
        track.loc[index, "speed_km_per_day"] = distance / elapsed if distance is not None else np.nan
    return track


def prepare_family_daily_series(observations: pd.DataFrame, family_id: str) -> pd.DataFrame:
    """Aggregate cached family observations by date for dashboard display."""
    family = observations.loc[observations.event_family_id.astype(str) == str(family_id)].copy()
    if family.empty:
        return pd.DataFrame(columns=[
            "date", "total_family_area_km2", "patch_count", "centroid_latitude",
            "centroid_longitude", "mean_exceedance", "maximum_exceedance",
        ])
    family["date"] = pd.to_datetime(family.date, errors="coerce")
    rows: list[dict[str, Any]] = []
    for date, group in family.dropna(subset=["date"]).groupby("date", sort=True):
        areas = pd.to_numeric(group.area_km2, errors="coerce").to_numpy(dtype=float)
        valid = np.isfinite(areas) & (areas > 0)
        total = float(areas[valid].sum())
        if total > 0:
            latitude = float(np.average(pd.to_numeric(group.centroid_latitude, errors="coerce")[valid], weights=areas[valid]))
            longitude = float(np.average(pd.to_numeric(group.centroid_longitude, errors="coerce")[valid], weights=areas[valid]))
            exceedance = pd.to_numeric(group.mean_exceedance, errors="coerce").to_numpy(dtype=float)
            mean_exceedance = float(np.average(exceedance[valid], weights=areas[valid]))
        else:
            latitude = longitude = mean_exceedance = np.nan
        rows.append({
            "date": pd.Timestamp(date),
            "total_family_area_km2": total,
            "patch_count": int(len(group)),
            "centroid_latitude": latitude,
            "centroid_longitude": longitude,
            "mean_exceedance": mean_exceedance,
            "maximum_exceedance": pd.to_numeric(group.maximum_exceedance, errors="coerce").max(),
        })
    return pd.DataFrame(rows)


def active_ids(observations: pd.DataFrame, selected_date: str | pd.Timestamp) -> tuple[list[str], list[str]]:
    current = rows_for_date(observations, selected_date)
    tracks = sorted(current.get("track_id", pd.Series(dtype=object)).dropna().astype(str).unique())
    families = sorted(current.get("event_family_id", pd.Series(dtype=object)).dropna().astype(str).unique())
    return tracks, families


def _selector(
    label: str,
    options: list[str],
    *,
    key: str,
    all_label: str = "All",
    help_text: str | None = None,
    format_func: Any | None = None,
) -> str | None:
    widget_options = [all_label, *options]
    if key in st.session_state and st.session_state[key] not in widget_options:
        del st.session_state[key]
    selected = st.selectbox(
        label, widget_options, key=key, help=help_text,
        format_func=format_func or str,
    )
    return None if selected == all_label else str(selected)


def _reset_event_filters() -> None:
    for key in list(st.session_state):
        if key.startswith("event_dashboard_"):
            del st.session_state[key]


def _product_status_messages(products: EventProducts) -> None:
    for label, status, reason in product_messages(products):
        message = f"{label.capitalize()} is {status}: {reason}"
        if status == "invalid":
            st.error(message)
        else:
            st.warning(message)


def _overview_panel(
    products: EventProducts,
    event_date: pd.Timestamp,
    representativeness: pd.DataFrame,
    *,
    source_variable: str | None,
    threshold_type: str | None,
    direction: str | None,
    event_status: str | None,
    maximum_rows: int,
    language: str,
) -> None:
    overview = prepare_event_overview(products, event_date, representativeness)
    with st.container(horizontal=True):
        st.metric(tr("active_univariate_event", language), tr("active", language) if overview.active_event else tr("not_active", language), border=True)
        st.metric(tr("current_event", language), format_identifier(overview.event_id, language) if overview.event_id else tr("not_active", language), border=True)
        st.metric(tr("event_day", language), overview.event_day if overview.event_day is not None else "—", border=True)
        st.metric(tr("daily_patch_count", language), overview.patch_count, border=True)
        st.metric(tr("largest_patch", language), format_scalar(overview.largest_patch_area_km2, "km²"), border=True)
        st.metric(tr("active_tracks", language), overview.active_track_count, help=metric_tooltip("track", language), border=True)
        st.metric(tr("active_families", language), overview.active_family_count, help=metric_tooltip("event_family", language), border=True)

    st.info(interpret_thermal_event(event_date, overview, language))
    with st.expander("Evidence and supporting diagnostics", expanded=False):
        with st.container(horizontal=True):
            st.metric(tr("event_duration", language), format_scalar(overview.event_duration_days, tr("days", language), precision=0), border=True)
            st.metric(tr("current_intensity", language), format_scalar(overview.current_intensity), border=True)
            st.metric(tr("cumulative_intensity", language), format_scalar(overview.cumulative_event_intensity), border=True)
            st.metric(tr("total_patch_area", language), format_scalar(overview.total_patch_area_km2, "km²"), border=True)
            st.metric(tr("representativeness", language), human_label(overview.representativeness_class, language), border=True)
            st.metric(tr("valid_coverage", language), format_fraction(overview.valid_coverage), help=metric_tooltip("valid_coverage", language), border=True)
        if overview.event_id:
            st.caption(internal_id_caption(overview.event_id, language))
    with st.expander("Interpretation and limitations", expanded=False):
        for concept in event_concept_hierarchy(language):
            st.write(concept)

    flags = products.table("univariate_flags")
    events = products.table("univariate_events")
    for column, value in (("source_variable", source_variable), ("threshold_type", threshold_type), ("direction", direction), ("status", event_status)):
        if value and column in flags:
            flags = flags.loc[flags[column].astype(str) == value]
        if value and column in events:
            events = events.loc[events[column].astype(str) == value]
    chart = univariate_event_timeline(flags, events, event_date)
    if chart is None:
        st.info("No compatible univariate timeline is available for the selected filters.")
    else:
        st.altair_chart(chart, width="stretch")
    st.markdown("#### Detected univariate events")
    event_options = sorted(events.event_id.dropna().astype(str).unique()) if "event_id" in events else []
    selected_event = _selector(
        tr("selected_event", language), event_options, key="event_dashboard_event_id",
        all_label=tr("all", language),
        format_func=lambda value: tr("all", language) if value == tr("all", language) else format_identifier(value, language),
    )
    if events.empty:
        st.info("No univariate events match the selected filters.")
    else:
        with st.expander("Detailed event table and technical identifiers", expanded=False):
            event_display = stable_table(events, EVENT_DISPLAY_COLUMNS, maximum_rows)
            table_state = st.dataframe(
                event_display, hide_index=True, width="stretch", on_select="rerun",
                selection_mode="single-row", key="event_dashboard_event_table",
            )
            selected_rows = table_state.selection.rows
            if selected_rows and "event_id" in event_display:
                selected_event = str(event_display.iloc[selected_rows[0]].event_id)
            if selected_event:
                event_record = selected_record(events, "event_id", selected_event)
                if event_record:
                    st.json(event_record)

    with st.expander("Complete event catalogues", expanded=False):
        catalogue_tabs = st.tabs(["Univariate events", "Tracks", "Event families"])
        all_frames = [events, products.table("tracks"), products.table("event_families")]
        for tab, name, frame in zip(catalogue_tabs, ("events", "tracks", "families"), all_frames, strict=True):
            with tab:
                search = st.text_input("Search", key=f"event_dashboard_search_{name}", placeholder="ID, status, or metadata")
                dates = []
                for column in ("start_date", "end_date"):
                    if column in frame:
                        dates.extend(pd.to_datetime(frame[column], errors="coerce").dropna().tolist())
                if dates:
                    date_range = st.date_input(
                        "Catalogue date range",
                        value=(min(dates).date(), max(dates).date()),
                        min_value=min(dates).date(), max_value=max(dates).date(),
                        key=f"event_dashboard_dates_{name}",
                    )
                    if isinstance(date_range, tuple) and len(date_range) == 2:
                        start, end = date_range
                    else:
                        start = end = None
                else:
                    start = end = None
                filters = st.columns(4)
                minimum_duration = filters[0].number_input("Minimum duration (days)", min_value=0.0, value=0.0, key=f"event_dashboard_duration_{name}")
                minimum_area = filters[1].number_input("Minimum area (km²)", min_value=0.0, value=0.0, key=f"event_dashboard_area_{name}")
                minimum_severity = filters[2].number_input("Minimum severity", min_value=0.0, value=0.0, key=f"event_dashboard_severity_{name}")
                boundary = filters[3].checkbox("Boundary contact", key=f"event_dashboard_boundary_{name}")
                relation_columns = st.columns(2)
                require_split = relation_columns[0].checkbox("Contains split", key=f"event_dashboard_split_{name}")
                require_merge = relation_columns[1].checkbox("Contains merge", key=f"event_dashboard_merge_{name}")
                filtered = filter_catalogue(
                    frame, start_date=start, end_date=end, minimum_duration=minimum_duration,
                    minimum_area=minimum_area, minimum_severity=minimum_severity,
                    require_split=require_split, require_merge=require_merge,
                    require_boundary_contact=boundary, search=search,
                )
                st.caption(f"{len(filtered):,} rows after filtering")
                st.dataframe(filtered.head(maximum_rows), hide_index=True, width="stretch")
                st.download_button(
                    f"Download filtered {name} CSV", dataframe_to_iso_csv(filtered),
                    file_name=f"filtered_{name}.csv", mime="text/csv", key=f"event_dashboard_filtered_download_{name}",
                )


def _patch_panel(
    products: EventProducts,
    config: dict[str, Any],
    root: Path,
    event_date: pd.Timestamp,
    *,
    anomaly: xr.DataArray | None,
    anomaly_date: pd.Timestamp,
    maximum_rows: int,
    language: str,
) -> None:
    patches = rows_for_date(products.table("daily_patches"), event_date)
    dashboard_config = config["event_dashboard"]
    with st.expander("Map options", expanded=False):
        controls = st.columns(4)
        show_background = controls[0].checkbox("Show anomaly background", value=dashboard_config["show_anomaly_background_default"], key="event_dashboard_patch_background")
        show_labels = controls[1].checkbox("Show patch labels", value=dashboard_config["show_patch_labels_default"], key="event_dashboard_patch_labels")
        show_centroids = controls[2].checkbox("Show centroids", value=dashboard_config["show_patch_centroids_default"], key="event_dashboard_patch_centroids")
        show_boundaries = controls[3].checkbox("Show patch boundaries", value=True, key="event_dashboard_patch_boundaries")
    patch_options = sorted(patches.patch_id.dropna().astype(int).unique().tolist()) if "patch_id" in patches else []
    selected_patch_text = _selector("Selected local patch ID", [str(value) for value in patch_options], key="event_dashboard_patch_id")
    selected_patch = int(selected_patch_text) if selected_patch_text else None
    label_path = root / config["patch_detection"]["labels_output"]
    label_state = load_netcdf_date(label_path, event_date, required_variables=("patch_id", "valid_ocean_mask"))
    if label_state.status == "invalid":
        st.error(label_state.reason)
    elif not label_state.available:
        st.warning(label_state.reason or "Daily patch labels are unavailable")
    else:
        anomaly_layer = anomaly if event_date.normalize() == anomaly_date.normalize() else None
        if show_background and anomaly_layer is None:
            st.info("The optional anomaly background is shown only when the event date matches the main analysis date.")
        try:
            figure = plot_daily_patch_map(
                label_state.value, patches, event_date, anomaly=anomaly_layer,
                selected_patch=selected_patch, show_anomaly_background=show_background,
                show_patch_labels=show_labels, show_centroids=show_centroids,
                show_patch_boundaries=show_boundaries,
            )
        except (TypeError, ValueError) as exc:
            st.error(f"Daily patch map cannot be rendered: {exc}")
        else:
            st.pyplot(figure, width="stretch")
            plt.close(figure)
    if patches.empty:
        st.info("No retained daily patches are present on this date.")
        return
    patch_table = prepare_patch_table(products.table("daily_patches"), event_date)
    essential_columns = [
        "patch_id", "area_km2", "area_fraction_of_threshold_total",
        "centroid_latitude", "centroid_longitude", "mean_source_value",
        "maximum_source_value", "mean_exceedance", "maximum_exceedance",
    ]
    st.dataframe(
        patch_table[[column for column in essential_columns if column in patch_table]].head(maximum_rows),
        hide_index=True,
        width="stretch",
    )
    with st.expander("Complete patch geometry and technical details", expanded=False):
        st.dataframe(patch_table.head(maximum_rows), hide_index=True, width="stretch")
    st.download_button(
        "Download selected-date patch CSV", dataframe_to_iso_csv(patches),
        file_name=f"daily_patches_{event_date.date().isoformat()}.csv", mime="text/csv",
        key="event_dashboard_selected_date_patches_download",
    )
    if selected_patch is not None:
        selected = patches.loc[patches.patch_id.astype(int) == selected_patch]
        observation = rows_for_date(products.table("patch_observations"), event_date)
        if "local_patch_id" in observation:
            observation = observation.loc[observation.local_patch_id.astype(int) == selected_patch]
        else:
            observation = observation.iloc[0:0]
        if not selected.empty:
            with st.container(horizontal=True):
                st.metric("Area", format_scalar(selected.iloc[0].area_km2, "km²"), border=True)
                st.metric("Mean exceedance", format_scalar(selected.iloc[0].mean_exceedance, "°C"), border=True)
                st.metric("Maximum exceedance", format_scalar(selected.iloc[0].maximum_exceedance, "°C"), border=True)
        if observation.empty:
            st.warning("This local patch has no matching cached tracking observation.")
        else:
            row = observation.iloc[0]
            with st.expander("Tracking relationships and technical identifiers", expanded=False):
                st.write({
                    "Track": format_identifier(row.get("track_id"), language),
                    "Family": format_identifier(row.get("event_family_id"), language),
                    "Role": human_label(row.get("node_event_type"), language),
                    "Predecessors": row.get("parent_node_ids", []),
                    "Successors": row.get("child_node_ids", []),
                })
                st.caption(
                    f"{internal_id_caption(row.get('track_id'), language)} · "
                    f"{internal_id_caption(row.get('event_family_id'), language)}"
                )


def _track_panel(
    products: EventProducts,
    config: dict[str, Any],
    root: Path,
    event_date: pd.Timestamp,
    selected_track: str | None,
    *,
    maximum_arrows: int,
    language: str,
) -> None:
    if selected_track is None:
        st.info("Select a track in the shared filters to view its trajectory and metrics.")
        return
    observations = products.table("patch_observations")
    if "track_id" not in observations:
        st.warning("Cached patch observations are unavailable or structurally invalid.")
        return
    track_observations = observations.loc[observations.track_id.astype(str) == selected_track].copy()
    if track_observations.empty:
        st.warning("The selected track is no longer available after filtering.")
        return
    node_ids = set(track_observations.node_id.astype(str))
    all_edges = products.table("lineage_edges")
    track_edges = all_edges.loc[
        all_edges.predecessor_node_id.astype(str).isin(node_ids)
        & all_edges.successor_node_id.astype(str).isin(node_ids)
    ].copy() if not all_edges.empty else all_edges
    track_row = products.table("tracks")
    track_row = track_row.loc[track_row.track_id.astype(str) == selected_track]
    active = rows_for_date(track_observations, event_date)
    label_state = None
    numeric_id = None
    if not active.empty:
        mapping = products.table("track_mapping")
        match = mapping.loc[mapping.track_id.astype(str) == selected_track] if "track_id" in mapping else mapping.iloc[0:0]
        numeric_id = int(match.iloc[0].track_numeric_id) if not match.empty else None
        label_state = load_netcdf_date(
            root / config["patch_tracking"]["track_labels_output"], event_date,
            required_variables=("track_numeric_id", "event_family_numeric_id", "local_patch_id"),
        )
    try:
        figure = plot_track_map(
            label_state.value if label_state and label_state.available else None,
            track_observations, track_edges, event_date,
            track_numeric_id=numeric_id, maximum_arrows=maximum_arrows,
        )
    except ValueError as exc:
        st.info(str(exc))
    else:
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    if active.empty:
        st.info("This track is not active on the selected date; its complete trajectory is still shown.")
    if not track_row.empty:
        row = track_row.iloc[0]
        metrics = [
            ("Duration", row.get("duration_calendar_days"), "days"), ("Observed days", row.get("observed_days"), "days"),
            ("Maximum area", row.get("maximum_area_km2"), "km²"), ("Area-days", row.get("observed_area_days_km2_days"), "km² days"),
            ("Cumulative severity", row.get("cumulative_severity"), "°C km² days"), ("Trajectory length", row.get("trajectory_length_km"), "km"),
            ("Net displacement", row.get("net_displacement_km"), "km"), ("Mean speed", row.get("mean_speed_km_per_day"), "km/day"),
            ("Maximum speed", row.get("maximum_speed_km_per_day"), "km/day"), ("Tortuosity", row.get("trajectory_tortuosity"), ""),
            ("Expansion days", row.get("expansion_day_count"), "days"), ("Contraction days", row.get("contraction_day_count"), "days"),
            ("Family ID", row.get("event_family_id"), ""),
        ]
        essential_indexes = (0, 2, 4, 5, 7, 12)
        with st.container(horizontal=True):
            for index in essential_indexes:
                label, value, unit = metrics[index]
                st.metric(
                    label,
                    format_identifier(value, language) if label == "Family ID" else format_scalar(value, unit),
                    border=True,
                )
        with st.expander("Supporting track metrics and technical details", expanded=False):
            with st.container(horizontal=True):
                for index, (label, value, unit) in enumerate(metrics):
                    if index in essential_indexes:
                        continue
                    st.metric(
                        label,
                        format_identifier(value, language) if label == "Family ID" else format_scalar(value, unit),
                        border=True,
                    )
            st.caption(internal_id_caption(selected_track, language))
    daily = prepare_track_daily_series(track_observations, all_edges, selected_track)
    specs = [
        ("area_km2", "Area through time", "km²"), ("mean_exceedance", "Mean exceedance", "°C"),
        ("maximum_exceedance", "Maximum exceedance", "°C"), ("centroid_latitude", "Centroid latitude", "degrees north"),
        ("centroid_longitude", "Centroid longitude", "degrees east"), ("speed_km_per_day", "Speed", "km/day"),
        ("area_change_km2_per_day", "Daily expansion or contraction", "km²/day"),
    ]
    def render_metric_charts(chart_specs: list[tuple[str, str, str]]) -> None:
        for start in range(0, len(chart_specs), 2):
            columns = st.columns(2)
            for column, (metric, title, unit) in zip(columns, chart_specs[start : start + 2], strict=False):
                with column.container(border=True):
                    chart = temporal_metric_chart(daily, value_column=metric, selected_date=event_date, title=title, unit=unit)
                    if chart is None:
                        st.info(f"{title} is not calculated for this track.")
                    else:
                        st.altair_chart(chart, width="stretch")

    render_metric_charts([specs[index] for index in (0, 1, 3, 4)])
    with st.expander("Detailed temporal track diagnostics", expanded=False):
        render_metric_charts([specs[index] for index in (2, 5, 6)])
    st.markdown("#### Non-cartographic trajectory")
    try:
        trajectory = plot_track_trajectory(track_observations, event_date, maximum_arrows)
    except ValueError as exc:
        st.info(str(exc))
    else:
        st.pyplot(trajectory, width="stretch")
        plt.close(trajectory)


def _family_panel(
    products: EventProducts,
    config: dict[str, Any],
    root: Path,
    event_date: pd.Timestamp,
    selected_family: str | None,
    selected_track: str | None,
    *,
    maximum_nodes: int,
    maximum_rows: int,
) -> None:
    if selected_family is None:
        st.info("Select an event family (or one of its tracks) in the shared filters.")
        return
    observations = products.table("patch_observations")
    if "event_family_id" not in observations:
        st.warning("Cached patch observations are unavailable or structurally invalid.")
        return
    family = observations.loc[observations.event_family_id.astype(str) == selected_family].copy()
    if family.empty:
        st.warning("The selected family is no longer available after filtering.")
        return
    node_ids = set(family.node_id.astype(str))
    edges = products.table("lineage_edges")
    family_edges = edges.loc[
        edges.predecessor_node_id.astype(str).isin(node_ids)
        & edges.successor_node_id.astype(str).isin(node_ids)
    ].copy() if not edges.empty else edges
    with st.expander("Family map options", expanded=False):
        controls = st.columns(5)
        show_all = controls[0].checkbox("All family tracks", value=config["event_dashboard"]["show_family_tracks_default"], key="event_dashboard_family_all_tracks")
        show_links = controls[1].checkbox("Split/merge links", value=True, key="event_dashboard_family_links")
        show_centroid = controls[2].checkbox("Family centroid", value=True, key="event_dashboard_family_centroid")
        show_footprints = controls[3].checkbox("Patch footprints", value=True, key="event_dashboard_family_footprints")
        controls[4].caption("Track colors are categorical, not severity.")
    active = rows_for_date(family, event_date)
    label_state = None
    family_numeric_id = None
    if not active.empty:
        mapping = products.table("family_mapping")
        match = mapping.loc[mapping.event_family_id.astype(str) == selected_family] if "event_family_id" in mapping else mapping.iloc[0:0]
        family_numeric_id = int(match.iloc[0].event_family_numeric_id) if not match.empty else None
        label_state = load_netcdf_date(
            root / config["patch_tracking"]["track_labels_output"], event_date,
            required_variables=("track_numeric_id", "event_family_numeric_id", "local_patch_id"),
        )
    try:
        figure = plot_family_map(
            label_state.value if label_state and label_state.available else None,
            family, family_edges, event_date, family_numeric_id=family_numeric_id,
            show_all_tracks=show_all, show_split_merge_links=show_links,
            show_family_centroid=show_centroid, show_patch_footprints=show_footprints,
        )
    except ValueError as exc:
        st.info(str(exc))
    else:
        st.pyplot(figure, width="stretch")
        plt.close(figure)
    family_table = products.table("event_families")
    family_row = (
        family_table.loc[family_table.event_family_id.astype(str) == selected_family]
        if "event_family_id" in family_table else family_table.iloc[0:0]
    )
    if not family_row.empty:
        row = family_row.iloc[0]
        summary = [
            ("Start", row.get("start_date")), ("End", row.get("end_date")),
            ("Duration", format_scalar(row.get("duration_calendar_days"), "days")),
            ("Tracks", format_scalar(row.get("track_count"), precision=0)),
            ("Patch observations", format_scalar(row.get("unique_patch_observation_count"), precision=0)),
            ("Splits", format_scalar(row.get("split_count"), precision=0)),
            ("Merges", format_scalar(row.get("merge_count"), precision=0)),
            ("Complex branches", format_scalar(row.get("complex_branch_count"), precision=0)),
            ("Maximum simultaneous patches", format_scalar(row.get("maximum_simultaneous_patch_count"), precision=0)),
            ("Maximum daily area", format_scalar(row.get("maximum_total_daily_area_km2"), "km²")),
            ("Family area-days", format_scalar(row.get("family_area_days_km2_days"), "km² days")),
            ("Cumulative severity", format_scalar(row.get("cumulative_severity"), "°C km² days")),
            ("Trajectory length", format_scalar(row.get("family_trajectory_length_km"), "km")),
            ("Net displacement", format_scalar(row.get("family_net_displacement_km"), "km")),
            ("Maximum extent", format_scalar(row.get("maximum_family_extent_km"), "km")),
            ("Boundary contact days", format_scalar(row.get("boundary_contact_day_count"), "days", precision=0)),
        ]
        essential_indexes = (0, 1, 2, 3, 4, 9, 10, 11)
        with st.container(horizontal=True):
            for index in essential_indexes:
                label, value = summary[index]
                if isinstance(value, pd.Timestamp):
                    value = value.date().isoformat()
                st.metric(label, str(value), border=True)
        with st.expander("Supporting family metrics and technical identifiers", expanded=False):
            with st.container(horizontal=True):
                for index, (label, value) in enumerate(summary):
                    if index in essential_indexes:
                        continue
                    if isinstance(value, pd.Timestamp):
                        value = value.date().isoformat()
                    st.metric(label, str(value), border=True)
            st.caption(internal_id_caption(selected_family, "en"))
    daily = prepare_family_daily_series(family, selected_family)
    family_specs = [
        ("total_family_area_km2", "Total family area", "km²"), ("patch_count", "Daily patch count", "patches"),
        ("centroid_latitude", "Family centroid latitude", "degrees north"), ("centroid_longitude", "Family centroid longitude", "degrees east"),
        ("mean_exceedance", "Area-weighted mean exceedance", "°C"), ("maximum_exceedance", "Maximum exceedance", "°C"),
    ]
    def render_family_charts(chart_specs: list[tuple[str, str, str]]) -> None:
        for start in range(0, len(chart_specs), 2):
            columns = st.columns(2)
            for column, (metric, title, unit) in zip(columns, chart_specs[start : start + 2], strict=False):
                with column.container(border=True):
                    chart = temporal_metric_chart(daily, value_column=metric, selected_date=event_date, title=title, unit=unit)
                    if chart is None:
                        st.info(f"No finite values are available for {title.lower()}.")
                    else:
                        st.altair_chart(chart, width="stretch")

    render_family_charts([family_specs[index] for index in (0, 1, 2, 3)])
    with st.expander("Detailed family diagnostics and lineage", expanded=False):
        render_family_charts([family_specs[index] for index in (4, 5)])
        lineage = lineage_chart(family, family_edges, selected_date=event_date, selected_track=selected_track, maximum_nodes=maximum_nodes)
        if lineage.chart is None:
            st.info("This family has no lineage edges or drawable observations.")
        else:
            if lineage.simplified:
                st.warning(f"The lineage view was simplified to {lineage.displayed_node_count} nodes; use the tables for complete detail.")
            st.altair_chart(lineage.chart, width="stretch")
        observation_table, edge_table, track_table = prepare_family_tables(
            selected_family, observations, edges, products.table("tracks")
        )
        table_tabs = st.tabs(["Patch observations", "Lineage edges", "Track summaries"])
        for tab, table in zip(table_tabs, (observation_table, edge_table, track_table), strict=True):
            with tab:
                if table.empty:
                    st.info("No rows are available for this relationship table.")
                else:
                    st.dataframe(table.head(maximum_rows), hide_index=True, width="stretch")


def _downloads_panel(
    products: EventProducts,
    *,
    selected_track: str | None,
    selected_family: str | None,
    language: str,
) -> None:
    st.markdown(f"#### {tr('cached_downloads', language)}")
    downloads = [
        ("Univariate events", products.table("univariate_events"), "univariate_events.csv"),
        ("Univariate daily flags", products.table("univariate_flags"), "univariate_event_daily_flags.csv"),
        ("Daily patches", products.table("daily_patches"), "daily_patches.csv"),
        ("Daily patch summary", products.table("daily_patch_summary"), "daily_patch_summary.csv"),
        ("Patch observations", products.table("patch_observations"), "spatiotemporal_patch_observations.csv"),
        ("Lineage edges", products.table("lineage_edges"), "patch_lineage_edges.csv"),
        ("Tracks", products.table("tracks"), "spatiotemporal_tracks.csv"),
        ("Event families", products.table("event_families"), "spatiotemporal_event_families.csv"),
    ]
    columns = st.columns(4)
    for index, (label, frame, name) in enumerate(downloads):
        columns[index % 4].download_button(
            f"{label} CSV", dataframe_to_iso_csv(frame), name, "text/csv",
            key=f"event_dashboard_complete_{name}", disabled=frame.empty,
        )
    selected_track_data = products.table("tracks")
    selected_track_data = (
        selected_track_data.loc[selected_track_data.track_id.astype(str) == str(selected_track)]
        if selected_track and "track_id" in selected_track_data else selected_track_data.iloc[0:0]
    )
    selected_family_data = products.table("event_families")
    selected_family_data = (
        selected_family_data.loc[selected_family_data.event_family_id.astype(str) == str(selected_family)]
        if selected_family and "event_family_id" in selected_family_data else selected_family_data.iloc[0:0]
    )
    json_columns = st.columns(3)
    json_columns[0].download_button(
        "Selected track JSON", record_to_json(selected_track_data), "selected_track.json", "application/json",
        key="event_dashboard_track_json", disabled=selected_track_data.empty,
    )
    json_columns[1].download_button(
        "Selected family JSON", record_to_json(selected_family_data), "selected_family.json", "application/json",
        key="event_dashboard_family_json", disabled=selected_family_data.empty,
    )
    summary = products.tracking_summary.value or {}
    json_columns[2].download_button(
        "Tracking summary JSON", record_to_json(summary), "spatiotemporal_tracking_summary.json", "application/json",
        key="event_dashboard_summary_json", disabled=not summary,
    )


def render_thermal_events_tab(
    products: EventProducts,
    config: dict[str, Any],
    *,
    analysis_date: str | pd.Timestamp,
    representativeness: pd.DataFrame,
    anomaly: xr.DataArray | None = None,
    project_root: str | Path = ".",
) -> None:
    """Render the complete Increment 4D interface from cached products only."""
    language = "en"
    st.subheader(tr("thermal_events", language))
    st.caption(
        "Cached Increment 4 products: regional events, date-local patches, non-branching tracks, "
        "and split/merge event families. No detection or tracking is recalculated in this dashboard."
    )
    _product_status_messages(products)
    flags = products.table("univariate_flags")
    patch_summary = products.table("daily_patch_summary")
    date_values = pd.concat([
        pd.to_datetime(flags.get("date", pd.Series(dtype="datetime64[ns]")), errors="coerce"),
        pd.to_datetime(patch_summary.get("date", pd.Series(dtype="datetime64[ns]")), errors="coerce"),
    ]).dropna().dt.normalize().drop_duplicates().sort_values()
    if date_values.empty:
        st.warning("No event-product dates are available. Existing dashboard tabs remain usable.")
        return
    date_options = [value.date().isoformat() for value in date_values]
    requested_date = pd.Timestamp(analysis_date).date().isoformat()
    if requested_date not in date_options:
        st.info("The main analysis date is outside the cached event products; using the latest cached event date.")
        requested_date = date_options[-1]
    previous_synced_date = st.session_state.get("event_dashboard_synced_main_date")
    if "event_dashboard_analysis_date" not in st.session_state:
        st.session_state["event_dashboard_analysis_date"] = requested_date
    elif previous_synced_date and previous_synced_date != requested_date and st.session_state["event_dashboard_analysis_date"] == previous_synced_date:
        st.session_state["event_dashboard_analysis_date"] = requested_date
    elif st.session_state["event_dashboard_analysis_date"] not in date_options:
        st.session_state["event_dashboard_analysis_date"] = requested_date
    st.session_state["event_dashboard_synced_main_date"] = requested_date
    header_columns = st.columns([2, 1])
    event_date_text = header_columns[0].selectbox(
        tr("event_analysis_date", language), date_options, key="event_dashboard_analysis_date",
        help=filter_description("event_analysis_date", language),
    )
    if header_columns[1].button(
        tr("reset_event_filters", language), icon=":material/restart_alt:", width="stretch",
        help=filter_description("reset_event_filters", language),
    ):
        _reset_event_filters()
        st.rerun()
    event_date = pd.Timestamp(event_date_text)
    events = products.table("univariate_events")
    patches = products.table("daily_patches")
    observations = products.table("patch_observations")
    tracks = products.table("tracks")
    families = products.table("event_families")
    source_options = sorted(set(events.get("source_variable", pd.Series(dtype=object)).dropna().astype(str)) | set(patches.get("source_variable", pd.Series(dtype=object)).dropna().astype(str)))
    threshold_options = sorted(set(events.get("threshold_type", pd.Series(dtype=object)).dropna().astype(str)) | set(patches.get("threshold_type", pd.Series(dtype=object)).dropna().astype(str)))
    direction_options = sorted(set(events.get("direction", pd.Series(dtype=object)).dropna().astype(str)) | set(patches.get("direction", pd.Series(dtype=object)).dropna().astype(str)))
    status_options = sorted(set(events.get("status", pd.Series(dtype=object)).dropna().astype(str)) | set(tracks.get("status", pd.Series(dtype=object)).dropna().astype(str)))
    with st.expander("Additional event filters", expanded=False):
        filter_columns = st.columns(4)
        format_value = lambda value: tr("all", language) if value == tr("all", language) else human_label(value, language)
        with filter_columns[0]:
            source_variable = _selector(
                tr("source_variable", language), source_options, key="event_dashboard_source",
                all_label=tr("all", language), help_text=filter_description("source_variable", language),
                format_func=format_value,
            )
        with filter_columns[1]:
            threshold_type = _selector(
                tr("threshold_type", language), threshold_options, key="event_dashboard_threshold",
                all_label=tr("all", language), help_text=filter_description("threshold_type", language),
                format_func=format_value,
            )
        with filter_columns[2]:
            direction = _selector(
                tr("direction", language), direction_options, key="event_dashboard_direction",
                all_label=tr("all", language), help_text=filter_description("direction", language),
                format_func=format_value,
            )
        with filter_columns[3]:
            event_status = _selector(
                tr("event_status", language), status_options, key="event_dashboard_status",
                all_label=tr("all", language), help_text=filter_description("event_status", language),
                format_func=format_value,
            )
        numeric_filters = st.columns(2)
        minimum_duration = numeric_filters[0].number_input(
            tr("minimum_track_duration", language), min_value=0.0, value=0.0,
            key="event_dashboard_minimum_duration", help=filter_description("minimum_track_duration", language),
        )
        minimum_area = numeric_filters[1].number_input(
            tr("minimum_track_area", language), min_value=0.0, value=0.0,
            key="event_dashboard_minimum_area", help=filter_description("minimum_track_area", language),
        )
        with st.expander(tr("how_use_filters", language), expanded=False):
            for key in ("event_analysis_date", "source_variable", "threshold_type", "direction", "event_status", "minimum_track_duration", "minimum_track_area", "family_selector", "track_selector"):
                st.write(f"**{tr(key, language)}:** {filter_description(key, language)}")
    filtered_tracks = filter_catalogue(tracks, minimum_duration=minimum_duration, minimum_area=minimum_area)
    if event_status and "status" in filtered_tracks:
        filtered_tracks = filtered_tracks.loc[filtered_tracks.status.astype(str) == event_status]
    family_options = sorted(families.event_family_id.dropna().astype(str).unique()) if "event_family_id" in families else []
    selection_columns = st.columns(2)
    with selection_columns[0]:
        requested_family = _selector(
            tr("family_selector", language), family_options, key="event_dashboard_family_id",
            all_label=tr("all", language), help_text=filter_description("family_selector", language),
            format_func=lambda value: tr("all", language) if value == tr("all", language) else format_identifier(value, language),
        )
    candidate_tracks = filtered_tracks.loc[filtered_tracks.event_family_id.astype(str) == requested_family] if requested_family else filtered_tracks
    track_options = sorted(candidate_tracks.track_id.dropna().astype(str).unique()) if "track_id" in candidate_tracks else []
    with selection_columns[1]:
        selected_track = _selector(
            tr("track_selector", language), track_options, key="event_dashboard_track_id",
            all_label=tr("all", language), help_text=filter_description("track_selector", language),
            format_func=lambda value: tr("all", language) if value == tr("all", language) else format_identifier(value, language),
        )
    selected_family = requested_family
    if selected_track:
        match = filtered_tracks.loc[filtered_tracks.track_id.astype(str) == selected_track]
        if not match.empty:
            selected_family = str(match.iloc[0].event_family_id)
            st.caption(f"{format_identifier(selected_track, language)} → {format_identifier(selected_family, language)}")
            with st.expander("Technical identifiers", expanded=False):
                st.caption(
                    f"{internal_id_caption(selected_track, language)} · "
                    f"{internal_id_caption(selected_family, language)}"
                )
    if requested_family and candidate_tracks.empty:
        st.info("No tracks in the selected family satisfy the duration and area filters.")

    dashboard_config = config["event_dashboard"]
    subtab_labels = [
        tr("event_overview", language), tr("daily_patches", language),
        tr("tracks_trajectories", language), tr("families_lineage", language),
    ]
    configured_default = {
        "event_overview": subtab_labels[0], "daily_patches": subtab_labels[1],
        "tracks": subtab_labels[2], "event_families": subtab_labels[3],
    }.get(dashboard_config["default_subtab"], subtab_labels[0])
    if st.session_state.get("event_dashboard_subtab") not in (None, *subtab_labels):
        del st.session_state["event_dashboard_subtab"]
    sub_tabs = st.tabs(
        subtab_labels, default=configured_default, key="event_dashboard_subtab", on_change="rerun"
    )
    if sub_tabs[0].open:
        with sub_tabs[0]:
            _overview_panel(
                products, event_date, representativeness,
                source_variable=source_variable, threshold_type=threshold_type,
                direction=direction, event_status=event_status,
                maximum_rows=dashboard_config["maximum_table_rows"],
                language=language,
            )
            with st.expander("Cached thermal-event downloads", expanded=False):
                _downloads_panel(
                    products, selected_track=selected_track, selected_family=selected_family,
                    language=language,
                )
    if sub_tabs[1].open:
        with sub_tabs[1]:
            _patch_panel(
                products, config, Path(project_root), event_date,
                anomaly=anomaly, anomaly_date=pd.Timestamp(analysis_date),
                maximum_rows=dashboard_config["maximum_table_rows"],
                language=language,
            )
    if sub_tabs[2].open:
        with sub_tabs[2]:
            _track_panel(
                products, config, Path(project_root), event_date, selected_track,
                maximum_arrows=dashboard_config["trajectory_maximum_arrows"],
                language=language,
            )
    if sub_tabs[3].open:
        with sub_tabs[3]:
            _family_panel(
                products, config, Path(project_root), event_date, selected_family, selected_track,
                maximum_nodes=dashboard_config["maximum_lineage_nodes"],
                maximum_rows=dashboard_config["maximum_table_rows"],
            )
