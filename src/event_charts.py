"""Charts for univariate events, tracks, and lineage relationships."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import altair as alt
import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd

from src.event_ui_utils import add_temporal_segments, finite_domain


RELATION_COLORS = {
    "continuation": "#4C78A8",
    "split": "#F58518",
    "merge": "#54A24B",
    "complex": "#B279A2",
}


def univariate_event_timeline(
    flags: pd.DataFrame,
    events: pd.DataFrame,
    selected_date: str | pd.Timestamp,
) -> alt.LayerChart | None:
    """Show source, threshold, exceedances, event intervals, peaks, and date."""
    required = {"date", "value", "threshold", "intensity", "event_id", "event_day"}
    if flags.empty or not required.issubset(flags.columns):
        return None
    frame = flags[list(required)].copy()
    frame["date"] = pd.to_datetime(frame.date, errors="coerce")
    for column in ("value", "threshold", "intensity"):
        frame[column] = pd.to_numeric(frame[column], errors="coerce").replace([np.inf, -np.inf], np.nan)
    frame = add_temporal_segments(frame, value_columns=("value", "threshold"))
    domain = finite_domain(pd.concat([frame.value, frame.threshold]), padding_fraction=0.05)
    if domain is None:
        return None
    tooltip = [
        alt.Tooltip("date:T", title="Date"),
        alt.Tooltip("value:Q", title="Value", format=".3f"),
        alt.Tooltip("threshold:Q", title="Threshold", format=".3f"),
        alt.Tooltip("intensity:Q", title="Intensity", format=".3f"),
        alt.Tooltip("event_id:N", title="Event ID"),
        alt.Tooltip("event_day:Q", title="Event day", format=".0f"),
    ]
    source = alt.Chart(frame).mark_line(color="#0072B2", strokeWidth=2, invalid="break-paths-show-domains").encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title="Source value", scale=alt.Scale(domain=list(domain), zero=False)),
        detail="segment_id:N",
        tooltip=tooltip,
    )
    threshold = alt.Chart(frame).mark_line(color="#D55E00", strokeDash=[7, 4], invalid="break-paths-show-domains").encode(
        x="date:T", y=alt.Y("threshold:Q", scale=alt.Scale(domain=list(domain), zero=False)), detail="segment_id:N"
    )
    exceedance = alt.Chart(frame.loc[frame.event_id.notna()]).mark_point(color="#CC79A7", filled=True, size=45).encode(
        x="date:T", y="value:Q", tooltip=tooltip
    )
    layers: list[alt.Chart] = []
    if not events.empty and {"start_date", "end_date"}.issubset(events.columns):
        intervals = events[["event_id", "start_date", "end_date"]].copy()
        intervals["start_date"] = pd.to_datetime(intervals.start_date, errors="coerce")
        intervals["end_date"] = pd.to_datetime(intervals.end_date, errors="coerce") + pd.Timedelta(days=1)
        intervals = intervals.dropna(subset=["start_date", "end_date"])
        if not intervals.empty:
            layers.append(alt.Chart(intervals).mark_rect(color="#E69F00", opacity=0.12).encode(
                x="start_date:T", x2="end_date:T", tooltip=["event_id:N", "start_date:T", "end_date:T"]
            ))
    layers.extend([source, threshold, exceedance])
    if not events.empty and "peak_date" in events.columns:
        peaks = events[["event_id", "peak_date"]].copy()
        peaks["date"] = pd.to_datetime(peaks.peak_date, errors="coerce")
        peaks = peaks.merge(frame[["date", "value"]], on="date", how="left").dropna(subset=["date", "value"])
        if not peaks.empty:
            layers.append(alt.Chart(peaks).mark_point(shape="diamond", color="#000000", filled=True, size=90).encode(
                x="date:T", y="value:Q", tooltip=["event_id:N", alt.Tooltip("date:T", title="Peak date")]
            ))
    date_rule = alt.Chart(pd.DataFrame({"selected_date": [pd.Timestamp(selected_date)]})).mark_rule(
        color="#009E73", strokeDash=[4, 3], strokeWidth=2
    ).encode(x="selected_date:T")
    layers.append(date_rule)
    return alt.layer(*layers).properties(
        title="Regional univariate thermal-event timeline",
        description="Source and active threshold with detected event intervals and selected date.",
        height=340,
    ).interactive()


def temporal_metric_chart(
    data: pd.DataFrame,
    *,
    value_column: str,
    selected_date: str | pd.Timestamp,
    title: str,
    unit: str,
    date_column: str = "date",
) -> alt.LayerChart | None:
    """Create a gap-aware single-metric chart with a selected-date rule."""
    if data.empty or date_column not in data or value_column not in data:
        return None
    frame = data[[date_column, value_column]].copy().rename(columns={date_column: "date", value_column: "value"})
    frame = add_temporal_segments(frame, value_columns=("value",))
    domain = finite_domain(frame.value, padding_fraction=0.05)
    if domain is None:
        return None
    line = alt.Chart(frame).mark_line(point=True, invalid="break-paths-show-domains").encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("value:Q", title=f"{title} ({unit})" if unit else title, scale=alt.Scale(domain=list(domain), zero=False)),
        detail="segment_id:N",
        tooltip=[alt.Tooltip("date:T", title="Date"), alt.Tooltip("value:Q", title=f"{title} ({unit})", format=".3f")],
    )
    rule = alt.Chart(pd.DataFrame({"selected_date": [pd.Timestamp(selected_date)]})).mark_rule(
        color="#D55E00", strokeDash=[5, 4]
    ).encode(x="selected_date:T")
    return (line + rule).properties(title=title, height=250).interactive()


def plot_track_trajectory(
    observations: pd.DataFrame,
    selected_date: str | pd.Timestamp,
    maximum_arrows: int = 25,
    *,
    allow_gap_pairs: set[tuple[str, str]] | None = None,
) -> matplotlib.figure.Figure:
    """Plot a dynamic non-cartographic track trajectory without bridging gaps."""
    required = {"date", "centroid_longitude", "centroid_latitude"}
    missing = required - set(observations.columns)
    if missing:
        raise ValueError(f"Track trajectory is missing columns: {sorted(missing)}")
    if maximum_arrows < 0:
        raise ValueError("maximum_arrows must be non-negative")
    frame = observations[list(required)].copy()
    frame["date"] = pd.to_datetime(frame.date, errors="coerce")
    for coordinate in ("centroid_longitude", "centroid_latitude"):
        frame[coordinate] = pd.to_numeric(frame[coordinate], errors="coerce")
    frame = frame[
        frame.date.notna()
        & np.isfinite(frame.centroid_longitude)
        & np.isfinite(frame.centroid_latitude)
    ].sort_values("date", kind="mergesort").reset_index(drop=True)
    if frame.empty:
        raise ValueError("No finite track centroid coordinates are available")
    allow_gap_pairs = allow_gap_pairs or set()
    segments: list[tuple[int, int, bool]] = []
    for index in range(len(frame) - 1):
        first = frame.date.iloc[index]
        second = frame.date.iloc[index + 1]
        key = (first.date().isoformat(), second.date().isoformat())
        elapsed = (second - first).days
        if elapsed == 1 or key in allow_gap_pairs:
            segments.append((index, index + 1, elapsed > 1))
    figure, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True, facecolor="white")
    axis.set_facecolor("white")
    longitude = frame.centroid_longitude.to_numpy(dtype=float)
    latitude = frame.centroid_latitude.to_numpy(dtype=float)
    for start, end, bridged in segments:
        axis.plot(
            longitude[start : end + 1], latitude[start : end + 1],
            color="0.3", linewidth=1.5, linestyle="--" if bridged else "-", zorder=1,
        )
    arrow_count = min(maximum_arrows, len(segments))
    chosen = np.unique(np.linspace(0, len(segments) - 1, arrow_count, dtype=int)) if arrow_count else []
    for segment_index in chosen:
        start, end, _ = segments[int(segment_index)]
        axis.add_patch(FancyArrowPatch(
            (longitude[start], latitude[start]), (longitude[end], latitude[end]),
            arrowstyle="-|>", mutation_scale=10, color="0.25", linewidth=0.9, zorder=2,
        ))
    markers = [(0, "Start", "#009E73", "o"), (len(frame) - 1, "End", "#D55E00", "s")]
    selected = pd.Timestamp(selected_date).normalize()
    selected_indices = frame.index[frame.date.dt.normalize() == selected].tolist()
    if selected_indices:
        markers.append((selected_indices[-1], "Selected date", "#F0E442", "*"))
    for index, label, color, marker in markers:
        axis.scatter(longitude[index], latitude[index], s=85, color=color, marker=marker, edgecolor="black", label=label, zorder=4)
        axis.annotate(frame.date.iloc[index].date().isoformat(), (longitude[index], latitude[index]), xytext=(5, 5), textcoords="offset points", fontsize=8)
    x_domain = finite_domain(longitude, padding_fraction=0.08, minimum_padding=0.10)
    y_domain = finite_domain(latitude, padding_fraction=0.08, minimum_padding=0.10)
    assert x_domain is not None and y_domain is not None
    axis.set_xlim(*x_domain)
    axis.set_ylim(*y_domain)
    axis.set_xlabel("Longitude (degrees east)")
    axis.set_ylabel("Latitude (degrees north)")
    axis.set_title("Track centroid trajectory")
    axis.grid(color="0.85", linewidth=0.6)
    axis.legend(loc="best")
    return figure


@dataclass(frozen=True)
class LineageChartResult:
    chart: alt.LayerChart | None
    simplified: bool
    displayed_node_count: int


def lineage_chart(
    observations: pd.DataFrame,
    edges: pd.DataFrame,
    *,
    selected_date: str | pd.Timestamp,
    selected_track: str | None = None,
    maximum_nodes: int = 300,
) -> LineageChartResult:
    """Create a deterministic date-versus-track lineage graph."""
    if observations.empty or not {"node_id", "date", "track_id"}.issubset(observations.columns):
        return LineageChartResult(None, False, 0)
    nodes = observations.copy()
    nodes["date"] = pd.to_datetime(nodes.date, errors="coerce")
    nodes = nodes.dropna(subset=["node_id", "date", "track_id"]).sort_values(["date", "track_id", "node_id"], kind="mergesort")
    simplified = len(nodes) > maximum_nodes
    if simplified:
        highlighted = nodes.loc[nodes.track_id.astype(str) == str(selected_track)] if selected_track else nodes.iloc[0:0]
        backbone_ids = set()
        if not edges.empty and "is_continuation_backbone" in edges:
            backbone = edges.loc[edges.is_continuation_backbone.fillna(False).astype(bool)]
            backbone_ids.update(backbone.predecessor_node_id.astype(str))
            backbone_ids.update(backbone.successor_node_id.astype(str))
        sample = nodes.loc[nodes.node_id.astype(str).isin(backbone_ids)]
        nodes = pd.concat([highlighted, sample, nodes.groupby("track_id", sort=False).head(1)]).drop_duplicates("node_id").head(maximum_nodes)
    node_ids = set(nodes.node_id.astype(str))
    selected_edges = edges.loc[
        edges.predecessor_node_id.astype(str).isin(node_ids)
        & edges.successor_node_id.astype(str).isin(node_ids)
    ].copy() if not edges.empty else edges.copy()
    lookup = nodes.set_index(nodes.node_id.astype(str))
    edge_rows: list[dict[str, Any]] = []
    for row in selected_edges.itertuples(index=False):
        predecessor = str(row.predecessor_node_id)
        successor = str(row.successor_node_id)
        if predecessor not in lookup.index or successor not in lookup.index:
            continue
        edge_rows.append({
            "x": lookup.loc[predecessor, "date"], "x2": lookup.loc[successor, "date"],
            "y": str(lookup.loc[predecessor, "track_id"]), "y2": str(lookup.loc[successor, "track_id"]),
            "relation": getattr(row, "lineage_relation", "continuation"),
        })
    edge_frame = pd.DataFrame(edge_rows, columns=["x", "x2", "y", "y2", "relation"])
    node_chart = alt.Chart(nodes).mark_circle(size=70).encode(
        x=alt.X("date:T", title="Date"),
        y=alt.Y("track_id:N", title="Track / branch"),
        color=alt.condition(
            alt.datum.track_id == (selected_track or ""), alt.value("#D55E00"),
            alt.Color("node_event_type:N", title="Node-event type"),
        ),
        tooltip=[
            alt.Tooltip("node_id:N", title="Node ID"), alt.Tooltip("date:T", title="Date"),
            alt.Tooltip("local_patch_id:Q", title="Patch ID"), alt.Tooltip("track_id:N", title="Track ID"),
            alt.Tooltip("event_family_id:N", title="Family ID"), alt.Tooltip("node_event_type:N", title="Node type"),
            alt.Tooltip("area_km2:Q", title="Area (km²)", format=".1f"),
            alt.Tooltip("mean_exceedance:Q", title="Mean exceedance", format=".3f"),
        ],
    )
    charts: list[alt.Chart] = []
    if not edge_frame.empty:
        charts.append(alt.Chart(edge_frame).mark_rule(strokeWidth=1.6).encode(
            x="x:T", x2="x2:T", y="y:N", y2="y2:N",
            color=alt.Color(
                "relation:N", title="Lineage relation",
                scale=alt.Scale(domain=list(RELATION_COLORS), range=list(RELATION_COLORS.values())),
            ),
        ))
    charts.append(node_chart)
    charts.append(alt.Chart(pd.DataFrame({"selected_date": [pd.Timestamp(selected_date)]})).mark_rule(
        color="black", strokeDash=[5, 4]
    ).encode(x="selected_date:T"))
    chart = alt.layer(*charts).properties(title="Event-family lineage", height=max(260, min(620, 20 * nodes.track_id.nunique()))).interactive()
    return LineageChartResult(chart, simplified, len(nodes))

