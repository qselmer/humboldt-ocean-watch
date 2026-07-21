"""Cartographic views for cached daily patches, tracks, and event families."""

from __future__ import annotations

from collections.abc import Callable
import logging

import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.colors import BoundaryNorm, ListedColormap
from matplotlib.lines import Line2D
from matplotlib.patches import FancyArrowPatch
import numpy as np
import pandas as pd
import xarray as xr

from src.plotting import NINO12_EXTENT, prepare_spatial_field

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:  # pragma: no cover - environment-dependent optional path.
    ccrs = None
    cfeature = None


LOGGER = logging.getLogger(__name__)


def categorical_patch_colormap(maximum_label: int) -> tuple[ListedColormap, BoundaryNorm]:
    """Return categorical colors with local patch ID zero fully transparent."""
    count = max(int(maximum_label), 1)
    palette = plt.get_cmap("tab20", count)
    colors = [(1.0, 1.0, 1.0, 0.0), *[palette(index) for index in range(count)]]
    cmap = ListedColormap(colors).with_extremes(bad="white")
    return cmap, BoundaryNorm(np.arange(-0.5, count + 1.5), cmap.N)


def _coordinate_names(dataset: xr.Dataset) -> tuple[str, str]:
    latitude = next((name for name in ("latitude", "lat") if name in dataset.coords), None)
    longitude = next((name for name in ("longitude", "lon") if name in dataset.coords), None)
    if latitude is None or longitude is None:
        raise ValueError("Label slice must contain latitude/lat and longitude/lon coordinates")
    return latitude, longitude


def _validated_label_field(dataset: xr.Dataset, variable: str) -> xr.DataArray:
    if not isinstance(dataset, xr.Dataset):
        raise TypeError("Patch map input must be an xarray.Dataset")
    if variable not in dataset:
        raise ValueError(f"Label slice is missing {variable!r}")
    latitude, longitude = _coordinate_names(dataset)
    field = dataset[variable].rename({latitude: "latitude", longitude: "longitude"})
    extra = [dimension for dimension in field.dims if dimension not in {"latitude", "longitude"}]
    for dimension in extra:
        if field.sizes[dimension] != 1:
            raise ValueError(f"{variable!r} must be two-dimensional after date selection")
        field = field.isel({dimension: 0}, drop=True)
    if set(field.dims) != {"latitude", "longitude"}:
        raise ValueError(f"{variable!r} requires latitude and longitude dimensions")
    return field.transpose("latitude", "longitude")


def _new_axes(use_cartopy: bool) -> tuple[matplotlib.figure.Figure, object]:
    if use_cartopy:
        figure, axis = plt.subplots(
            figsize=(8.2, 6.0), constrained_layout=True, facecolor="white",
            subplot_kw={"projection": ccrs.PlateCarree()},
        )
    else:
        figure, axis = plt.subplots(figsize=(8.2, 6.0), constrained_layout=True, facecolor="white")
    axis.set_facecolor("white")
    return figure, axis


def _context(axis: object, *, use_cartopy: bool) -> None:
    west, east, south, north = NINO12_EXTENT
    if use_cartopy:
        axis.set_extent([west, east, south, north], crs=ccrs.PlateCarree())
        axis.add_feature(cfeature.LAND, facecolor="#FAFAFA", edgecolor="black", linewidth=0.6, zorder=20)
        axis.coastlines(resolution="50m", color="black", linewidth=1.6, zorder=21)
        axis.add_feature(cfeature.BORDERS, edgecolor="black", linewidth=0.9, zorder=21)
        admin = cfeature.NaturalEarthFeature(
            category="cultural", name="admin_1_states_provinces_lines", scale="10m", facecolor="none"
        )
        axis.add_feature(admin, edgecolor="black", linewidth=0.55, zorder=21)
        grid = axis.gridlines(draw_labels=True, color="0.78", linewidth=0.5, alpha=0.7)
        grid.top_labels = False
        grid.right_labels = False
    else:
        axis.set_xlim(west, east)
        axis.set_ylim(south, north)
        axis.set_xlabel("Longitude (degrees east)")
        axis.set_ylabel("Latitude (degrees north)")
        axis.grid(color="0.82", linewidth=0.5)
        for spine in axis.spines.values():
            spine.set_color("black")


def _xy_kwargs(use_cartopy: bool) -> dict[str, object]:
    return {"transform": ccrs.PlateCarree()} if use_cartopy else {}


def _safe_map(render: Callable[[bool], matplotlib.figure.Figure]) -> matplotlib.figure.Figure:
    if ccrs is not None and cfeature is not None:
        figure: matplotlib.figure.Figure | None = None
        try:
            figure = render(True)
            figure.canvas.draw()
            return figure
        except Exception as exc:
            LOGGER.warning("Cartopy event-map rendering failed; using Matplotlib fallback: %s", exc)
            if figure is not None:
                plt.close(figure)
    return render(False)


def plot_daily_patch_map(
    labels: xr.Dataset,
    patches: pd.DataFrame,
    selected_date: str | pd.Timestamp,
    *,
    anomaly: xr.DataArray | None = None,
    selected_patch: int | None = None,
    show_anomaly_background: bool = True,
    show_patch_labels: bool = True,
    show_centroids: bool = True,
    show_patch_boundaries: bool = True,
) -> matplotlib.figure.Figure:
    """Plot date-local patch IDs without assigning them temporal meaning."""
    field = _validated_label_field(labels, "patch_id")
    valid = _validated_label_field(labels, "valid_ocean_mask") if "valid_ocean_mask" in labels else xr.ones_like(field)
    maximum = int(np.nanmax(field.values)) if field.size else 0
    cmap, norm = categorical_patch_colormap(maximum)

    def render(use_cartopy: bool) -> matplotlib.figure.Figure:
        figure, axis = _new_axes(use_cartopy)
        kwargs = _xy_kwargs(use_cartopy)
        if show_anomaly_background and anomaly is not None:
            prepared = prepare_spatial_field(anomaly)
            anomaly_cmap = plt.get_cmap("RdBu_r").with_extremes(bad="white")
            background = axis.pcolormesh(
                prepared.longitude, prepared.latitude, np.ma.masked_invalid(prepared.values),
                cmap=anomaly_cmap, vmin=-5, vmax=5, shading="auto", zorder=1, **kwargs,
            )
            colorbar = figure.colorbar(background, ax=axis, shrink=0.82)
            colorbar.set_label("SST anomaly (°C)")
        masked = np.ma.masked_where((field.values <= 0) | (valid.values <= 0), field.values)
        axis.pcolormesh(field.longitude, field.latitude, masked, cmap=cmap, norm=norm, shading="auto", zorder=3, alpha=0.72, **kwargs)
        if show_patch_boundaries and maximum > 0:
            axis.contour(field.longitude, field.latitude, field.values, levels=np.arange(0.5, maximum + 0.5, 1), colors="black", linewidths=0.75, zorder=5, **kwargs)
        if selected_patch and bool(np.any(field.values == selected_patch)):
            axis.contour(field.longitude, field.latitude, (field.values == selected_patch).astype(int), levels=[0.5], colors="#F0E442", linewidths=2.8, zorder=7, **kwargs)
        if not patches.empty:
            for row in patches.itertuples(index=False):
                patch_id = int(getattr(row, "patch_id"))
                longitude = float(getattr(row, "centroid_longitude"))
                latitude = float(getattr(row, "centroid_latitude"))
                if show_centroids:
                    axis.scatter(longitude, latitude, marker="x", color="black", s=35, linewidth=1.2, zorder=8, **kwargs)
                if show_patch_labels:
                    axis.text(longitude, latitude, str(patch_id), ha="center", va="bottom", fontsize=8, color="black", zorder=9, **kwargs)
        _context(axis, use_cartopy=use_cartopy)
        axis.set_title(f"Daily threshold patches — {pd.Timestamp(selected_date).date().isoformat()}")
        return figure

    return _safe_map(render)


def _trajectory_segments(observations: pd.DataFrame, edges: pd.DataFrame) -> list[tuple[pd.Series, pd.Series, bool]]:
    ordered = observations.sort_values("date", kind="mergesort").reset_index(drop=True)
    segments: list[tuple[pd.Series, pd.Series, bool]] = []
    for index in range(len(ordered) - 1):
        first, second = ordered.iloc[index], ordered.iloc[index + 1]
        matching = edges.loc[
            (edges.predecessor_node_id.astype(str) == str(first.node_id))
            & (edges.successor_node_id.astype(str) == str(second.node_id))
        ] if not edges.empty else pd.DataFrame()
        if matching.empty:
            continue
        bridged = bool(matching.iloc[0].get("crosses_temporal_gap", False))
        segments.append((first, second, bridged))
    return segments


def plot_track_map(
    labels: xr.Dataset | None,
    observations: pd.DataFrame,
    edges: pd.DataFrame,
    selected_date: str | pd.Timestamp,
    *,
    track_numeric_id: int | None = None,
    maximum_arrows: int = 25,
) -> matplotlib.figure.Figure:
    """Plot one cached non-branching track and its selected-date footprint."""
    track = observations.copy()
    track["date"] = pd.to_datetime(track.date, errors="coerce")
    track = track.dropna(subset=["date", "centroid_latitude", "centroid_longitude"])
    if track.empty:
        raise ValueError("The selected track has no finite centroid observations")
    segments = _trajectory_segments(track, edges)

    def render(use_cartopy: bool) -> matplotlib.figure.Figure:
        figure, axis = _new_axes(use_cartopy)
        kwargs = _xy_kwargs(use_cartopy)
        if labels is not None and track_numeric_id is not None and "track_numeric_id" in labels:
            label = _validated_label_field(labels, "track_numeric_id")
            footprint = np.ma.masked_where(label.values != track_numeric_id, label.values)
            axis.pcolormesh(label.longitude, label.latitude, footprint, cmap=ListedColormap(["#56B4E9"]), shading="auto", alpha=0.65, zorder=2, **kwargs)
            if np.any(label.values == track_numeric_id):
                axis.contour(label.longitude, label.latitude, (label.values == track_numeric_id).astype(int), levels=[0.5], colors="#0072B2", linewidths=2.0, zorder=4, **kwargs)
        arrow_count = min(maximum_arrows, len(segments))
        arrow_indices = set(np.unique(np.linspace(0, len(segments) - 1, arrow_count, dtype=int))) if arrow_count else set()
        for index, (first, second, bridged) in enumerate(segments):
            longitude = [first.centroid_longitude, second.centroid_longitude]
            latitude = [first.centroid_latitude, second.centroid_latitude]
            axis.plot(longitude, latitude, color="0.25", linestyle="--" if bridged else "-", linewidth=1.6, zorder=6, **kwargs)
            if index in arrow_indices:
                arrow_kwargs = dict(arrowstyle="-|>", mutation_scale=10, color="0.2", linewidth=0.9, zorder=7)
                if use_cartopy:
                    arrow_kwargs["transform"] = ccrs.PlateCarree()
                axis.add_patch(FancyArrowPatch((longitude[0], latitude[0]), (longitude[1], latitude[1]), **arrow_kwargs))
        first, last = track.sort_values("date").iloc[[0, -1]].to_dict("records")
        axis.scatter(first["centroid_longitude"], first["centroid_latitude"], color="#009E73", marker="o", s=80, edgecolor="black", zorder=9, label="Start", **kwargs)
        axis.scatter(last["centroid_longitude"], last["centroid_latitude"], color="#D55E00", marker="s", s=80, edgecolor="black", zorder=9, label="End", **kwargs)
        selected = track.loc[track.date.dt.normalize() == pd.Timestamp(selected_date).normalize()]
        if not selected.empty:
            axis.scatter(selected.centroid_longitude, selected.centroid_latitude, color="#F0E442", marker="*", s=130, edgecolor="black", zorder=10, label="Selected date", **kwargs)
        node_styles = {"split_parent": ("^", "#F58518", "Split"), "split_child": ("^", "#F58518", "Split"), "merge_parent": ("v", "#54A24B", "Merge"), "merge_child": ("v", "#54A24B", "Merge"), "complex_branch": ("D", "#B279A2", "Complex branch")}
        for node_type, (marker, color, label_text) in node_styles.items():
            nodes = track.loc[track.node_event_type == node_type]
            if not nodes.empty:
                axis.scatter(nodes.centroid_longitude, nodes.centroid_latitude, marker=marker, color=color, s=70, edgecolor="black", zorder=9, label=label_text, **kwargs)
        _context(axis, use_cartopy=use_cartopy)
        axis.set_title(f"Selected track — {pd.Timestamp(selected_date).date().isoformat()}")
        handles, labels_text = axis.get_legend_handles_labels()
        if any(bridged for _, _, bridged in segments):
            handles.append(Line2D([0], [0], color="0.25", linestyle="--")); labels_text.append("Bridged gap")
        unique = dict(zip(labels_text, handles, strict=False))
        if unique:
            axis.legend(unique.values(), unique.keys(), loc="lower left", fontsize=8)
        return figure

    return _safe_map(render)


def plot_family_map(
    labels: xr.Dataset | None,
    observations: pd.DataFrame,
    edges: pd.DataFrame,
    selected_date: str | pd.Timestamp,
    *,
    family_numeric_id: int | None = None,
    show_all_tracks: bool = True,
    show_split_merge_links: bool = True,
    show_family_centroid: bool = True,
    show_patch_footprints: bool = True,
) -> matplotlib.figure.Figure:
    """Plot all cached track branches and lineage nodes in one family."""
    family = observations.copy()
    family["date"] = pd.to_datetime(family.date, errors="coerce")
    family = family.dropna(subset=["date", "centroid_latitude", "centroid_longitude"])
    if family.empty:
        raise ValueError("The selected family has no finite patch observations")
    target = pd.Timestamp(selected_date).normalize()
    if show_all_tracks:
        visible = family
    else:
        active_track_ids = set(
            family.loc[family.date.dt.normalize() == target, "track_id"].dropna().astype(str)
        )
        visible = family.loc[family.track_id.astype(str).isin(active_track_ids)]

    def render(use_cartopy: bool) -> matplotlib.figure.Figure:
        figure, axis = _new_axes(use_cartopy)
        kwargs = _xy_kwargs(use_cartopy)
        if show_patch_footprints and labels is not None and family_numeric_id is not None and "event_family_numeric_id" in labels:
            label = _validated_label_field(labels, "event_family_numeric_id")
            footprint = np.ma.masked_where(label.values != family_numeric_id, label.values)
            axis.pcolormesh(label.longitude, label.latitude, footprint, cmap=ListedColormap(["#E6F2FF"]), shading="auto", alpha=0.8, zorder=2, **kwargs)
        track_ids = sorted(visible.track_id.dropna().astype(str).unique())
        colors = plt.get_cmap("tab20", max(len(track_ids), 1))
        for index, track_id in enumerate(track_ids):
            track = visible.loc[visible.track_id.astype(str) == track_id].sort_values("date")
            axis.plot(track.centroid_longitude, track.centroid_latitude, marker="o", markersize=3, color=colors(index), linewidth=1.5, label=track_id, zorder=6, **kwargs)
            if not track.empty:
                axis.text(track.centroid_longitude.iloc[-1], track.centroid_latitude.iloc[-1], track_id, fontsize=7, color=colors(index), zorder=7, **kwargs)
        if show_split_merge_links and not edges.empty:
            lookup = family.set_index(family.node_id.astype(str))
            for edge in edges.itertuples(index=False):
                predecessor, successor = str(edge.predecessor_node_id), str(edge.successor_node_id)
                if predecessor not in lookup.index or successor not in lookup.index:
                    continue
                first, second = lookup.loc[predecessor], lookup.loc[successor]
                relation = getattr(edge, "lineage_relation", "continuation")
                if relation != "continuation":
                    axis.plot([first.centroid_longitude, second.centroid_longitude], [first.centroid_latitude, second.centroid_latitude], color={"split": "#F58518", "merge": "#54A24B", "complex": "#B279A2"}.get(relation, "0.4"), linestyle=":", linewidth=1.4, zorder=5, **kwargs)
        branch = family.loc[family.node_event_type.isin(["split_parent", "split_child", "merge_parent", "merge_child", "complex_branch"])]
        if not branch.empty:
            axis.scatter(branch.centroid_longitude, branch.centroid_latitude, color="#B279A2", marker="D", s=55, edgecolor="black", zorder=9, label="Lineage branch", **kwargs)
        if show_family_centroid:
            daily = []
            for date, group in family.groupby("date", sort=True):
                weights = group.area_km2.to_numpy(dtype=float)
                daily.append((date, np.average(group.centroid_longitude, weights=weights), np.average(group.centroid_latitude, weights=weights)))
            centroid = pd.DataFrame(daily, columns=["date", "longitude", "latitude"])
            axis.plot(centroid.longitude, centroid.latitude, color="black", linewidth=2.3, marker=".", label="Family centroid", zorder=8, **kwargs)
            selected = centroid.loc[centroid.date.dt.normalize() == target]
            if not selected.empty:
                axis.scatter(selected.longitude, selected.latitude, color="#F0E442", marker="*", s=130, edgecolor="black", zorder=10, label="Selected date", **kwargs)
        _context(axis, use_cartopy=use_cartopy)
        axis.set_title(f"Event family — {pd.Timestamp(selected_date).date().isoformat()}")
        handles, labels_text = axis.get_legend_handles_labels()
        unique = dict(zip(labels_text, handles, strict=False))
        axis.legend(unique.values(), unique.keys(), loc="lower left", fontsize=7, ncols=2)
        return figure

    return _safe_map(render)
