"""Robust spatial map plotting for the Niño 1+2 region."""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.figure
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
import numpy as np
import xarray as xr

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:  # Cartopy is optional.
    ccrs = None
    cfeature = None

LOGGER = logging.getLogger(__name__)
NINO12_EXTENT = (-90.0, -80.0, -10.0, 0.0)
LATITUDE_ALIASES = ("latitude", "lat")
LONGITUDE_ALIASES = ("longitude", "lon")


def _coordinate_name(field: xr.DataArray, aliases: tuple[str, ...], kind: str) -> str:
    for alias in aliases:
        if alias in field.coords:
            return alias
    raise ValueError(f"Spatial field is missing a {kind} coordinate; accepted names: {aliases}")


def prepare_spatial_field(field: xr.DataArray) -> xr.DataArray:
    """Validate, normalize coordinate names, and order a two-dimensional field."""
    if not isinstance(field, xr.DataArray):
        raise TypeError(f"Spatial plotting requires xarray.DataArray, got {type(field).__name__}")
    latitude = _coordinate_name(field, LATITUDE_ALIASES, "latitude")
    longitude = _coordinate_name(field, LONGITUDE_ALIASES, "longitude")
    prepared = field.rename({latitude: "latitude", longitude: "longitude"})
    extra_dimensions = [
        dimension
        for dimension in prepared.dims
        if dimension not in {"latitude", "longitude"}
    ]
    for dimension in extra_dimensions:
        if prepared.sizes[dimension] != 1:
            raise ValueError(
                f"Spatial field must be two-dimensional after date selection; "
                f"dimension {dimension!r} has size {prepared.sizes[dimension]}"
            )
        prepared = prepared.isel({dimension: 0}, drop=True)
    if set(prepared.dims) != {"latitude", "longitude"} or prepared.ndim != 2:
        raise ValueError(
            "Spatial field must have exactly latitude and longitude dimensions; "
            f"received {prepared.dims}"
        )
    prepared = prepared.transpose("latitude", "longitude")
    finite = np.asarray(prepared.values)[np.isfinite(prepared.values)]
    if finite.size == 0:
        raise ValueError(f"Cannot plot {field.name or 'spatial field'}: all values are missing or non-finite")
    LOGGER.debug(
        "Spatial field %s: dims=%s coordinates=%s min=%s max=%s finite_cells=%s",
        field.name,
        prepared.dims,
        list(prepared.coords),
        float(finite.min()),
        float(finite.max()),
        finite.size,
    )
    return prepared


def _color_settings(cmap: str, vmin: float, vmax: float) -> tuple[object, dict[str, object]]:
    color_map = plt.get_cmap(cmap).with_extremes(bad="white")
    if vmin < 0 < vmax:
        return color_map, {"norm": TwoSlopeNorm(vmin=vmin, vcenter=0.0, vmax=vmax)}
    return color_map, {"vmin": vmin, "vmax": vmax}


def _style_colorbar(colorbar: object, label: str) -> None:
    colorbar.set_label(label, color="black")
    colorbar.ax.tick_params(colors="black")
    colorbar.outline.set_edgecolor("black")


def _plot_matplotlib(
    field: xr.DataArray, *, title: str, colorbar_label: str, cmap: str, vmin: float, vmax: float
) -> matplotlib.figure.Figure:
    figure, axis = plt.subplots(figsize=(7.2, 5.2), constrained_layout=True, facecolor="white")
    axis.set_facecolor("white")
    color_map, color_options = _color_settings(cmap, vmin, vmax)
    mesh = axis.pcolormesh(
        field.longitude.values,
        field.latitude.values,
        np.ma.masked_invalid(field.values),
        cmap=color_map,
        shading="auto",
        zorder=1,
        **color_options,
    )
    west, east, south, north = NINO12_EXTENT
    axis.set_xlim(west, east)
    axis.set_ylim(south, north)
    axis.set_xlabel("Longitude (°E)", color="black")
    axis.set_ylabel("Latitude (°N)", color="black")
    figure.suptitle(title, color="black", fontsize=14)
    axis.tick_params(colors="black")
    axis.grid(color="0.82", alpha=0.8, linewidth=0.55)
    for spine in axis.spines.values():
        spine.set_color("black")
    _style_colorbar(figure.colorbar(mesh, ax=axis, shrink=0.88), colorbar_label)
    return figure


def _plot_with_cartopy(
    field: xr.DataArray, *, title: str, colorbar_label: str, cmap: str, vmin: float, vmax: float
) -> matplotlib.figure.Figure:
    projection = ccrs.PlateCarree()
    figure, axis = plt.subplots(
        figsize=(7.2, 5.2), constrained_layout=True,
        subplot_kw={"projection": projection}, facecolor="white",
    )
    axis.set_facecolor("white")
    color_map, color_options = _color_settings(cmap, vmin, vmax)
    mesh = axis.pcolormesh(
        field.longitude.values,
        field.latitude.values,
        np.ma.masked_invalid(field.values),
        transform=ccrs.PlateCarree(),
        cmap=color_map,
        shading="auto",
        zorder=1,
        **color_options,
    )
    axis.set_extent(list(NINO12_EXTENT), crs=ccrs.PlateCarree())
    axis.add_feature(cfeature.LAND, facecolor="white", edgecolor="black", linewidth=0.6, zorder=5)
    axis.coastlines(resolution="50m", color="black", linewidth=0.9, zorder=6)
    axis.add_feature(cfeature.BORDERS, edgecolor="0.35", linewidth=0.5, zorder=6)
    grid = axis.gridlines(draw_labels=True, color="0.75", alpha=0.7, linewidth=0.5)
    grid.top_labels = False
    grid.right_labels = False
    grid.xlabel_style = {"color": "black"}
    grid.ylabel_style = {"color": "black"}
    figure.suptitle(title, color="black", fontsize=14)
    figure.supxlabel("Longitude (°E)", color="black")
    figure.supylabel("Latitude (°N)", color="black")
    _style_colorbar(figure.colorbar(mesh, ax=axis, shrink=0.88), colorbar_label)
    return figure


def plot_spatial_field(
    field: xr.DataArray,
    *,
    title: str,
    colorbar_label: str,
    cmap: str,
    vmin: float,
    vmax: float,
) -> matplotlib.figure.Figure:
    """Render a validated spatial field, falling back safely when Cartopy fails."""
    if not np.isfinite(vmin) or not np.isfinite(vmax) or vmin >= vmax:
        raise ValueError(f"Invalid color limits: vmin={vmin}, vmax={vmax}")
    prepared = prepare_spatial_field(field)
    if ccrs is not None and cfeature is not None:
        cartopy_figure: matplotlib.figure.Figure | None = None
        try:
            cartopy_figure = _plot_with_cartopy(
                prepared, title=title, colorbar_label=colorbar_label,
                cmap=cmap, vmin=vmin, vmax=vmax,
            )
            cartopy_figure.canvas.draw()  # Force Natural Earth failures before returning.
            return cartopy_figure
        except Exception as exc:
            LOGGER.warning("Cartopy rendering failed; using Matplotlib fallback: %s", exc)
            if cartopy_figure is not None:
                plt.close(cartopy_figure)
    return _plot_matplotlib(
        prepared, title=title, colorbar_label=colorbar_label,
        cmap=cmap, vmin=vmin, vmax=vmax,
    )


def plot_thermal_field(
    field: xr.DataArray,
    *,
    title: str,
    colorbar_label: str,
    cmap: str,
    symmetric: bool = False,
    vmin: float | None = None,
    vmax: float | None = None,
) -> matplotlib.figure.Figure:
    """Compatibility wrapper for callers using the former function name."""
    del symmetric
    if vmin is None or vmax is None:
        raise ValueError("Fixed vmin and vmax are required for comparable spatial maps")
    return plot_spatial_field(
        field, title=title, colorbar_label=colorbar_label,
        cmap=cmap, vmin=vmin, vmax=vmax,
    )


def plot_sst_map(sst: xr.DataArray, title: str = "Experimental SST") -> matplotlib.figure.Figure:
    """Create an SST map using fixed operational limits."""
    if "time" in sst.dims:
        sst = sst.isel(time=-1)
    return plot_spatial_field(
        sst, title=title, colorbar_label="SST (°C)", cmap="turbo", vmin=18, vmax=32
    )


def save_figure(figure: matplotlib.figure.Figure, path: str | Path) -> Path:
    """Save and close a figure after non-interactive rendering."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight", facecolor="white")
    plt.close(figure)
    return output
