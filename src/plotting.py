"""Map plotting helpers for experimental thermal fields."""

from __future__ import annotations

from pathlib import Path

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
except ImportError:  # Optional dependency: Matplotlib fallback remains complete.
    ccrs = None
    cfeature = None

NINO12_EXTENT = (-90.0, -80.0, -10.0, 0.0)


def plot_sst_map(sst: xr.DataArray, title: str = "Experimental SST") -> matplotlib.figure.Figure:
    """Create an SST map using the standard operational presentation."""
    if "time" in sst.dims:
        sst = sst.isel(time=-1)
    return plot_thermal_field(
        sst, title=title, colorbar_label="SST (°C)", cmap="turbo", vmin=18, vmax=32
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
    """Create a black-background Niño 1+2 map with optional Cartopy context."""
    values = np.asarray(field.values)
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        raise ValueError(f"Cannot plot {field.name or 'field'}: no finite values")
    options: dict[str, float] = {}
    if vmin is not None:
        options["vmin"] = vmin
    if vmax is not None:
        options["vmax"] = vmax
    if symmetric and vmin is None and vmax is None:
        limit = max(float(np.nanpercentile(np.abs(finite), 98)), 0.1)
        options.update(vmin=-limit, vmax=limit)

    projection = ccrs.PlateCarree() if ccrs is not None else None
    subplot_kw = {"projection": projection} if projection is not None else {}
    figure, axis = plt.subplots(
        figsize=(7.2, 5.2), constrained_layout=True, subplot_kw=subplot_kw,
        facecolor="black",
    )
    axis.set_facecolor("black")
    color_map = plt.get_cmap(cmap).copy()
    color_map.set_bad("black")
    plot_options: dict[str, object] = {}
    if projection is not None:
        plot_options["transform"] = projection
    image = field.plot(
        ax=axis, x="longitude", y="latitude", cmap=color_map,
        cbar_kwargs={"label": colorbar_label, "shrink": 0.88},
        **options, **plot_options,
    )
    west, east, south, north = NINO12_EXTENT
    if projection is not None and cfeature is not None:
        axis.set_extent([west, east, south, north], crs=projection)
        axis.add_feature(cfeature.LAND, facecolor="black", edgecolor="white", linewidth=0.5)
        axis.coastlines(color="white", linewidth=0.7)
        axis.add_feature(cfeature.BORDERS, edgecolor="white", linewidth=0.5)
        grid = axis.gridlines(draw_labels=True, color="white", alpha=0.18, linewidth=0.5)
        grid.top_labels = False
        grid.right_labels = False
        grid.xlabel_style = {"color": "white"}
        grid.ylabel_style = {"color": "white"}
    else:
        axis.set_xlim(west, east)
        axis.set_ylim(south, north)
        axis.grid(color="white", alpha=0.18, linewidth=0.5)
    axis.set_title(title, color="white")
    axis.set_xlabel("Longitude (°E)", color="white")
    axis.set_ylabel("Latitude (°N)", color="white")
    axis.tick_params(colors="white")
    for spine in axis.spines.values():
        spine.set_color("white")
    colorbar = image.colorbar
    colorbar.ax.yaxis.label.set_color("white")
    colorbar.ax.tick_params(colors="white")
    colorbar.outline.set_edgecolor("white")
    return figure


def save_figure(figure: matplotlib.figure.Figure, path: str | Path) -> Path:
    """Save a PNG figure and release its Matplotlib resources."""
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, bbox_inches="tight", facecolor="black")
    plt.close(figure)
    return output
