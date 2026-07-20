"""Plotting helpers for experimental thermal fields."""

from __future__ import annotations

import matplotlib.figure
import matplotlib.pyplot as plt
import xarray as xr


def plot_sst_map(sst: xr.DataArray, title: str = "Experimental SST") -> matplotlib.figure.Figure:
    """Create a minimal longitude/latitude SST map."""
    if "time" in sst.dims:
        sst = sst.isel(time=-1)
    figure, axis = plt.subplots(figsize=(7, 5))
    image = sst.plot(ax=axis, x="longitude", y="latitude", cmap="turbo", cbar_kwargs={"label": "°C"})
    axis.set_title(title)
    image.axes.set_xlabel("Longitude")
    image.axes.set_ylabel("Latitude")
    return figure
