from __future__ import annotations

import matplotlib.figure
import numpy as np
import pandas as pd
import xarray as xr

import src.event_maps as event_maps
from src.event_maps import categorical_patch_colormap, plot_daily_patch_map, plot_family_map, plot_track_map
from src.plotting import NINO12_EXTENT


def _labels(values: np.ndarray | None = None) -> xr.Dataset:
    values = np.asarray(values if values is not None else [[0, 1, 1], [0, 0, 2], [0, 2, 2]], dtype=np.int32)
    coords = {"latitude": [-6.0, -5.5, -5.0], "longitude": [-86.0, -85.5, -85.0]}
    return xr.Dataset({
        "patch_id": (("latitude", "longitude"), values),
        "local_patch_id": (("latitude", "longitude"), values),
        "track_numeric_id": (("latitude", "longitude"), np.where(values == 1, 4, 0)),
        "event_family_numeric_id": (("latitude", "longitude"), np.where(values > 0, 7, 0)),
        "valid_ocean_mask": (("latitude", "longitude"), np.ones_like(values, dtype=np.uint8)),
    }, coords=coords)


def _patches() -> pd.DataFrame:
    return pd.DataFrame({"patch_id": [1, 2], "centroid_longitude": [-85.5, -85.2], "centroid_latitude": [-6.0, -5.2]})


def _observations() -> pd.DataFrame:
    return pd.DataFrame({
        "node_id": ["A", "B"], "date": pd.to_datetime(["2026-01-01", "2026-01-02"]),
        "track_id": ["TRK000001", "TRK000001"], "event_family_id": ["FAM000001", "FAM000001"],
        "node_event_type": ["appearance", "termination"], "area_km2": [10.0, 11.0],
        "centroid_longitude": [-85.5, -85.2], "centroid_latitude": [-5.5, -5.3],
    })


def _edges() -> pd.DataFrame:
    return pd.DataFrame({"predecessor_node_id": ["A"], "successor_node_id": ["B"], "lineage_relation": ["continuation"], "crosses_temporal_gap": [False]})


def test_patch_id_zero_is_transparent() -> None:
    cmap, _ = categorical_patch_colormap(3)
    assert cmap.colors[0][3] == 0.0


def test_daily_patch_map_fixed_extent_white_background_and_selected_highlight(monkeypatch) -> None:
    monkeypatch.setattr(event_maps, "ccrs", None)
    monkeypatch.setattr(event_maps, "cfeature", None)
    figure = plot_daily_patch_map(_labels(), _patches(), "2026-01-01", selected_patch=1)
    assert isinstance(figure, matplotlib.figure.Figure)
    axis = figure.axes[0]
    assert np.allclose((*axis.get_xlim(), *axis.get_ylim()), NINO12_EXTENT)
    assert axis.get_facecolor()[:3] == (1.0, 1.0, 1.0)
    assert len(axis.collections) >= 2


def test_no_patch_date_still_returns_a_figure(monkeypatch) -> None:
    monkeypatch.setattr(event_maps, "ccrs", None)
    monkeypatch.setattr(event_maps, "cfeature", None)
    figure = plot_daily_patch_map(_labels(np.zeros((3, 3))), pd.DataFrame(columns=_patches().columns), "2026-01-01")
    assert isinstance(figure, matplotlib.figure.Figure)
    assert np.allclose((*figure.axes[0].get_xlim(), *figure.axes[0].get_ylim()), NINO12_EXTENT)


def test_cartopy_failure_falls_back_to_matplotlib(monkeypatch) -> None:
    monkeypatch.setattr(event_maps, "ccrs", object())
    monkeypatch.setattr(event_maps, "cfeature", object())
    monkeypatch.setattr(event_maps, "_new_axes", lambda use_cartopy: (_ for _ in ()).throw(RuntimeError("features unavailable")) if use_cartopy else __import__("matplotlib.pyplot").pyplot.subplots())
    figure = plot_daily_patch_map(_labels(), _patches(), "2026-01-01")
    assert isinstance(figure, matplotlib.figure.Figure)


def test_track_map_has_trajectory_selected_point_and_fixed_extent(monkeypatch) -> None:
    monkeypatch.setattr(event_maps, "ccrs", None)
    monkeypatch.setattr(event_maps, "cfeature", None)
    figure = plot_track_map(_labels(), _observations(), _edges(), "2026-01-02", track_numeric_id=4)
    assert isinstance(figure, matplotlib.figure.Figure)
    assert np.allclose((*figure.axes[0].get_xlim(), *figure.axes[0].get_ylim()), NINO12_EXTENT)
    assert figure.axes[0].lines


def test_family_map_supports_one_track_and_empty_lineage(monkeypatch) -> None:
    monkeypatch.setattr(event_maps, "ccrs", None)
    monkeypatch.setattr(event_maps, "cfeature", None)
    figure = plot_family_map(_labels(), _observations(), pd.DataFrame(), "2026-01-02", family_numeric_id=7)
    assert isinstance(figure, matplotlib.figure.Figure)
    assert np.allclose((*figure.axes[0].get_xlim(), *figure.axes[0].get_ylim()), NINO12_EXTENT)

