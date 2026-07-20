"""Tests for robust Niño 1+2 spatial plotting."""

import matplotlib.figure
import matplotlib.pyplot as plt
import numpy as np
import pytest
import xarray as xr

import src.plotting as plotting


def field(dims=("latitude", "longitude"), coordinates=None) -> xr.DataArray:
    coordinates = coordinates or {"latitude": [-10.0, 0.0], "longitude": [-90.0, -80.0]}
    return xr.DataArray([[1.0, 2.0], [3.0, 4.0]], dims=dims, coords=coordinates, name="test_field")


def test_valid_two_dimensional_field_returns_figure() -> None:
    figure = plotting.plot_spatial_field(field(), title="Test", colorbar_label="Value", cmap="turbo", vmin=0, vmax=5)
    assert isinstance(figure, matplotlib.figure.Figure)
    plt.close(figure)


def test_lat_lon_aliases_are_detected() -> None:
    aliased = field(dims=("lat", "lon"), coordinates={"lat": [-10.0, 0.0], "lon": [-90.0, -80.0]})
    prepared = plotting.prepare_spatial_field(aliased)
    assert prepared.dims == ("latitude", "longitude")


def test_longitude_latitude_field_is_transposed() -> None:
    transposed = xr.DataArray(
        [[1.0, 3.0], [2.0, 4.0]],
        dims=("longitude", "latitude"),
        coords={"longitude": [-90.0, -80.0], "latitude": [-10.0, 0.0]},
    )
    prepared = plotting.prepare_spatial_field(transposed)
    assert prepared.dims == ("latitude", "longitude")
    np.testing.assert_allclose(prepared, [[1.0, 2.0], [3.0, 4.0]])


def test_all_nan_field_is_rejected() -> None:
    with pytest.raises(ValueError, match="all values"):
        plotting.prepare_spatial_field(xr.full_like(field(), np.nan))


def test_cartopy_failure_falls_back_to_matplotlib(monkeypatch) -> None:
    monkeypatch.setattr(plotting, "ccrs", object())
    monkeypatch.setattr(plotting, "cfeature", object())
    monkeypatch.setattr(plotting, "_plot_with_cartopy", lambda *args, **kwargs: (_ for _ in ()).throw(RuntimeError("feature failure")))
    figure = plotting.plot_spatial_field(field(), title="Fallback", colorbar_label="Value", cmap="turbo", vmin=0, vmax=5)
    assert isinstance(figure, matplotlib.figure.Figure)
    plt.close(figure)


def test_admin1_feature_failure_falls_back_to_matplotlib(monkeypatch) -> None:
    if plotting.ccrs is None or plotting.cfeature is None:
        pytest.skip("Cartopy is not installed")
    monkeypatch.setattr(
        plotting.cfeature,
        "NaturalEarthFeature",
        lambda **kwargs: (_ for _ in ()).throw(RuntimeError("admin-1 unavailable")),
    )
    figure = plotting.plot_spatial_field(
        field(), title="Admin fallback", colorbar_label="Value", cmap="turbo", vmin=0, vmax=5
    )
    assert isinstance(figure, matplotlib.figure.Figure)
    np.testing.assert_allclose(figure.axes[0].get_xlim(), [-90.0, -80.0])
    plt.close(figure)


def test_fixed_nino12_extent_and_white_background() -> None:
    figure = plotting.plot_spatial_field(field(), title="Extent", colorbar_label="Value", cmap="turbo", vmin=0, vmax=5)
    axis = figure.axes[0]
    np.testing.assert_allclose(axis.get_xlim(), [-90.0, -80.0])
    np.testing.assert_allclose(axis.get_ylim(), [-10.0, 0.0])
    np.testing.assert_allclose(figure.get_facecolor(), (1.0, 1.0, 1.0, 1.0))
    np.testing.assert_allclose(axis.get_facecolor(), (1.0, 1.0, 1.0, 1.0))
    plt.close(figure)
