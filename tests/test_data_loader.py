"""Tests for SST unit normalization, alias discovery, and Niño 1+2 subsetting."""

import numpy as np
import pytest
import xarray as xr

from src.data_loader import (
    COPERNICUS_CACHED_MODE,
    SYNTHETIC_DEMO_MODE,
    find_sst_variable,
    load_active_sst_dataset,
    normalize_sst,
    subset_nino12,
)


def _temperature_dataset(variable: str, values: list[float], units: str) -> xr.Dataset:
    dataset = xr.Dataset({variable: ("point", values)})
    dataset[variable].attrs["units"] = units
    return dataset


def test_kelvin_sst_is_converted_to_celsius() -> None:
    dataset = _temperature_dataset("analysed_sst", [273.15, 298.15, np.nan], "K")

    normalized = normalize_sst(dataset, "analysed_sst")

    np.testing.assert_allclose(normalized.sst.values, [0.0, 25.0, np.nan], equal_nan=True)
    assert normalized.sst.attrs["units"] == "degrees_Celsius"


def test_celsius_sst_is_not_converted_again() -> None:
    dataset = _temperature_dataset("sst", [20.0, 25.0, np.nan], "degrees_Celsius")

    normalized = normalize_sst(dataset, "sst")

    np.testing.assert_allclose(normalized.sst.values, [20.0, 25.0, np.nan], equal_nan=True)


@pytest.mark.parametrize(
    "alias", ["analysed_sst", "sst", "thetao", "sea_surface_temperature"]
)
def test_recognizes_supported_sst_aliases(alias: str) -> None:
    dataset = _temperature_dataset(alias, [25.0], "degrees_Celsius")

    assert find_sst_variable(dataset) == alias


@pytest.mark.parametrize(
    "latitudes", [np.array([-15.0, -10.0, -5.0, 0.0, 5.0]), np.array([5.0, 0.0, -5.0, -10.0, -15.0])]
)
def test_nino12_subset_handles_ascending_and_descending_latitude(
    latitudes: np.ndarray,
) -> None:
    data = xr.DataArray(
        np.ones((latitudes.size, 5)),
        dims=("latitude", "longitude"),
        coords={"latitude": latitudes, "longitude": [-95.0, -90.0, -85.0, -80.0, -75.0]},
    )

    subset = subset_nino12(data)

    np.testing.assert_array_equal(np.sort(subset.latitude.values), [-10.0, -5.0, 0.0])
    np.testing.assert_array_equal(subset.longitude.values, [-90.0, -85.0, -80.0])


@pytest.mark.parametrize(
    ("longitudes", "expected"),
    [
        ([-95.0, -90.0, -85.0, -80.0, -75.0], [-90.0, -85.0, -80.0]),
        ([265.0, 270.0, 275.0, 280.0, 285.0], [-90.0, -85.0, -80.0]),
    ],
)
def test_nino12_subset_handles_both_longitude_conventions(
    longitudes: list[float], expected: list[float]
) -> None:
    data = xr.DataArray(
        np.ones((3, len(longitudes))),
        dims=("latitude", "longitude"),
        coords={"latitude": [-10.0, -5.0, 0.0], "longitude": longitudes},
    )

    subset = subset_nino12(data)

    np.testing.assert_array_equal(subset.longitude.values, expected)
    assert float(subset.longitude.min()) == -90.0
    assert float(subset.longitude.max()) == -80.0


def _daily_sst_dataset() -> xr.Dataset:
    return xr.Dataset(
        {"sst": (("time", "latitude", "longitude"), np.ones((2, 2, 2)) * 25)},
        coords={"time": ["2026-01-01", "2026-01-02"], "latitude": [-10.0, 0.0], "longitude": [-90.0, -80.0]},
    )


def test_live_data_has_priority(tmp_path, monkeypatch) -> None:
    live = tmp_path / "live.nc"
    live.touch()
    calls: list[str] = []

    def fake_load(path, **kwargs):
        calls.append(str(path))
        return _daily_sst_dataset()

    monkeypatch.setattr("src.data_loader.load_sst_dataset", fake_load)
    _, mode = load_active_sst_dataset(live, tmp_path / "demo.nc")
    assert mode == COPERNICUS_CACHED_MODE
    assert calls == [str(live)]


def test_demo_is_used_when_live_file_is_missing(tmp_path, monkeypatch) -> None:
    demo = tmp_path / "demo.nc"
    calls: list[str] = []

    def fake_load(path, **kwargs):
        calls.append(str(path))
        return _daily_sst_dataset()

    monkeypatch.setattr("src.data_loader.load_sst_dataset", fake_load)
    _, mode = load_active_sst_dataset(tmp_path / "missing.nc", demo)
    assert mode == SYNTHETIC_DEMO_MODE
    assert calls == [str(demo)]
