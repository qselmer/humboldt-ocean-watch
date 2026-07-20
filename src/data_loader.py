"""NetCDF discovery, validation, normalization, and demo generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import pandas as pd
import xarray as xr

LOGGER = logging.getLogger(__name__)
DEFAULT_SST_ALIASES = (
    "analysed_sst",
    "sst",
    "thetao",
    "sea_surface_temperature",
)
NINO12_LONGITUDE_BOUNDS = (-90.0, -80.0)
NINO12_LATITUDE_BOUNDS = (-10.0, 0.0)
COPERNICUS_CACHED_MODE = "Copernicus cached data"
SYNTHETIC_DEMO_MODE = "Synthetic demonstration data"


def find_sst_variable(
    dataset: xr.Dataset, aliases: Sequence[str] = DEFAULT_SST_ALIASES
) -> str:
    """Return the first recognized SST variable name."""
    for alias in aliases:
        if alias in dataset.data_vars:
            return alias
    available = ", ".join(dataset.data_vars) or "none"
    raise ValueError(
        "No recognized SST variable found. Expected one of "
        f"{list(aliases)}; available data variables: {available}."
    )


def _temperature_is_kelvin(data: xr.DataArray) -> bool:
    units = str(data.attrs.get("units", "")).strip().lower()
    if units in {"k", "kelvin", "degrees_kelvin", "degree_kelvin"}:
        return True
    if units in {"c", "°c", "degc", "celsius", "degrees_celsius", "degree_celsius"}:
        return False
    finite = np.asarray(data.values)[np.isfinite(data.values)]
    if finite.size == 0:
        raise ValueError("SST variable contains no finite temperature values")
    median = float(np.median(finite))
    if median > 150.0:
        LOGGER.warning("SST units are missing or ambiguous; values indicate Kelvin")
        return True
    if -10.0 <= median <= 60.0:
        LOGGER.warning("SST units are missing or ambiguous; values indicate Celsius")
        return False
    raise ValueError(
        f"Cannot infer SST units from median value {median:.2f}; add a valid units attribute"
    )


def normalize_sst(dataset: xr.Dataset, variable: str) -> xr.Dataset:
    """Return a dataset with SST renamed to ``sst`` and expressed in Celsius."""
    result = dataset.copy()
    data = result[variable]
    if _temperature_is_kelvin(data):
        data = data - 273.15
        LOGGER.info("Converted SST from Kelvin to degrees Celsius")
    data.attrs = dict(data.attrs)
    data.attrs.update(units="degrees_Celsius", long_name="sea surface temperature")
    if variable != "sst":
        result = result.drop_vars(variable)
    result["sst"] = data
    return result


def subset_nino12(dataset: xr.Dataset | xr.DataArray) -> xr.Dataset | xr.DataArray:
    """Subset Niño 1+2, accepting either common longitude convention.

    Returned longitudes use the -180 to 180 convention and are sorted. Boolean
    selection makes latitude ordering irrelevant.
    """
    missing = {"latitude", "longitude"} - set(dataset.coords)
    if missing:
        raise ValueError(f"Data is missing required coordinates: {sorted(missing)}")

    longitude = dataset["longitude"]
    normalized_longitude = (longitude + 180.0) % 360.0 - 180.0
    normalized = dataset.assign_coords(longitude=normalized_longitude).sortby("longitude")
    lon_min, lon_max = NINO12_LONGITUDE_BOUNDS
    lat_min, lat_max = NINO12_LATITUDE_BOUNDS
    in_region = normalized.where(
        (normalized["longitude"] >= lon_min)
        & (normalized["longitude"] <= lon_max)
        & (normalized["latitude"] >= lat_min)
        & (normalized["latitude"] <= lat_max),
        drop=True,
    )
    if in_region.sizes.get("longitude", 0) == 0 or in_region.sizes.get("latitude", 0) == 0:
        raise ValueError("Dataset does not overlap the Niño 1+2 study region")
    return in_region


def validate_dataset(dataset: xr.Dataset) -> None:
    """Validate normalized SST coordinates and values."""
    missing = {"time", "latitude", "longitude"} - set(dataset.coords)
    if missing:
        raise ValueError(f"SST dataset is missing required coordinates: {sorted(missing)}")
    if "sst" not in dataset.data_vars:
        raise ValueError("Normalized dataset must contain an 'sst' variable")
    if dataset.sizes.get("time", 0) == 0:
        raise ValueError("SST dataset has no time steps")
    if not bool(np.isfinite(dataset["sst"]).any()):
        raise ValueError("SST dataset contains no finite values")


def generate_synthetic_sst(
    path: str | Path,
    *,
    longitude: tuple[float, float] = (-90.0, -80.0),
    latitude: tuple[float, float] = (-10.0, 0.0),
    days: int = 10,
    seed: int = 42,
) -> Path:
    """Generate a compact deterministic daily SST NetCDF demo dataset."""
    if days < 2:
        raise ValueError("Synthetic dataset requires at least two days")
    output = Path(path)
    output.parent.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(seed)
    lons = np.linspace(*longitude, 21)
    lats = np.linspace(*latitude, 21)
    times = pd.date_range("2024-01-01", periods=days, freq="D")
    lon_grid, lat_grid = np.meshgrid(lons, lats)
    baseline = 25.0 + 0.12 * (lat_grid + 10.0) + 0.05 * (lon_grid + 90.0)
    values = np.stack(
        [baseline + 0.08 * index + rng.normal(0.0, 0.08, baseline.shape) for index in range(days)]
    )
    dataset = xr.Dataset(
        {"analysed_sst": (("time", "latitude", "longitude"), values + 273.15)},
        coords={"time": times, "latitude": lats, "longitude": lons},
        attrs={
            "title": "Synthetic daily SST for Humboldt Ocean Watch",
            "product_status": "experimental",
        },
    )
    dataset["analysed_sst"].attrs.update(units="kelvin", standard_name="sea_surface_temperature")
    dataset.to_netcdf(output)
    LOGGER.info("Generated synthetic SST dataset at %s", output)
    return output


def load_sst_dataset(
    path: str | Path,
    *,
    aliases: Sequence[str] = DEFAULT_SST_ALIASES,
    create_demo_if_missing: bool = True,
    demo_options: dict[str, Any] | None = None,
) -> xr.Dataset:
    """Load SST NetCDF, creating synthetic data when the requested file is absent."""
    source = Path(path)
    if not source.exists():
        if not create_demo_if_missing:
            raise FileNotFoundError(f"SST NetCDF file not found: {source}")
        LOGGER.warning("SST file %s does not exist; creating synthetic demo data", source)
        generate_synthetic_sst(source, **(demo_options or {}))
    try:
        with xr.open_dataset(source) as opened:
            dataset = opened.load()
    except (OSError, ValueError) as exc:
        raise ValueError(f"Could not read SST NetCDF file {source}: {exc}") from exc
    variable = find_sst_variable(dataset, aliases)
    normalized = normalize_sst(dataset, variable)
    validate_dataset(normalized)
    normalized.attrs["source_path"] = str(source)
    return normalized


def load_active_sst_dataset(
    live_path: str | Path,
    demo_path: str | Path,
    *,
    aliases: Sequence[str] = DEFAULT_SST_ALIASES,
    demo_options: dict[str, Any] | None = None,
) -> tuple[xr.Dataset, str]:
    """Load cached Copernicus SST first, falling back to synthetic demo SST."""
    live_source = Path(live_path)
    if live_source.exists():
        dataset = load_sst_dataset(
            live_source, aliases=aliases, create_demo_if_missing=False
        )
        mode = COPERNICUS_CACHED_MODE
    else:
        dataset = load_sst_dataset(
            demo_path,
            aliases=aliases,
            create_demo_if_missing=True,
            demo_options=demo_options,
        )
        mode = SYNTHETIC_DEMO_MODE
    dataset = subset_nino12(dataset)
    assert isinstance(dataset, xr.Dataset)
    dataset.attrs.update(data_mode=mode)
    return dataset.sortby("time"), mode
