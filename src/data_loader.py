"""NetCDF discovery, validation, normalization, and demo generation."""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Mapping, Sequence

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

LATITUDE_ALIASES = ("latitude", "lat")
LONGITUDE_ALIASES = ("longitude", "lon")


def _coordinate_alias(
    data: xr.Dataset | xr.DataArray, aliases: Sequence[str], canonical: str
) -> str:
    matches = [name for name in aliases if name in data.coords]
    if canonical in matches:
        return canonical
    if len(matches) == 1:
        return matches[0]
    if not matches:
        raise ValueError(f"Data is missing required coordinate: {canonical}")
    raise ValueError(f"Ambiguous {canonical} coordinates: {matches}")


def normalize_longitude_coordinates(
    data: xr.Dataset | xr.DataArray,
) -> xr.Dataset | xr.DataArray:
    """Return an independent object with sorted -180..180 longitudes."""
    longitude_name = _coordinate_alias(data, LONGITUDE_ALIASES, "longitude")
    result = data.copy()
    if longitude_name != "longitude":
        result = result.rename({longitude_name: "longitude"})
    values = np.asarray(result.longitude.values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Longitude coordinate must be one-dimensional and finite")
    normalized = (values + 180.0) % 360.0 - 180.0
    if np.unique(np.round(normalized, 12)).size != normalized.size:
        raise ValueError("Longitude normalization creates duplicate coordinates")
    return result.assign_coords(longitude=normalized).sortby("longitude")


def normalize_latitude_coordinates(
    data: xr.Dataset | xr.DataArray,
) -> xr.Dataset | xr.DataArray:
    """Return an independent object with a sorted canonical latitude coordinate."""
    latitude_name = _coordinate_alias(data, LATITUDE_ALIASES, "latitude")
    result = data.copy()
    if latitude_name != "latitude":
        result = result.rename({latitude_name: "latitude"})
    values = np.asarray(result.latitude.values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all():
        raise ValueError("Latitude coordinate must be one-dimensional and finite")
    if np.unique(np.round(values, 12)).size != values.size:
        raise ValueError("Latitude coordinate contains duplicates")
    return result.sortby("latitude")


def normalize_spatial_coordinates(
    data: xr.Dataset | xr.DataArray,
) -> xr.Dataset | xr.DataArray:
    """Canonicalize longitude and latitude names, conventions, and ordering."""
    return normalize_latitude_coordinates(normalize_longitude_coordinates(data))


def validate_domain_overlap(
    data: xr.Dataset | xr.DataArray,
    longitude_bounds: tuple[float, float],
    latitude_bounds: tuple[float, float],
) -> None:
    """Raise when a canonical data grid does not intersect requested bounds."""
    if longitude_bounds[0] >= longitude_bounds[1] or latitude_bounds[0] >= latitude_bounds[1]:
        raise ValueError("Geographic bounds must be finite and ordered")
    if not np.isfinite((*longitude_bounds, *latitude_bounds)).all():
        raise ValueError("Geographic bounds must be finite and ordered")
    if (
        float(data.longitude.max()) < longitude_bounds[0]
        or float(data.longitude.min()) > longitude_bounds[1]
        or float(data.latitude.max()) < latitude_bounds[0]
        or float(data.latitude.min()) > latitude_bounds[1]
    ):
        raise ValueError("Dataset does not overlap the requested geographic domain")


def subset_geographic_bounds(
    data: xr.Dataset | xr.DataArray,
    *,
    longitude_bounds: tuple[float, float],
    latitude_bounds: tuple[float, float],
) -> xr.Dataset | xr.DataArray:
    """Normalize and inclusively subset explicit non-antimeridian bounds."""
    normalized = normalize_spatial_coordinates(data)
    validate_domain_overlap(normalized, longitude_bounds, latitude_bounds)
    subset = normalized.where(
        (normalized.longitude >= longitude_bounds[0])
        & (normalized.longitude <= longitude_bounds[1])
        & (normalized.latitude >= latitude_bounds[0])
        & (normalized.latitude <= latitude_bounds[1]),
        drop=True,
    )
    if subset.sizes.get("longitude", 0) == 0 or subset.sizes.get("latitude", 0) == 0:
        raise ValueError("Dataset has no grid-cell centres inside requested bounds")
    return subset


def resolve_domain_bounds(
    domain: str | Any, config: Mapping[str, Any] | None = None
) -> tuple[tuple[float, float], tuple[float, float]]:
    """Resolve bounds from an SSTDomainSpec or a configured domain ID."""
    if hasattr(domain, "longitude_bounds") and hasattr(domain, "latitude_bounds"):
        return tuple(domain.longitude_bounds), tuple(domain.latitude_bounds)
    if config is None:
        raise ValueError("config is required when resolving a domain ID")
    from src.sst_domains import load_sst_domain_specs

    registry = load_sst_domain_specs(config)
    try:
        spec = registry.domains[str(domain)]
    except KeyError as exc:
        raise ValueError(f"Unknown SST domain: {domain}") from exc
    return spec.longitude_bounds, spec.latitude_bounds


def _approximate_resolution(values: xr.DataArray) -> float | None:
    coordinate = np.asarray(values.values, dtype=float)
    if coordinate.size < 2:
        return None
    differences = np.diff(coordinate)
    finite = np.abs(differences[np.isfinite(differences)])
    return None if finite.size == 0 else float(np.median(finite))


def describe_sst_dataset(dataset: xr.Dataset) -> dict[str, Any]:
    """Describe a validated canonical SST dataset using JSON-safe values."""
    validate_dataset(dataset)
    latest = pd.to_datetime(dataset.time.values).max()
    return {
        "dimensions": {name: int(size) for name, size in dataset.sizes.items()},
        "longitude_bounds": [float(dataset.longitude.min()), float(dataset.longitude.max())],
        "latitude_bounds": [float(dataset.latitude.min()), float(dataset.latitude.max())],
        "longitude_resolution_degrees": _approximate_resolution(dataset.longitude),
        "latitude_resolution_degrees": _approximate_resolution(dataset.latitude),
        "latest_observation_time": pd.Timestamp(latest).isoformat(),
        "finite_value_count": int(np.isfinite(dataset.sst.values).sum()),
        "units": dataset.sst.attrs.get("units"),
    }


def load_sst_domain_dataset(
    path: str | Path,
    domain_spec: Any,
    *,
    aliases: Sequence[str] = DEFAULT_SST_ALIASES,
    source_mode: str,
) -> xr.Dataset:
    """Load, normalize, subset, validate, and annotate one configured SST domain."""
    source = Path(path)
    dataset = load_sst_dataset(source, aliases=aliases, create_demo_if_missing=False)
    longitude_bounds, latitude_bounds = resolve_domain_bounds(domain_spec)
    subset = subset_geographic_bounds(
        dataset,
        longitude_bounds=longitude_bounds,
        latitude_bounds=latitude_bounds,
    )
    assert isinstance(subset, xr.Dataset)
    validate_dataset(subset)
    description = describe_sst_dataset(subset)
    subset.attrs.update(
        domain_id=str(domain_spec.domain_id),
        geography_id=str(domain_spec.geography_id),
        source_path=str(source),
        source_mode=source_mode,
        source_product_family=(
            "synthetic_demo"
            if source_mode == "demo"
            else str(dataset.attrs.get("source_product_family", "ostia"))
        ),
        requested_longitude_bounds=",".join(map(str, longitude_bounds)),
        requested_latitude_bounds=",".join(map(str, latitude_bounds)),
        effective_longitude_bounds=",".join(map(str, description["longitude_bounds"])),
        effective_latitude_bounds=",".join(map(str, description["latitude_bounds"])),
        approximate_longitude_resolution_degrees=(
            description["longitude_resolution_degrees"] or "unknown"
        ),
        approximate_latitude_resolution_degrees=(
            description["latitude_resolution_degrees"] or "unknown"
        ),
        latest_observation_time=description["latest_observation_time"],
    )
    return subset.sortby("time")


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
    """Compatibility wrapper returning the canonical Niño 1+2 rectangle."""
    try:
        return subset_geographic_bounds(
            dataset,
            longitude_bounds=NINO12_LONGITUDE_BOUNDS,
            latitude_bounds=NINO12_LATITUDE_BOUNDS,
        )
    except ValueError as exc:
        if "overlap" in str(exc) or "grid-cell" in str(exc):
            raise ValueError("Dataset does not overlap the Niño 1+2 study region") from exc
        raise


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
