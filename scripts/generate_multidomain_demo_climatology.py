"""Generate deterministic synthetic 366-day climatologies without network access."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import dask.array as da
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.daily_climatology import calendar_coordinates
from src.sst_climatology_spec import SSTClimatologySpec, load_sst_climatology_specs
from src.sst_domains import load_sst_domain_specs
from src.utils import load_config, resolve_project_path


DEMO_DOMAINS = ("pacific_context", "humboldt_coastal")


def _coordinates(bounds: tuple[float, float], resolution: float) -> np.ndarray:
    count = int(round((bounds[1] - bounds[0]) / resolution)) + 1
    return np.linspace(bounds[0], bounds[1], count, dtype=np.float64)


def build_demo_climatology(
    climate_spec: SSTClimatologySpec,
    domain_spec,
    *,
    seed: int,
) -> xr.Dataset:
    """Return a compact, lazy synthetic climatology on the configured grid."""
    rng = np.random.default_rng(seed)
    resolution = climate_spec.target_resolution_degrees
    longitude = _coordinates(domain_spec.longitude_bounds, resolution)
    latitude = _coordinates(domain_spec.latitude_bounds, resolution)
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    phase_lon, phase_lat = rng.uniform(0.0, 2.0 * np.pi, size=2)
    amplitude = rng.uniform(0.20, 0.35)
    spatial = (
        27.4
        + 0.285 * lat_grid
        + 0.012 * (lon_grid - domain_spec.longitude_bounds[0])
        + amplitude * np.sin(np.deg2rad(lon_grid * 2.0) + phase_lon)
        + 0.18 * np.cos(np.deg2rad(lat_grid * 4.0) + phase_lat)
    ).astype(np.float32)
    day = np.arange(1, 367, dtype=np.float32)
    seasonal = (
        0.05 + 0.30 * np.sin(2.0 * np.pi * (day - 1.0) / 366.0)
    ).astype(np.float32)
    lat_chunk = min(128, len(latitude))
    lon_chunk = min(128, len(longitude))
    spatial_lazy = da.from_array(spatial, chunks=(lat_chunk, lon_chunk))[None, :, :]
    seasonal_lazy = da.from_array(seasonal, chunks=(8,))[:, None, None]
    mean = spatial_lazy + seasonal_lazy
    median = mean + np.float32(0.03) * da.sin(
        da.from_array(day, chunks=(8,))[:, None, None] * np.float32(2.0 * np.pi / 366.0)
    )
    variability = (
        np.float32(0.55)
        + np.float32(0.08) * da.cos(
            da.from_array(day, chunks=(8,))[:, None, None] * np.float32(2.0 * np.pi / 366.0)
        )
        + da.from_array(
            (0.03 * np.cos(np.deg2rad(lat_grid))).astype(np.float32),
            chunks=(lat_chunk, lon_chunk),
        )[None, :, :]
    )
    p10 = mean - np.float32(1.2815516) * variability
    p90 = mean + np.float32(1.2815516) * variability
    count = da.full(
        (366, len(latitude), len(longitude)),
        30,
        dtype=np.int16,
        chunks=(8, lat_chunk, lon_chunk),
    )
    dims = ("climatological_day", "latitude", "longitude")
    dataset = xr.Dataset(
        {
            "climatology_mean": (dims, mean),
            "climatology_median": (dims, median),
            "climatology_std": (dims, variability),
            "threshold_p10": (dims, p10),
            "threshold_p90": (dims, p90),
            "observation_count": (dims, count),
        },
        coords={
            **calendar_coordinates(),
            "latitude": latitude,
            "longitude": longitude,
        },
        attrs={
            "title": f"Synthetic demonstration climatology — {domain_spec.label}",
            "domain_id": climate_spec.domain_id,
            "geography_id": climate_spec.geography_id,
            "climatology_method": "daily_smoothed",
            "climatology_source_mode": "synthetic_demonstration",
            "source_product_family": "synthetic_demo",
            "reference_period": "synthetic-demonstration",
            "nominal_configuration_reference_period": (
                f"{climate_spec.reference_start}-{climate_spec.reference_end}; not observed"
            ),
            "calendar": "stable_366_gregorian",
            "calendar_mapping": "stable month-day mapping on canonical leap year 2000",
            "leap_day_method": climate_spec.leap_day_method,
            "sampling_half_window_days": climate_spec.sampling_half_window_days,
            "smoothing_window_days": climate_spec.smoothing_window_days,
            "scientific_use": "demonstration_only",
            "represents_observed_climatology": "false",
            "network_access": "false",
            "seed": int(seed),
        },
    )
    for name in (
        "climatology_mean",
        "climatology_median",
        "climatology_std",
        "threshold_p10",
        "threshold_p90",
    ):
        dataset[name].attrs.update(units="degrees_Celsius")
    dataset.observation_count.attrs.update(
        units="1",
        long_name="synthetic support count for software demonstration; not observations",
    )
    return dataset


def write_demo_climatology(
    dataset: xr.Dataset, output: Path, *, overwrite: bool
) -> Path:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Demo climatology exists (use --overwrite): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = output.with_name(f".{output.stem}.tmp.nc")
    chunks = (
        8,
        min(128, dataset.sizes["latitude"]),
        min(128, dataset.sizes["longitude"]),
    )
    encoding = {
        name: {
            "zlib": True,
            "complevel": 4,
            "shuffle": True,
            "chunksizes": chunks,
            "dtype": "float32",
        }
        for name in (
            "climatology_mean",
            "climatology_median",
            "climatology_std",
            "threshold_p10",
            "threshold_p90",
        )
    }
    encoding["observation_count"] = {
        "zlib": True,
        "complevel": 4,
        "shuffle": True,
        "chunksizes": chunks,
        "dtype": "int16",
    }
    try:
        dataset.to_netcdf(temporary, engine="netcdf4", encoding=encoding)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate local synthetic climatologies; no observations or network are used."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--domain", choices=DEMO_DOMAINS)
    selection.add_argument("--all", action="store_true")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    config = load_config()
    climate_specs = load_sst_climatology_specs(config)
    domain_specs = load_sst_domain_specs(config).domains
    domains = DEMO_DOMAINS if args.all else (args.domain,)
    outputs: list[Path] = []
    for offset, domain_id in enumerate(domains):
        spec = climate_specs[domain_id]
        if spec.demo_path is None:
            raise ValueError(f"No demo climatology path is configured for {domain_id}")
        output = (
            args.output_directory / spec.demo_path.name
            if args.output_directory is not None
            else resolve_project_path(spec.demo_path)
        )
        dataset = build_demo_climatology(
            spec, domain_specs[domain_id], seed=args.seed + offset
        )
        outputs.append(write_demo_climatology(dataset, output, overwrite=args.overwrite))
        print(f"{domain_id}: {output}")
    return outputs


if __name__ == "__main__":
    main()
