"""Explicitly generate deterministic local SST demonstration files."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.sst_domains import SSTDomainSpec, load_sst_domain_specs
from src.utils import load_config, resolve_project_path


DEMO_DOMAINS = ("pacific_context", "humboldt_coastal")


def _coordinates(bounds: tuple[float, float], resolution: float) -> np.ndarray:
    count = int(round((bounds[1] - bounds[0]) / resolution)) + 1
    return np.linspace(bounds[0], bounds[1], count, dtype=np.float64)


def build_synthetic_domain_dataset(
    spec: SSTDomainSpec, *, days: int, seed: int
) -> xr.Dataset:
    """Create a continuous mathematical demo field in source Kelvin units."""
    if days < 2:
        raise ValueError("Multidomain demonstration data requires at least two days")
    rng = np.random.default_rng(seed)
    longitude = _coordinates(spec.longitude_bounds, spec.target_resolution_degrees)
    latitude = _coordinates(spec.latitude_bounds, spec.target_resolution_degrees)
    lon_grid, lat_grid = np.meshgrid(longitude, latitude)
    phase_lon, phase_lat = rng.uniform(0.0, 2.0 * np.pi, size=2)
    amplitude = rng.uniform(0.20, 0.35)
    baseline_c = (
        27.4
        + 0.285 * lat_grid
        + 0.012 * (lon_grid - spec.longitude_bounds[0])
        + amplitude * np.sin(np.deg2rad(lon_grid * 2.0) + phase_lon)
        + 0.18 * np.cos(np.deg2rad(lat_grid * 4.0) + phase_lat)
    )
    values = np.stack(
        [
            baseline_c
            + 0.12 * np.sin(2.0 * np.pi * day / max(days, 3))
            + 0.025 * day
            for day in range(days)
        ]
    ).astype(np.float32)
    dataset = xr.Dataset(
        {
            "analysed_sst": (
                ("time", "latitude", "longitude"),
                values + np.float32(273.15),
            )
        },
        coords={
            "time": pd.date_range("2024-01-01", periods=days, freq="D"),
            "latitude": latitude,
            "longitude": longitude,
        },
        attrs={
            "title": f"Synthetic demonstration SST — {spec.label}",
            "product_status": "synthetic demonstration data",
            "scientific_representation": "mathematical demonstration only",
            "represents_current_conditions": "false",
            "domain_id": spec.domain_id,
            "seed": int(seed),
        },
    )
    dataset.analysed_sst.attrs.update(
        units="kelvin",
        standard_name="sea_surface_temperature",
        long_name="synthetic analysed sea surface temperature",
    )
    return dataset


def write_demo_dataset(
    spec: SSTDomainSpec,
    output: Path,
    *,
    days: int,
    seed: int,
    overwrite: bool,
) -> Path:
    if output.exists() and not overwrite:
        raise FileExistsError(f"Demo file already exists (use --overwrite): {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    dataset = build_synthetic_domain_dataset(spec, days=days, seed=seed)
    temporary = output.with_name(f".{output.stem}.tmp.nc")
    try:
        dataset.to_netcdf(temporary)
        temporary.replace(output)
    finally:
        if temporary.exists():
            temporary.unlink()
    return output


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate local synthetic SST demos; no network access is used."
    )
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--domain", choices=DEMO_DOMAINS)
    selection.add_argument("--all", action="store_true", help="Generate both contextual domains")
    parser.add_argument("--days", type=int, default=5, help="Consecutive demo days (default: 5)")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument(
        "--output-directory",
        type=Path,
        help="Override configured directory while retaining configured file names",
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> list[Path]:
    args = parse_args(argv)
    config = load_config()
    specs = load_sst_domain_specs(config).domains
    domains = DEMO_DOMAINS if args.all else (args.domain,)
    written: list[Path] = []
    for offset, domain_id in enumerate(domains):
        spec = specs[domain_id]
        output = (
            args.output_directory / spec.demo_path.name
            if args.output_directory is not None
            else resolve_project_path(spec.demo_path)
        )
        path = write_demo_dataset(
            spec,
            output,
            days=args.days,
            seed=args.seed + offset,
            overwrite=args.overwrite,
        )
        print(f"{domain_id}: {path}")
        written.append(path)
    return written


if __name__ == "__main__":
    main()
