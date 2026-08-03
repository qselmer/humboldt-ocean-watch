"""Immutable, deterministic resource plans for real SST climatologies."""

from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

from src.climatology_tiles import TileGrid, build_spatial_tiles, coordinate_count, validate_tile_coverage
from src.geography import load_geography_registry
from src.sst_domains import load_sst_domain_specs

SECONDS_PER_DAY = 86_400
DAYS_PER_YEAR_ESTIMATE = 365.25
FLOAT_BYTES = 4


@dataclass(frozen=True)
class StorageEstimate:
    source_download_bytes: int
    checkpoint_bytes: int
    final_climatology_bytes: int
    regional_products_bytes: int
    temporary_bytes: int
    peak_memory_bytes: int
    total_disk_bytes: int


@dataclass(frozen=True)
class TilePlan:
    tile_id: str
    bounds: tuple[float, float, float, float]
    target_cells: int
    estimated_source_cells: int
    estimated_peak_memory_bytes: int


@dataclass(frozen=True)
class DomainBuildPlan:
    domain_id: str
    bounds: tuple[float, float, float, float]
    source_resolution_degrees: float
    target_resolution_degrees: float
    start_year: int
    end_year: int
    number_of_years: int
    estimated_source_cells: int
    estimated_target_cells: int
    estimated_daily_observations: int
    tile_dimensions_strategy: str
    number_of_tiles: int
    checkpoint_strategy: str
    aggregation_strategy: str
    percentile_strategy: str
    output_path: str
    pilot_output_path: str
    resume_path: str
    tiles: tuple[TilePlan, ...]
    storage: StorageEstimate


@dataclass(frozen=True)
class ClimatologyBuildPlan:
    status: str
    dataset_id: str
    source_variable: str
    start_year: int
    end_year: int
    pilot: bool
    domains: tuple[DomainBuildPlan, ...]
    maximum_estimated_disk_bytes: int
    maximum_estimated_memory_bytes: int
    warnings: tuple[str, ...]

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _domain_plan(
    config: Mapping[str, Any], domain_id: str, start_year: int, end_year: int,
    *, pilot: bool,
) -> DomainBuildPlan:
    build = config["real_climatology_build"]
    domain = load_sst_domain_specs(config).domains[domain_id]
    source_resolution = float(build["source_resolution_degrees"])
    target_resolution = domain.target_resolution_degrees
    years = end_year - start_year + 1
    if years < 1:
        raise ValueError("start_year must not exceed end_year")
    tile_target_cells = int(
        build["pilot"].get("tile_target_cells", build["tile_target_cells"])
        if pilot else build["tile_target_cells"]
    )
    target_tiles = build_spatial_tiles(domain.bounds, target_resolution, tile_target_cells)
    validate_tile_coverage(target_tiles, domain.bounds, target_resolution)
    if pilot:
        if domain_id == "pacific_context":
            regions = load_geography_registry(config).standard_regions.values()
            def score(tile: TileGrid) -> float:
                return sum(
                    max(0.0, min(tile.bounds.east, region.bounds.east) - max(tile.bounds.west, region.bounds.west))
                    * max(0.0, min(tile.bounds.north, region.bounds.north) - max(tile.bounds.south, region.bounds.south))
                    for region in regions
                )
            target_tiles = tuple(sorted(target_tiles, key=lambda item: (-score(item), item.tile_id)))
        target_tiles = target_tiles[: int(build["pilot"]["maximum_tiles"])]
    source_nlat = coordinate_count(domain.bounds.south, domain.bounds.north, source_resolution)
    source_nlon = coordinate_count(domain.bounds.west, domain.bounds.east, source_resolution)
    target_nlat = coordinate_count(domain.bounds.south, domain.bounds.north, target_resolution)
    target_nlon = coordinate_count(domain.bounds.west, domain.bounds.east, target_resolution)
    source_cells = source_nlat * source_nlon
    target_cells = target_nlat * target_nlon
    days = round(years * DAYS_PER_YEAR_ESTIMATE)
    tile_plans: list[TilePlan] = []
    factor = int(round(target_resolution / source_resolution))
    for tile in target_tiles:
        source_west = domain.bounds.west + tile.longitude_start * factor * source_resolution
        source_east = min(
            domain.bounds.east,
            domain.bounds.west + (tile.longitude_stop * factor - 1) * source_resolution,
        )
        source_south = domain.bounds.south + tile.latitude_start * factor * source_resolution
        source_north = min(
            domain.bounds.north,
            domain.bounds.south + (tile.latitude_stop * factor - 1) * source_resolution,
        )
        tile_source_cells = (
            coordinate_count(source_west, source_east, source_resolution)
            * coordinate_count(source_south, source_north, source_resolution)
        )
        # Raw pooled values plus quantile/smoothing work arrays. Exact quantiles
        # remain feasible by shrinking spatial tiles, never by approximation.
        graph_overhead = int(float(build.get("exact_quantile_graph_overhead_gb", 0.0)) * 1024**3)
        peak = int(
            tile.target_cells * days * FLOAT_BYTES * 1.75
            + tile.target_cells * 366 * 32
            + graph_overhead
        )
        tile_plans.append(TilePlan(
            tile_id=tile.tile_id,
            bounds=(source_west, source_east, source_south, source_north),
            target_cells=tile.target_cells,
            estimated_source_cells=tile_source_cells,
            estimated_peak_memory_bytes=peak,
        ))
    selected_target_cells = sum(tile.target_cells for tile in target_tiles)
    selected_source_cells = sum(tile.estimated_source_cells for tile in tile_plans)
    daily_checkpoint = selected_target_cells * days * FLOAT_BYTES
    source_download = selected_source_cells * days * FLOAT_BYTES
    final = selected_target_cells * 366 * (5 * FLOAT_BYTES + 8)
    regional = years * 366 * 3 * 64 if domain_id == "pacific_context" else 0
    temporary = max((tile.estimated_source_cells for tile in tile_plans), default=0) * 366 * FLOAT_BYTES
    if pilot and domain_id == "pacific_context":
        # The single map tile cannot cover all three complete Nino rectangles.
        # Small, separate fine-grid regional checkpoints validate those compact
        # intermediates without increasing the climatology tile count.
        geography = load_geography_registry(config)
        regional_cells = sum(
            round(region.bounds.width / source_resolution)
            * round(region.bounds.height / source_resolution)
            for region in geography.standard_regions.values()
        )
        largest_region_cells = max(
            round(region.bounds.width / source_resolution)
            * round(region.bounds.height / source_resolution)
            for region in geography.standard_regions.values()
        )
        source_download += regional_cells * days * FLOAT_BYTES
        temporary = max(temporary, largest_region_cells * 366 * FLOAT_BYTES)
    peak = max((tile.estimated_peak_memory_bytes for tile in tile_plans), default=0)
    storage = StorageEstimate(
        source_download_bytes=source_download,
        checkpoint_bytes=daily_checkpoint,
        final_climatology_bytes=final,
        regional_products_bytes=regional,
        temporary_bytes=temporary,
        peak_memory_bytes=peak,
        # Downloads are processed one tile-year at a time. Cumulative transfer
        # volume is reported separately and is not simultaneous disk residency.
        total_disk_bytes=daily_checkpoint + final + regional + temporary,
    )
    output_path = str(domain.climatology_daily_path)
    pilot_output = f"data/climatology/pilot/{domain_id}_daily_1991_1992_pilot.nc"
    root = Path(build["pilot_work_directory"] if pilot else build["work_directory"])
    aggregation = (
        "cosine_latitude_weighted_block_mean_0.05_to_0.25"
        if target_resolution > source_resolution else "preserve_native_0.05"
    )
    return DomainBuildPlan(
        domain_id=domain_id,
        bounds=(domain.bounds.west, domain.bounds.east, domain.bounds.south, domain.bounds.north),
        source_resolution_degrees=source_resolution,
        target_resolution_degrees=target_resolution,
        start_year=start_year, end_year=end_year, number_of_years=years,
        estimated_source_cells=source_cells, estimated_target_cells=target_cells,
        estimated_daily_observations=selected_target_cells * days,
        tile_dimensions_strategy=f"target_cells<={tile_target_cells}",
        number_of_tiles=len(target_tiles), checkpoint_strategy="yearly_spatial_tile_netcdf",
        aggregation_strategy=aggregation, percentile_strategy="exact_tile_wise_full_time_chunk",
        output_path=output_path, pilot_output_path=pilot_output,
        resume_path=str(root / domain_id), tiles=tuple(tile_plans), storage=storage,
    )


def build_climatology_plan(
    config: Mapping[str, Any], domain_ids: Sequence[str],
    start_year: int | None = None, end_year: int | None = None, *, pilot: bool = False,
) -> ClimatologyBuildPlan:
    build = config["real_climatology_build"]
    if pilot:
        start_year = int(build["pilot"]["start_year"])
        end_year = int(build["pilot"]["end_year"])
    start = int(build["start_year"] if start_year is None else start_year)
    end = int(build["end_year"] if end_year is None else end_year)
    allowed = {"pacific_context", "humboldt_coastal"}
    if not domain_ids or not set(domain_ids) <= allowed:
        raise ValueError("Domains must be pacific_context and/or humboldt_coastal")
    domains = tuple(_domain_plan(config, item, start, end, pilot=pilot) for item in domain_ids)
    disk_limit = int(float(build["maximum_estimated_disk_gb"]) * 1024**3)
    memory_limit = int(float(build["maximum_estimated_memory_gb"]) * 1024**3)
    disk = sum(item.storage.total_disk_bytes for item in domains)
    memory = max(item.storage.peak_memory_bytes for item in domains)
    warnings: list[str] = []
    if disk > disk_limit:
        warnings.append("Estimated disk use exceeds configured limit; reduce scope only with explicit authorization")
    if memory > memory_limit:
        warnings.append("Estimated peak memory exceeds configured limit; reduce tile_target_cells")
    return ClimatologyBuildPlan(
        status="valid" if not warnings else "resource_limit_exceeded",
        dataset_id=str(build["dataset_id"]), source_variable=str(build["source_variable"]),
        start_year=start, end_year=end, pilot=pilot, domains=domains,
        maximum_estimated_disk_bytes=disk_limit, maximum_estimated_memory_bytes=memory_limit,
        warnings=tuple(warnings),
    )
