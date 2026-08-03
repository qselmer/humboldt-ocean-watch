from copy import deepcopy

import pytest

from src.climatology_build_plan import build_climatology_plan
from src.climatology_tiles import build_spatial_tiles, validate_tile_coverage
from src.geography import GeographicBounds
from src.utils import load_config


def test_tiles_are_deterministic_complete_and_non_overlapping() -> None:
    bounds = GeographicBounds(-10.0, 0.0, -5.0, 5.0)
    first = build_spatial_tiles(bounds, 0.25, 500)
    second = build_spatial_tiles(bounds, 0.25, 500)
    assert first == second
    assert len({tile.tile_id for tile in first}) == len(first)
    validate_tile_coverage(first, bounds, 0.25)
    assert all(tile.target_cells <= 500 for tile in first)


def test_full_plan_is_deterministic_and_within_configured_limits() -> None:
    config = load_config()
    first = build_climatology_plan(config, ("pacific_context", "humboldt_coastal"))
    second = build_climatology_plan(config, ("pacific_context", "humboldt_coastal"))
    assert first == second
    assert first.status == "valid"
    assert (first.start_year, first.end_year) == (1991, 2020)
    domains = {item.domain_id: item for item in first.domains}
    assert domains["pacific_context"].aggregation_strategy.startswith("cosine")
    assert domains["humboldt_coastal"].aggregation_strategy == "preserve_native_0.05"
    assert domains["pacific_context"].number_of_tiles == 6
    assert domains["humboldt_coastal"].number_of_tiles == 12
    assert max(item.storage.peak_memory_bytes for item in first.domains) < 8 * 1024**3


def test_pilot_is_two_years_one_smaller_tile_and_separate_paths() -> None:
    config = load_config()
    plan = build_climatology_plan(
        config, ("pacific_context", "humboldt_coastal"), pilot=True
    )
    assert (plan.start_year, plan.end_year) == (1991, 1992)
    assert all(item.number_of_tiles == 1 for item in plan.domains)
    assert all("pilot" in item.pilot_output_path for item in plan.domains)
    assert all(item.pilot_output_path != item.output_path for item in plan.domains)
    assert max(item.storage.peak_memory_bytes for item in plan.domains) < plan.maximum_estimated_memory_bytes


def test_resource_limits_reject_before_download() -> None:
    config = deepcopy(load_config())
    config["real_climatology_build"]["maximum_estimated_disk_gb"] = 0.000001
    plan = build_climatology_plan(config, ("pacific_context",))
    assert plan.status == "resource_limit_exceeded"
    assert "disk" in plan.warnings[0].lower()


def test_invalid_domains_and_grid_alignment_are_rejected() -> None:
    with pytest.raises(ValueError, match="Domains"):
        build_climatology_plan(load_config(), ("nino12",))
    with pytest.raises(ValueError, match="align"):
        build_spatial_tiles(GeographicBounds(-10.0, 0.1, -5.0, 5.0), 0.25, 100)
