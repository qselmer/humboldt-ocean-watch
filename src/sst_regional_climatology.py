"""Regional Niño climatologies derived from compatible Pacific spatial fields."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from src.geography import load_geography_registry


REGIONAL_CLIMATOLOGY_COLUMNS = [
    "climatological_day",
    "month",
    "day",
    "month_day",
    "region_id",
    "region_label",
    "climatology_mean_c",
    "climatology_std_c",
    "threshold_p10_c",
    "threshold_p90_c",
    "valid_coverage",
    "source_domain",
    "climatology_method",
    "reference_start",
    "reference_end",
    "source_product_family",
    "source_mode",
]


def _resolution(values: xr.DataArray) -> float:
    raw = np.asarray(values.values, dtype=float)
    return 0.0 if raw.size < 2 else float(np.median(np.abs(np.diff(raw))))


def _require_complete_region(dataset: xr.Dataset, region: Any) -> None:
    lon_tolerance = _resolution(dataset.longitude) / 2.0 + 1.0e-9
    lat_tolerance = _resolution(dataset.latitude) / 2.0 + 1.0e-9
    if (
        float(dataset.longitude.min()) > region.bounds.west + lon_tolerance
        or float(dataset.longitude.max()) < region.bounds.east - lon_tolerance
        or float(dataset.latitude.min()) > region.bounds.south + lat_tolerance
        or float(dataset.latitude.max()) < region.bounds.north - lat_tolerance
    ):
        raise ValueError(
            f"Source climatology does not completely cover configured region {region.id}"
        )


def _weighted(field: xr.DataArray, weights: xr.DataArray) -> tuple[xr.DataArray, xr.DataArray]:
    valid = field.notnull()
    valid_weight = weights.where(valid).sum(("latitude", "longitude"), skipna=True)
    total_weight = weights.sum(("latitude", "longitude"), skipna=True)
    coverage = (valid_weight / total_weight).rename("valid_coverage")
    mean = ((field * weights).where(valid).sum(("latitude", "longitude"), skipna=True) / valid_weight)
    return mean, coverage


def build_nino_regional_climatology(
    climatology: xr.Dataset,
    config: Mapping[str, Any],
    *,
    region_ids: Sequence[str] | None = None,
    source_domain: str = "pacific_context",
    minimum_valid_coverage: float | None = None,
) -> pd.DataFrame:
    """Build daily, long-format regional climatology without recoding bounds."""
    if "climatological_day" not in climatology.dims:
        raise ValueError("Regional climatology requires a stable 366-day spatial climatology")
    if climatology.sizes["climatological_day"] != 366:
        raise ValueError("Regional climatology requires exactly 366 climatological days")
    required = {"climatology_mean", "latitude", "longitude"}
    missing = required - set(climatology.variables)
    if missing:
        raise ValueError(f"Spatial climatology is missing: {sorted(missing)}")
    geography = load_geography_registry(config)
    index_config = config.get("sst", {}).get("regional_indices", {})
    requested = tuple(
        region_ids or index_config.get("regions", ("nino34", "nino3", "nino12"))
    )
    minimum = float(
        index_config.get("minimum_valid_coverage", 0.80)
        if minimum_valid_coverage is None
        else minimum_valid_coverage
    )
    if not 0.0 < minimum <= 1.0:
        raise ValueError("minimum_valid_coverage must be in (0, 1]")
    source = climatology.sortby("latitude").sortby("longitude")
    rows: list[pd.DataFrame] = []
    for region_id in requested:
        if region_id not in geography.standard_regions:
            raise ValueError(f"Unknown configured Niño region: {region_id}")
        region = geography.standard_regions[region_id]
        _require_complete_region(source, region)
        selected = source.sel(
            longitude=slice(region.bounds.west, region.bounds.east),
            latitude=slice(region.bounds.south, region.bounds.north),
        )
        weights = np.cos(np.deg2rad(selected.latitude)).broadcast_like(
            selected.climatology_mean.isel(climatological_day=0, drop=True)
        )
        mean, coverage = _weighted(selected.climatology_mean, weights)
        values: dict[str, Any] = {
            "climatological_day": selected.climatological_day.values.astype(int),
            "month": selected.month.values.astype(int),
            "day": selected.day.values.astype(int),
            "month_day": selected.month_day.astype(str).values,
            "region_id": region.id,
            "region_label": region.label,
            "climatology_mean_c": mean.where(coverage >= minimum).values,
            "valid_coverage": coverage.values,
        }
        for source_name, target_name in (
            ("climatology_std", "climatology_std_c"),
            ("threshold_p10", "threshold_p10_c"),
            ("threshold_p90", "threshold_p90_c"),
        ):
            if source_name in selected:
                weighted, variable_coverage = _weighted(selected[source_name], weights)
                values[target_name] = weighted.where(variable_coverage >= minimum).values
            else:
                values[target_name] = np.full(366, np.nan)
        period = str(source.attrs.get("reference_period", "-")).replace("–", "-").split("-")
        try:
            reference_start, reference_end = int(period[0]), int(period[-1])
        except (TypeError, ValueError):
            reference_start = reference_end = None
        values.update(
            source_domain=source_domain,
            climatology_method=str(source.attrs.get("climatology_method", "daily_smoothed")),
            reference_start=reference_start,
            reference_end=reference_end,
            source_product_family=str(source.attrs.get("source_product_family", "unknown")),
            source_mode=str(source.attrs.get("climatology_source_mode", "unknown")),
        )
        rows.append(pd.DataFrame(values))
    return pd.concat(rows, ignore_index=True)[REGIONAL_CLIMATOLOGY_COLUMNS].sort_values(
        ["climatological_day", "region_id"], ignore_index=True
    )
