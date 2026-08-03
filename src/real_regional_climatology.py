"""Fine-grid regional Nino intermediates and experimental daily climatology."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from src.daily_climatology import build_daily_statistics
from src.geography import load_geography_registry


def regional_partial_series(
    dataset: xr.Dataset, config: Mapping[str, Any], *, tile_id: str, checkpoint_year: int,
    region_ids: Sequence[str] = ("nino34", "nino3", "nino12"),
) -> pd.DataFrame:
    """Return additive weighted sums from one fine-grid Pacific tile."""
    geography = load_geography_registry(config)
    rows: list[pd.DataFrame] = []
    for region_id in region_ids:
        region = geography.standard_regions[region_id]
        subset = dataset.sel(
            longitude=slice(region.bounds.west, region.bounds.east),
            latitude=slice(region.bounds.south, region.bounds.north),
        )
        if subset.sizes.get("latitude", 0) == 0 or subset.sizes.get("longitude", 0) == 0:
            continue
        field = subset.sst.transpose("time", "latitude", "longitude")
        latitude_resolution = float(np.median(np.diff(dataset.latitude.values)))
        longitude_resolution = float(np.median(np.diff(dataset.longitude.values)))
        expected_latitudes = region.bounds.south + latitude_resolution * (
            np.arange(round(region.bounds.height / latitude_resolution)) + 0.5
        )
        expected_longitudes = round(region.bounds.width / longitude_resolution)
        region_total_weight = float(np.cos(np.deg2rad(expected_latitudes)).sum() * expected_longitudes)
        region_total_cells = int(len(expected_latitudes) * expected_longitudes)
        weights = np.cos(np.deg2rad(subset.latitude.astype("float64"))).broadcast_like(
            field.isel(time=0, drop=True)
        )
        valid = field.notnull()
        weighted_sum = (field * weights).where(valid).sum(("latitude", "longitude"), skipna=True)
        valid_weight = weights.where(valid).sum(("latitude", "longitude"), skipna=True)
        total_weight = weights.sum(("latitude", "longitude"))
        frame = pd.DataFrame({
            "date": pd.DatetimeIndex(subset.time.values), "region_id": region_id,
            "weighted_sum": np.asarray(weighted_sum.values),
            "valid_weight": np.asarray(valid_weight.values), "total_weight": float(total_weight),
            "valid_cell_count": np.asarray(valid.sum(("latitude", "longitude")).values, dtype=np.int64),
            "covered_cell_count": int(field.sizes["latitude"] * field.sizes["longitude"]),
            "region_total_weight": region_total_weight,
            "region_total_cell_count": region_total_cells,
            "tile_id": tile_id, "checkpoint_year": checkpoint_year,
        })
        rows.append(frame)
    return pd.concat(rows, ignore_index=True) if rows else pd.DataFrame()


def merge_regional_partials(
    partials: pd.DataFrame, *, source_resolution: float, dataset_id: str,
) -> pd.DataFrame:
    columns = ["date", "region_id", "weighted_mean_sst_c", "valid_coverage",
               "valid_cell_count", "total_cell_count", "source_resolution",
               "source_dataset", "checkpoint_year"]
    if partials.empty:
        return pd.DataFrame(columns=columns)
    grouped = partials.groupby(["date", "region_id"], as_index=False).agg({
        "weighted_sum": "sum", "valid_weight": "sum", "total_weight": "sum",
        "valid_cell_count": "sum", "covered_cell_count": "sum",
        "region_total_weight": "max", "region_total_cell_count": "max",
        "checkpoint_year": "first",
    })
    grouped["weighted_mean_sst_c"] = grouped.weighted_sum / grouped.valid_weight
    grouped["valid_coverage"] = grouped.valid_weight / grouped.region_total_weight
    grouped["total_cell_count"] = grouped.region_total_cell_count
    grouped["source_resolution"] = source_resolution
    grouped["source_dataset"] = dataset_id
    return grouped[columns]


def build_regional_daily_climatology(series: pd.DataFrame) -> pd.DataFrame:
    """Apply the exact spatial-climatology definitions to compact regional series."""
    required = {"date", "region_id", "weighted_mean_sst_c", "valid_coverage"}
    if required - set(series):
        raise ValueError("Regional series contract is incomplete")
    frames: list[pd.DataFrame] = []
    for region_id, group in series.groupby("region_id"):
        ordered = group.sort_values("date")
        data = xr.DataArray(
            ordered.weighted_mean_sst_c.to_numpy(), dims=("time",),
            coords={"time": pd.DatetimeIndex(ordered.date.to_numpy())}, name="sst",
            attrs={"units": "degrees_Celsius"},
        )
        climate = build_daily_statistics(data, sampling_half_window_days=5, smoothing_window_days=31)
        frames.append(pd.DataFrame({
            "climatological_day": climate.climatological_day.values,
            "month": climate.month.values, "day": climate.day.values,
            "month_day": climate.month_day.values, "region_id": region_id,
            "climatology_mean_c": climate.climatology_mean.values,
            "climatology_median_c": climate.climatology_median.values,
            "climatology_std_c": climate.climatology_std.values,
            "threshold_p10_c": climate.threshold_p10.values,
            "threshold_p90_c": climate.threshold_p90.values,
            "observation_count": climate.observation_count.values,
            "valid_coverage": float(ordered.valid_coverage.mean()),
        }))
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()
