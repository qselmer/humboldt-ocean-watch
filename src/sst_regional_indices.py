"""Daily regional mean SST for configured Niño rectangles."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import pandas as pd
import xarray as xr

from src.data_loader import subset_geographic_bounds, validate_dataset
from src.geography import load_geography_registry


INDEX_COLUMNS = [
    "date",
    "region_id",
    "region_label",
    "mean_sst_c",
    "valid_coverage",
    "valid_cell_count",
    "total_cell_count",
    "source_domain",
    "source_mode",
    "climatological_mean_c",
    "anomaly_c",
    "standardized_anomaly",
    "threshold_p90_c",
    "exceeds_p90",
    "anomaly_status",
    "climatology_method",
]


def _resolution(values: xr.DataArray) -> float:
    coordinate = np.asarray(values.values, dtype=float)
    if coordinate.size < 2:
        return 0.0
    return float(np.median(np.abs(np.diff(coordinate))))


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
            f"Source domain does not completely cover configured region {region.id}"
        )


def calculate_nino_region_sst(
    dataset: xr.Dataset,
    config: Mapping[str, Any],
    *,
    region_ids: Sequence[str] | None = None,
    source_domain: str = "pacific_context",
    source_mode: str | None = None,
    minimum_valid_coverage: float | None = None,
    regional_climatology: pd.DataFrame | None = None,
) -> pd.DataFrame:
    """Calculate cosine-latitude-weighted daily SST means in long format.

    Anomalies are added only from a compatible regional climatology. No ONI or
    official ENSO classification is produced.
    """
    validate_dataset(dataset)
    geography = load_geography_registry(config)
    index_config = config.get("sst", {}).get("regional_indices", {})
    requested = tuple(region_ids or index_config.get("regions", ("nino34", "nino3", "nino12")))
    minimum = float(
        index_config.get("minimum_valid_coverage", 0.80)
        if minimum_valid_coverage is None
        else minimum_valid_coverage
    )
    if not 0.0 < minimum <= 1.0:
        raise ValueError("minimum_valid_coverage must be in (0, 1]")
    mode = source_mode or str(dataset.attrs.get("source_mode", "unknown"))
    climate_lookup: dict[tuple[str, int], pd.Series] = {}
    climate_error: str | None = None
    if regional_climatology is not None and not regional_climatology.empty:
        required_climate = {
            "region_id",
            "climatological_day",
            "climatology_mean_c",
            "climatology_std_c",
            "threshold_p90_c",
            "climatology_method",
            "source_product_family",
            "source_mode",
        }
        missing_climate = required_climate - set(regional_climatology.columns)
        if missing_climate:
            climate_error = "missing regional climatology columns: " + ", ".join(
                sorted(missing_climate)
            )
        elif regional_climatology.duplicated(["region_id", "climatological_day"]).any():
            climate_error = "regional climatology contains duplicate region/day rows"
        else:
            climate_lookup = {
                (str(row.region_id), int(row.climatological_day)): row
                for _, row in regional_climatology.iterrows()
            }
    rows: list[dict[str, Any]] = []
    for region_id in requested:
        try:
            region = geography.standard_regions[region_id]
        except KeyError as exc:
            raise ValueError(f"Unknown configured Niño region: {region_id}") from exc
        _require_complete_region(dataset, region)
        regional = subset_geographic_bounds(
            dataset,
            longitude_bounds=region.bounds.longitude,
            latitude_bounds=region.bounds.latitude,
        )
        assert isinstance(regional, xr.Dataset)
        field = regional.sst.transpose("time", "latitude", "longitude")
        weights = np.cos(np.deg2rad(regional.latitude)).broadcast_like(
            field.isel(time=0, drop=True)
        )
        total_weight = float(weights.sum())
        total_cells = int(regional.sizes["latitude"] * regional.sizes["longitude"])
        for index, time_value in enumerate(regional.time.values):
            daily = field.isel(time=index)
            valid = daily.notnull()
            valid_weight = float(weights.where(valid).sum(skipna=True))
            coverage = valid_weight / total_weight if total_weight > 0 else 0.0
            mean = (
                float((daily * weights).where(valid).sum(skipna=True) / valid_weight)
                if valid_weight > 0 and coverage >= minimum
                else np.nan
            )
            date = pd.Timestamp(str(time_value))
            climatological_day = pd.Timestamp(2000, date.month, date.day).dayofyear
            climate = climate_lookup.get((region.id, climatological_day))
            climate_mean = np.nan
            anomaly = np.nan
            standardized = np.nan
            threshold_p90 = np.nan
            exceeds_p90: bool | None = None
            method = "not_available"
            anomaly_status = "not_calculated_no_compatible_climatology"
            if climate_error is not None:
                anomaly_status = "not_calculated_incompatible_climatology"
            elif climate is not None:
                climate_mode = str(climate.source_mode)
                synthetic = climate_mode in {
                    "demo",
                    "synthetic",
                    "synthetic_demonstration",
                } or str(climate.source_product_family) == "synthetic_demo"
                if mode in {"live", "Copernicus cached data"} and synthetic:
                    anomaly_status = "not_calculated_incompatible_climatology"
                elif np.isfinite(mean) and np.isfinite(climate.climatology_mean_c):
                    climate_mean = float(climate.climatology_mean_c)
                    anomaly = float(mean - climate_mean)
                    method = str(climate.climatology_method)
                    if np.isfinite(climate.climatology_std_c) and float(climate.climatology_std_c) > 0:
                        standardized = anomaly / float(climate.climatology_std_c)
                    if np.isfinite(climate.threshold_p90_c):
                        threshold_p90 = float(climate.threshold_p90_c)
                        exceeds_p90 = bool(mean > threshold_p90)
                    anomaly_status = (
                        "calculated"
                        if np.isfinite(standardized) and exceeds_p90 is not None
                        else "partial"
                    )
            rows.append(
                {
                    "date": date,
                    "region_id": region.id,
                    "region_label": region.label,
                    "mean_sst_c": mean,
                    "valid_coverage": coverage,
                    "valid_cell_count": int(np.isfinite(daily.values).sum()),
                    "total_cell_count": total_cells,
                    "source_domain": source_domain,
                    "source_mode": mode,
                    "climatological_mean_c": climate_mean,
                    "anomaly_c": anomaly,
                    "standardized_anomaly": standardized,
                    "threshold_p90_c": threshold_p90,
                    "exceeds_p90": exceeds_p90,
                    "anomaly_status": anomaly_status,
                    "climatology_method": method,
                }
            )
    return pd.DataFrame(rows, columns=INDEX_COLUMNS).sort_values(
        ["date", "region_id"], ignore_index=True
    )
