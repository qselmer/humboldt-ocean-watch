"""Temporal and spatial operational metrics for Niño 1+2."""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

from src.spatial_metrics import (
    area_weighted_mean,
    percentage_area_at_or_above_threshold,
    valid_data_coverage,
    warm_anomaly_centroid,
)

METRIC_COLUMNS = [
    "date", "mean_sst_c", "seven_day_mean_sst_c", "mean_sst_anomaly_c",
    "seven_day_mean_anomaly_c", "maximum_anomaly_c", "p90_anomaly_c",
    "mean_daily_sst_change_c", "seven_day_sst_change_c",
    "area_anomaly_ge_1c_percent", "area_anomaly_ge_2c_percent",
    "area_anomaly_ge_3c_percent", "area_zscore_ge_2_percent",
    "valid_data_coverage_percent", "warm_centroid_longitude", "warm_centroid_latitude",
]


def build_metrics_table(dataset: xr.Dataset, climatology: xr.Dataset | None) -> pd.DataFrame:
    """Build one cosine-weighted regional metrics row per available SST date."""
    sst = dataset.sst.sortby("time")
    mean_sst = area_weighted_mean(sst)
    daily_change = sst - sst.shift(time=1)
    seven_change = sst - sst.shift(time=7)
    result = pd.DataFrame({
        "date": pd.to_datetime(sst.time.values),
        "mean_sst_c": mean_sst.values,
        "seven_day_mean_sst_c": mean_sst.rolling(time=7, min_periods=1).mean().values,
        "mean_daily_sst_change_c": area_weighted_mean(daily_change).values,
        "seven_day_sst_change_c": area_weighted_mean(seven_change).values,
        "valid_data_coverage_percent": valid_data_coverage(sst).values,
    })
    anomaly_columns = [column for column in METRIC_COLUMNS if column not in result.columns and column != "date"]
    for column in anomaly_columns:
        result[column] = np.nan
    if climatology is not None:
        mean = climatology.climatological_mean.sel(month=sst.time.dt.month)
        std = climatology.climatological_standard_deviation.sel(month=sst.time.dt.month)
        anomaly = sst - mean
        zscore = xr.where(std != 0, anomaly / std, np.nan)
        mean_anomaly = area_weighted_mean(anomaly)
        lat_weights = np.cos(np.deg2rad(anomaly.latitude))
        centroid_lon, centroid_lat = warm_anomaly_centroid(anomaly, 2.0)
        result.update({
            "mean_sst_anomaly_c": mean_anomaly.values,
            "seven_day_mean_anomaly_c": mean_anomaly.rolling(time=7, min_periods=1).mean().values,
            "maximum_anomaly_c": anomaly.max(("latitude", "longitude"), skipna=True).values,
            "p90_anomaly_c": anomaly.weighted(lat_weights).quantile(0.9, dim=("latitude", "longitude"), skipna=True).values,
            "area_anomaly_ge_1c_percent": percentage_area_at_or_above_threshold(anomaly, 1.0).values,
            "area_anomaly_ge_2c_percent": percentage_area_at_or_above_threshold(anomaly, 2.0).values,
            "area_anomaly_ge_3c_percent": percentage_area_at_or_above_threshold(anomaly, 3.0).values,
            "area_zscore_ge_2_percent": percentage_area_at_or_above_threshold(zscore, 2.0).values,
            "warm_centroid_longitude": centroid_lon.values,
            "warm_centroid_latitude": centroid_lat.values,
        })
    return result[METRIC_COLUMNS]
