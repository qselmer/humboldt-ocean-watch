"""Date-specific thermal diagnosis for Niño 1+2."""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
import xarray as xr

from src.anomalies import calculate_anomaly, calculate_z_score
from src.data_loader import SYNTHETIC_DEMO_MODE, subset_nino12
from src.spatial_metrics import (
    anomaly_persistence,
    anomaly_profiles,
    area_weighted_mean,
    percentage_area_at_or_above_threshold,
    valid_data_coverage,
    warm_anomaly_centroid,
)


def available_dates(dataset: xr.Dataset) -> list[str]:
    """Return ISO dates strictly from the dataset time coordinate."""
    return [pd.Timestamp(value).strftime("%Y-%m-%d") for value in dataset.time.values]


def load_climatology(path: str) -> xr.Dataset:
    """Load and validate an operational monthly climatology."""
    with xr.open_dataset(path) as opened:
        climatology = opened.load()
    required = {
        "climatological_mean",
        "climatological_standard_deviation",
        "valid_observation_count",
    }
    missing = required - set(climatology.data_vars)
    if missing:
        raise ValueError(f"Climatology is missing variables: {sorted(missing)}")
    return climatology


def diagnose_date(
    dataset: xr.Dataset,
    analysis_date: str,
    *,
    climatology: xr.Dataset | None = None,
    data_mode: str | None = None,
    threshold: float = 2.0,
    persistence_window: int = 7,
) -> tuple[xr.Dataset, dict[str, Any]]:
    """Diagnose any exact date available in the SST time coordinate."""
    regional = subset_nino12(dataset).sortby("time")
    assert isinstance(regional, xr.Dataset)
    dates = available_dates(regional)
    if analysis_date not in dates:
        raise ValueError(
            f"Analysis date {analysis_date} is unavailable; choose a date present in the dataset"
        )
    position = dates.index(analysis_date)
    current = regional.sst.isel(time=position, drop=True).rename("sst_current")
    daily_change = xr.full_like(current, np.nan).rename("sst_daily_change")
    if position >= 1:
        daily_change = (current - regional.sst.isel(time=position - 1, drop=True)).rename(
            "sst_daily_change"
        )
    seven_day_change = xr.full_like(current, np.nan).rename("sst_seven_day_change")
    if position >= 7:
        seven_day_change = (current - regional.sst.isel(time=position - 7, drop=True)).rename(
            "sst_seven_day_change"
        )
    for field in (daily_change, seven_day_change):
        field.attrs["units"] = "degrees_Celsius"

    mode = data_mode or str(regional.attrs.get("data_mode", SYNTHETIC_DEMO_MODE))
    fields = xr.Dataset(
        {"sst_current": current, "sst_daily_change": daily_change, "sst_seven_day_change": seven_day_change},
        attrs={"data_mode": mode, "analysis_date": analysis_date},
    )
    metrics: dict[str, float | None] = {
        "mean_sst_c": float(area_weighted_mean(current)),
        "mean_sst_anomaly_c": None,
        "maximum_anomaly_c": None,
        "p90_anomaly_c": None,
        "mean_daily_change_c": float(area_weighted_mean(daily_change)),
        "seven_day_change_c": float(area_weighted_mean(seven_day_change)),
        "area_anomaly_ge_threshold_percent": None,
        "valid_data_coverage_percent": float(valid_data_coverage(current)),
        "warm_centroid_longitude": None,
        "warm_centroid_latitude": None,
    }
    warning = None
    if climatology is None:
        warning = (
            "The 1991–2020 climatology file is unavailable. Anomaly, z-score, "
            "persistence, and anomaly spatial products are disabled."
        )
    else:
        month = int(pd.Timestamp(analysis_date).month)
        mean = climatology.climatological_mean.sel(month=month, drop=True)
        std = climatology.climatological_standard_deviation.sel(month=month, drop=True)
        anomaly = calculate_anomaly(current, mean)
        zscore = calculate_z_score(anomaly, std)
        anomaly_series = regional.sst.groupby("time.month") - climatology.climatological_mean
        persistence = anomaly_persistence(
            anomaly_series, analysis_date, persistence_window, threshold
        )
        latitude_profile, longitude_profile = anomaly_profiles(anomaly)
        centroid_lon, centroid_lat = warm_anomaly_centroid(anomaly, threshold)
        fields = fields.assign(
            climatological_mean=mean,
            climatological_standard_deviation=std,
            sst_anomaly=anomaly,
            sst_z_score=zscore,
            anomaly_persistence=persistence,
            latitude_anomaly_profile=latitude_profile,
            longitude_anomaly_profile=longitude_profile,
        )
        lat_weights = np.cos(np.deg2rad(anomaly.latitude))
        metrics.update(
            mean_sst_anomaly_c=float(area_weighted_mean(anomaly)),
            maximum_anomaly_c=float(anomaly.max(skipna=True)),
            p90_anomaly_c=float(
                anomaly.weighted(lat_weights).quantile(
                    0.9, dim=("latitude", "longitude"), skipna=True
                )
            ),
            area_anomaly_ge_threshold_percent=float(
                percentage_area_at_or_above_threshold(anomaly, threshold)
            ),
            warm_centroid_longitude=float(centroid_lon) if centroid_lon.notnull() else None,
            warm_centroid_latitude=float(centroid_lat) if centroid_lat.notnull() else None,
        )
    report: dict[str, Any] = {
        "date": analysis_date,
        "data_mode": mode,
        "climatology_available": climatology is not None,
        "warning": warning,
        "threshold_c": threshold,
        "persistence_window_days": persistence_window,
        "experimental_product": True,
        "official_enso_classification": "not provided",
        "metrics": metrics,
    }
    return fields, report


def calculate_daily_diagnosis(dataset: xr.Dataset) -> tuple[xr.Dataset, dict[str, Any]]:
    """Backward-compatible latest-date synthetic diagnosis."""
    from src.climatology import monthly_climatology

    return diagnose_date(
        dataset,
        available_dates(dataset)[-1],
        climatology=monthly_climatology(dataset.sst).assign(
            valid_observation_count=dataset.sst.groupby("time.month").count("time")
        ),
        data_mode="demo",
    )
