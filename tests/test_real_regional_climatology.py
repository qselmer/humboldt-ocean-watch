import numpy as np
import pandas as pd
import xarray as xr

from src.real_regional_climatology import (
    build_regional_daily_climatology, merge_regional_partials, regional_partial_series,
)
from src.utils import load_config


def test_fine_grid_regional_partial_merge_uses_registry_and_cosine_weights() -> None:
    longitude = np.arange(-89.5, -79.9, 1.0)
    latitude = np.arange(-9.5, 0.0, 1.0)
    values = np.broadcast_to(latitude[None, :, None], (2, len(latitude), len(longitude))).copy()
    dataset = xr.Dataset(
        {"sst": (("time", "latitude", "longitude"), values)},
        coords={"time": pd.to_datetime(["1991-01-01", "1991-01-02"]),
                "latitude": latitude, "longitude": longitude},
    )
    dataset.sst.attrs["units"] = "degrees_Celsius"
    partials = regional_partial_series(dataset, load_config(), tile_id="tile", checkpoint_year=1991)
    merged = merge_regional_partials(partials, source_resolution=1.0, dataset_id="test")
    assert set(merged.region_id) == {"nino12"}
    assert np.allclose(merged.valid_coverage, 1.0)
    assert (merged.valid_coverage <= 1.0 + 1.0e-12).all()
    assert (merged.valid_cell_count == merged.total_cell_count).all()


def test_regional_climatology_has_exact_366_contract_without_oni_or_classification() -> None:
    dates = pd.date_range("1991-01-01", "1992-12-31")
    series = pd.DataFrame({
        "date": dates, "region_id": "nino12",
        "weighted_mean_sst_c": 20 + np.sin(np.arange(len(dates)) * 2 * np.pi / 366),
        "valid_coverage": 1.0,
    })
    result = build_regional_daily_climatology(series)
    assert len(result) == 366
    assert result.loc[result.climatological_day == 60, "month_day"].item() == "02-29"
    assert result.loc[result.climatological_day == 61, "month_day"].item() == "03-01"
    assert {"threshold_p10_c", "threshold_p90_c", "observation_count"} <= set(result)
    assert not any("oni" in name.lower() or "enso" in name.lower() for name in result.columns)
