import numpy as np
import pytest
import xarray as xr

from src.daily_climatology import calendar_coordinates
from src.sst_regional_climatology import build_nino_regional_climatology
from src.utils import load_config


def _climatology():
    longitude = np.arange(-170.0, -69.9, 5.0)
    latitude = np.arange(-45.0, 10.1, 5.0)
    values = np.broadcast_to(
        latitude[None, :, None], (366, len(latitude), len(longitude))
    ).copy()
    dataset = xr.Dataset(
        {
            "climatology_mean": (("climatological_day", "latitude", "longitude"), values),
            "climatology_std": (("climatological_day", "latitude", "longitude"), np.ones_like(values)),
            "threshold_p10": (("climatological_day", "latitude", "longitude"), values - 1),
            "threshold_p90": (("climatological_day", "latitude", "longitude"), values + 1),
        },
        coords={**calendar_coordinates(), "latitude": latitude, "longitude": longitude},
        attrs={
            "climatology_method": "daily_smoothed",
            "reference_period": "1991-2020",
            "source_product_family": "ostia",
            "climatology_source_mode": "real",
        },
    )
    return dataset


def test_regional_climatology_uses_configured_regions_and_cosine_weights() -> None:
    source = _climatology()
    table = build_nino_regional_climatology(source, load_config())
    assert len(table) == 366 * 3
    assert set(table.region_id) == {"nino34", "nino3", "nino12"}
    subset = source.sel(latitude=slice(-5, 5), longitude=slice(-170, -120)).climatology_mean.isel(climatological_day=0)
    weights = np.cos(np.deg2rad(subset.latitude)).broadcast_like(subset)
    expected = float((subset * weights).sum() / weights.sum())
    actual = table.query("region_id == 'nino34' and climatological_day == 1").climatology_mean_c.iloc[0]
    np.testing.assert_allclose(actual, expected, atol=1.0e-12)
    np.testing.assert_allclose(table.valid_coverage, 1.0)


def test_incomplete_pacific_climatology_is_rejected() -> None:
    source = _climatology().sel(longitude=slice(-160, None))
    with pytest.raises(ValueError, match="completely cover"):
        build_nino_regional_climatology(source, load_config(), region_ids=["nino34"])
