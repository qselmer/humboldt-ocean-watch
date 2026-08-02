import numpy as np
import pytest
import xarray as xr

from src.sst_regional_indices import INDEX_COLUMNS, calculate_nino_region_sst
from src.utils import load_config


def pacific_dataset(value=25.0):
    longitude = np.arange(-170.0, -69.9, 5.0)
    latitude = np.arange(-45.0, 10.1, 5.0)
    values = np.full((2, len(latitude), len(longitude)), value)
    result = xr.Dataset(
        {"sst": (("time", "latitude", "longitude"), values)},
        coords={"time": ["2026-01-01", "2026-01-02"], "latitude": latitude, "longitude": longitude},
    )
    result.sst.attrs["units"] = "degrees_Celsius"
    result.attrs["source_mode"] = "demo"
    return result


def test_constant_field_has_known_means_coverage_and_no_anomaly() -> None:
    table = calculate_nino_region_sst(pacific_dataset(), load_config())
    assert list(table.columns) == INDEX_COLUMNS
    assert len(table) == 6
    np.testing.assert_allclose(table.mean_sst_c, 25.0)
    np.testing.assert_allclose(table.valid_coverage, 1.0)
    assert set(table.region_id) == {"nino34", "nino3", "nino12"}
    assert table.anomaly_c.isna().all()
    assert set(table.anomaly_status) == {"not_calculated_no_compatible_climatology"}
    assert "classification" not in " ".join(table.columns)


def test_latitude_gradient_matches_direct_cosine_weighting() -> None:
    source = pacific_dataset()
    source.sst[:] = source.latitude.broadcast_like(source.sst.isel(time=0))
    table = calculate_nino_region_sst(source, load_config(), region_ids=["nino34"])
    selected = source.sel(latitude=slice(-5, 5), longitude=slice(-170, -120)).sst.isel(time=0)
    weights = np.cos(np.deg2rad(selected.latitude)).broadcast_like(selected)
    expected = float((selected * weights).sum() / weights.sum())
    np.testing.assert_allclose(table.mean_sst_c, expected)


def test_incomplete_region_is_explicit_error() -> None:
    source = pacific_dataset().sel(longitude=slice(-160, None))
    with pytest.raises(ValueError, match="completely cover"):
        calculate_nino_region_sst(source, load_config(), region_ids=["nino34"])
