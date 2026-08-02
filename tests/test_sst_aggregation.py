import numpy as np
import pytest
import xarray as xr

from src.sst_aggregation import aggregate_sst_to_target_resolution


def dataset(values=None, *, step=0.05, size=6):
    if values is None:
        values = np.full((1, size, size), 20.0)
    result = xr.Dataset(
        {"sst": (("time", "latitude", "longitude"), values)},
        coords={
            "time": ["2026-01-01"],
            "latitude": np.arange(size) * step - 5.0,
            "longitude": np.arange(size) * step - 90.0,
        },
    )
    result.sst.attrs["units"] = "degrees_Celsius"
    return result


def test_005_to_025_uses_partial_edge_blocks_without_losing_them() -> None:
    aggregated, metadata = aggregate_sst_to_target_resolution(dataset(), 0.25)
    assert metadata.aggregation_applied is True
    assert metadata.latitude_factor == metadata.longitude_factor == 5
    assert metadata.partial_edge_blocks is True
    assert aggregated.sizes["latitude"] == aggregated.sizes["longitude"] == 2
    np.testing.assert_allclose(aggregated.sst, 20.0)


def test_weighting_nan_coverage_and_determinism() -> None:
    values = np.broadcast_to(np.arange(5.0)[:, None], (5, 5))[None, ...].copy()
    values[0, 0, 0] = np.nan
    first, _ = aggregate_sst_to_target_resolution(
        dataset(values, size=5), 0.25, minimum_valid_fraction=0.70
    )
    second, _ = aggregate_sst_to_target_resolution(
        dataset(values, size=5), 0.25, minimum_valid_fraction=0.70
    )
    xr.testing.assert_identical(first, second)
    assert 0.0 < float(first.valid_coverage.item()) < 1.0
    assert np.isfinite(first.sst).all()


def test_coverage_gate_noop_and_rejections() -> None:
    values = np.full((1, 5, 5), np.nan)
    values[:, :2, :] = 20.0
    aggregated, _ = aggregate_sst_to_target_resolution(
        dataset(values, size=5), 0.25, minimum_valid_fraction=0.80
    )
    assert np.isnan(aggregated.sst).all()
    unchanged, metadata = aggregate_sst_to_target_resolution(dataset(step=0.25), 0.25)
    assert metadata.aggregation_applied is False
    xr.testing.assert_equal(unchanged.sst, dataset(step=0.25).sst)
    with pytest.raises(ValueError, match="upsample"):
        aggregate_sst_to_target_resolution(dataset(step=0.25), 0.05)
    with pytest.raises(ValueError, match="approximately integer"):
        aggregate_sst_to_target_resolution(dataset(step=0.10), 0.25)
