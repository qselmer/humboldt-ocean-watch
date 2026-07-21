"""Deterministic tests for advanced daily spatial features."""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import xarray as xr

from src.spatial_features import (
    SPATIAL_COLUMNS,
    build_spatial_feature_table,
    calculate_spatial_features,
)


def _field(values: np.ndarray, *, units: str = "degrees_Celsius") -> xr.DataArray:
    return xr.DataArray(
        np.asarray(values, dtype=float),
        dims=("latitude", "longitude"),
        coords={
            "latitude": np.arange(values.shape[0], dtype=float),
            "longitude": 10.0 + np.arange(values.shape[1], dtype=float),
        },
        attrs={"units": units},
        name="field",
    )


def _by_family(results):
    return {(result.family, result.metric): result for result in results}


def test_weighted_spatial_state_quantiles_and_variability() -> None:
    field = _field(np.array([[0.0, 1.0], [2.0, 3.0]]))
    results = _by_family(calculate_spatial_features(
        field, variable="anomaly", weights=np.ones((2, 2)), threshold=1.0,
        minimum_valid_coverage=0.5, minimum_patch_cells=1,
    ))
    assert results[("state", "weighted_mean")].value == pytest.approx(1.5)
    assert results[("state", "weighted_median")].value == pytest.approx(1.5)
    assert results[("state", "weighted_p25")].value == pytest.approx(0.5)
    assert results[("state", "weighted_p75")].value == pytest.approx(2.5)
    assert results[("heterogeneity", "weighted_variance")].value == pytest.approx(1.25)
    assert results[("heterogeneity", "weighted_standard_deviation")].value == pytest.approx(np.sqrt(1.25))
    assert results[("heterogeneity", "weighted_iqr")].value == pytest.approx(2.0)


def test_threshold_area_excess_and_centroid_dispersion() -> None:
    values = np.zeros((3, 3), dtype=float)
    values[1, 1] = 3.0
    values[2, 2] = 5.0
    results = _by_family(calculate_spatial_features(
        _field(values), variable="anomaly", weights=np.ones((3, 3)), threshold=2.0,
        minimum_valid_coverage=0.5, minimum_patch_cells=1,
    ))
    assert results[("threshold", "area_over_threshold")].value == pytest.approx(200 / 9)
    assert results[("threshold", "weighted_excess_above_threshold")].value == pytest.approx(4 / 9)
    assert results[("centroid", "centroid_latitude")].value == pytest.approx(1.75)
    assert results[("centroid", "centroid_longitude")].value == pytest.approx(11.75)
    assert results[("centroid", "centroid_dispersion")].value == pytest.approx(np.sqrt(1.5 / 4.0))


def test_gradient_and_texture_calculations() -> None:
    latitude = np.arange(4.0)[:, None]
    longitude = (10.0 + np.arange(4.0))[None, :]
    values = 2.0 * latitude + 3.0 * longitude
    results = _by_family(calculate_spatial_features(
        _field(values), variable="sst", weights=np.ones((4, 4)), threshold=100.0,
        minimum_valid_coverage=0.5, minimum_patch_cells=1,
    ))
    assert results[("gradient", "latitudinal_gradient")].value == pytest.approx(2.0)
    assert results[("gradient", "longitudinal_gradient")].value == pytest.approx(3.0)
    assert results[("gradient", "thermal_gradient_magnitude")].value == pytest.approx(np.sqrt(13.0))
    assert results[("texture", "local_spatial_standard_deviation")].value > 0


def test_no_threshold_cells_make_centroid_and_patch_geometry_explicit() -> None:
    results = _by_family(calculate_spatial_features(
        _field(np.ones((3, 3))), variable="anomaly", threshold=2.0,
        minimum_valid_coverage=0.5, minimum_patch_cells=1,
    ))
    assert results[("threshold", "area_over_threshold")].value == 0
    assert results[("centroid", "centroid_latitude")].status == "not_calculated"
    assert results[("patch", "patch_count")].value == 0
    assert results[("patch", "orientation")].status == "not_calculated"


def test_insufficient_coverage_returns_not_calculated_features() -> None:
    values = np.full((3, 3), np.nan)
    values[0, 0] = 1.0
    results = _by_family(calculate_spatial_features(
        _field(values), variable="anomaly", minimum_valid_coverage=0.8,
    ))
    assert results[("coverage", "valid_coverage")].status == "valid"
    assert results[("state", "weighted_mean")].status == "not_calculated"
    assert results[("spatial_autocorrelation", "moran_i_rook")].status == "not_calculated"


def test_spatial_table_has_required_long_schema_and_both_moran_connectivities() -> None:
    first = _field(np.arange(16.0).reshape(4, 4)).expand_dims(time=[pd.Timestamp("2020-01-01")])
    second = (_field(np.arange(16.0).reshape(4, 4)) + 1).expand_dims(time=[pd.Timestamp("2020-01-02")])
    cube = xr.concat([first, second], dim="time").rename("anomaly")
    cube.attrs["units"] = "degrees_Celsius"
    table = build_spatial_feature_table(
        xr.Dataset({"anomaly": cube}), variables=["anomaly"],
        threshold=2.0, minimum_valid_coverage=0.5,
        connectivity="queen", minimum_patch_cells=1,
    )
    assert list(table.columns) == SPATIAL_COLUMNS
    assert table.date.nunique() == 2
    assert set(table.variable) == {"anomaly"}
    assert {"moran_i_rook", "moran_i_queen"}.issubset(set(table.metric))
    assert {"state", "heterogeneity", "threshold", "centroid", "gradient", "texture", "spatial_autocorrelation", "patch"}.issubset(set(table.family))
