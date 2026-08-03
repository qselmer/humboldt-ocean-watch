from dataclasses import replace

import numpy as np
import xarray as xr

from src.daily_climatology import calendar_coordinates
from src.multidomain_climatology import ClimatologyDomainResult
from src.multidomain_sst_anomalies import calculate_sst_anomaly_domain
from src.sst_climatology_compatibility import validate_sst_climatology_compatibility
from src.sst_climatology_spec import load_sst_climatology_specs
from src.sst_domains import load_sst_domain_specs
from src.utils import load_config


def _case(*, zero_std=False, percentiles=True):
    config = load_config()
    domain = load_sst_domain_specs(config).domains["pacific_context"]
    spec = load_sst_climatology_specs(config)["pacific_context"]
    coords = {"latitude": [-5.0, -4.75], "longitude": [-150.0, -149.75]}
    days = np.arange(1, 367, dtype=float)[:, None, None]
    mean = np.broadcast_to(days, (366, 2, 2)).copy()
    climate_vars = {
        "climatology_mean": (("climatological_day", "latitude", "longitude"), mean),
        "climatology_std": (("climatological_day", "latitude", "longitude"), np.zeros_like(mean) if zero_std else np.full_like(mean, 2.0)),
    }
    if percentiles:
        climate_vars.update(
            threshold_p10=(("climatological_day", "latitude", "longitude"), mean - 1),
            threshold_p90=(("climatological_day", "latitude", "longitude"), mean + 1),
        )
    climate = xr.Dataset(
        climate_vars,
        coords={**calendar_coordinates(), **coords},
        attrs={
            "domain_id": "pacific_context",
            "geography_id": "pacific_context",
            "climatology_method": "daily_smoothed",
            "climatology_source_mode": "synthetic_demonstration",
            "source_product_family": "synthetic_demo",
            "reference_period": "synthetic-demonstration",
            "calendar": "stable_366_gregorian",
        },
    )
    for name in climate.data_vars:
        climate[name].attrs["units"] = "degrees_Celsius"
    source_values = np.stack([np.full((2, 2), 62.0), np.full((2, 2), 63.0)])
    source = xr.Dataset(
        {"sst": (("time", "latitude", "longitude"), source_values)},
        coords={"time": ["2024-02-29", "2024-03-01"], **coords},
        attrs={
            "domain_id": "pacific_context",
            "geography_id": "pacific_context",
            "source_mode": "demo",
            "source_product_family": "synthetic_demo",
        },
    )
    source.sst.attrs["units"] = "degrees_Celsius"
    compatibility = validate_sst_climatology_compatibility(source, climate, domain, spec)
    selection = ClimatologyDomainResult(
        climate,
        "available",
        None,
        "daily_smoothed",
        "synthetic_demonstration",
        compatibility,
        (),
        (),
        None,
        compatibility.to_dict(),
        climatology_source_mode="synthetic_demonstration",
    )
    return source, selection, spec


def test_known_daily_anomaly_zscore_percentiles_and_leap_mapping() -> None:
    source, selection, spec = _case()
    result = calculate_sst_anomaly_domain(source, selection, spec)
    assert result.status == "calculated"
    np.testing.assert_allclose(result.dataset.sst_anomaly_c, 2.0)
    np.testing.assert_allclose(result.dataset.sst_z_score, 1.0)
    assert bool(result.dataset.exceeds_p90.all())
    assert not bool(result.dataset.below_p10.any())
    assert result.dataset.attrs["climatology_method"] == "daily_smoothed"
    # Feb 29 maps to bin 60 and Mar 1 remains bin 61; both yield anomaly 2 here.
    np.testing.assert_allclose(result.dataset.sst_anomaly_c[:, 0, 0], [2.0, 2.0])


def test_zero_standard_deviation_is_nan_and_deterministic() -> None:
    source, selection, spec = _case(zero_std=True)
    first = calculate_sst_anomaly_domain(source, selection, spec)
    second = calculate_sst_anomaly_domain(source, selection, spec)
    assert first.dataset.sst_z_score.isnull().all()
    xr.testing.assert_identical(first.dataset, second.dataset)


def test_missing_percentiles_is_explicit_partial_result() -> None:
    source, selection, spec = _case(percentiles=False)
    result = calculate_sst_anomaly_domain(source, selection, spec)
    assert result.status == "partial_missing_percentiles"
    assert "exceeds_p90" not in result.dataset


def test_insufficient_sst_coverage_stops_calculation() -> None:
    source, selection, spec = _case()
    source.sst[:] = np.nan
    result = calculate_sst_anomaly_domain(source, selection, spec)
    assert result.status == "not_calculated_insufficient_coverage"
    assert result.dataset is None
