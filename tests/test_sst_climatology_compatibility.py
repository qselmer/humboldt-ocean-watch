from copy import deepcopy

import numpy as np
import pytest
import xarray as xr

from src.daily_climatology import calendar_coordinates
from src.sst_climatology_compatibility import validate_sst_climatology_compatibility
from src.sst_climatology_spec import load_sst_climatology_specs
from src.sst_domains import load_sst_domain_specs
from src.utils import load_config


def _inputs(*, source_mode="demo"):
    coordinates = {"latitude": [-5.0, -4.75], "longitude": [-150.0, -149.75]}
    source = xr.Dataset(
        {"sst": (("time", "latitude", "longitude"), np.full((1, 2, 2), 25.0))},
        coords={"time": ["2024-03-01"], **coordinates},
        attrs={
            "domain_id": "pacific_context",
            "geography_id": "pacific_context",
            "source_mode": source_mode,
            "source_product_family": "synthetic_demo" if source_mode == "demo" else "ostia",
        },
    )
    source.sst.attrs["units"] = "degrees_Celsius"
    values = np.full((366, 2, 2), 24.0)
    climate = xr.Dataset(
        {
            "climatology_mean": (("climatological_day", "latitude", "longitude"), values),
            "climatology_std": (("climatological_day", "latitude", "longitude"), np.ones_like(values)),
            "threshold_p10": (("climatological_day", "latitude", "longitude"), values - 1),
            "threshold_p90": (("climatological_day", "latitude", "longitude"), values + 1),
        },
        coords={**calendar_coordinates(), **coordinates},
        attrs={
            "domain_id": "pacific_context",
            "geography_id": "pacific_context",
            "climatology_method": "daily_smoothed",
            "climatology_source_mode": "synthetic_demonstration" if source_mode == "demo" else "real",
            "source_product_family": "synthetic_demo" if source_mode == "demo" else "ostia",
            "reference_period": "synthetic-demonstration" if source_mode == "demo" else "1991-2020",
            "calendar": "stable_366_gregorian",
        },
    )
    for name in climate.data_vars:
        climate[name].attrs["units"] = "degrees_Celsius"
    config = load_config()
    domain = load_sst_domain_specs(config).domains["pacific_context"]
    spec = load_sst_climatology_specs(config)["pacific_context"]
    return source, climate, domain, spec


def _validate(source, climate, domain, spec):
    return validate_sst_climatology_compatibility(source, climate, domain, spec)


def test_compatible_demo_contract_and_reversed_latitude() -> None:
    source, climate, domain, spec = _inputs()
    result = _validate(source, climate.sortby("latitude", ascending=False), domain, spec)
    assert result.compatible
    assert result.status == "compatible"
    assert result.coordinate_match
    assert set(result.allowed_operations) == {"anomaly", "z_score", "percentile_exceedance"}


@pytest.mark.parametrize(
    ("change", "status"),
    [
        (lambda s, c: c.attrs.update(domain_id="nino12"), "domain_mismatch"),
        (lambda s, c: c.attrs.update(source_product_family="ostia"), "product_family_mismatch"),
        (lambda s, c: c.assign_coords(longitude=[-150.0, -149.5]), "resolution_mismatch"),
        (lambda s, c: c.assign_coords(longitude=[-151.0, -150.75]), "coordinate_mismatch"),
        (lambda s, c: c.drop_vars("climatology_mean"), "missing_mean"),
        (lambda s, c: c.isel(climatological_day=slice(0, 365)), "unsupported_calendar"),
    ],
)
def test_incompatible_contracts_are_explicit(change, status) -> None:
    source, climate, domain, spec = _inputs()
    changed = change(source, climate)
    if isinstance(changed, xr.Dataset):
        climate = changed
    result = _validate(source, climate, domain, spec)
    assert not result.compatible
    assert result.status == status


def test_live_sst_rejects_demo_climatology() -> None:
    source, climate, domain, spec = _inputs(source_mode="live")
    climate.attrs.update(
        climatology_source_mode="synthetic_demonstration",
        source_product_family="synthetic_demo",
    )
    result = _validate(source, climate, domain, spec)
    assert not result.compatible
    assert result.status == "synthetic_live_mismatch"


def test_missing_optional_variables_keep_mean_anomaly_compatible() -> None:
    source, climate, domain, spec = _inputs()
    result = _validate(
        source,
        climate.drop_vars(["climatology_std", "threshold_p10", "threshold_p90"]),
        domain,
        spec,
    )
    assert result.compatible
    assert result.status == "missing_standard_deviation"
    assert result.allowed_operations == ("anomaly",)


def test_insufficient_finite_coverage() -> None:
    source, climate, domain, spec = _inputs()
    climate.climatology_mean[:] = np.nan
    result = _validate(source, climate, domain, spec)
    assert result.status == "insufficient_coverage"
    assert not result.compatible
