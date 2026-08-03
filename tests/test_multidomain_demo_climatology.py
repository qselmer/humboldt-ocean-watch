from dataclasses import replace

import numpy as np

from scripts.generate_multidomain_demo_climatology import build_demo_climatology
from src.sst_climatology_spec import load_sst_climatology_specs
from src.sst_domains import load_sst_domain_specs
from src.utils import load_config


def test_demo_climatology_is_366_day_deterministic_and_unambiguous() -> None:
    config = load_config()
    climate = replace(
        load_sst_climatology_specs(config)["pacific_context"],
        target_resolution_degrees=5.0,
    )
    domain = load_sst_domain_specs(config).domains["pacific_context"]
    first = build_demo_climatology(climate, domain, seed=7).compute()
    second = build_demo_climatology(climate, domain, seed=7).compute()
    assert first.sizes["climatological_day"] == 366
    xr_variables = {"climatology_mean", "climatology_median", "climatology_std", "threshold_p10", "threshold_p90", "observation_count"}
    assert xr_variables <= set(first.data_vars)
    np.testing.assert_array_equal(first.climatology_mean, second.climatology_mean)
    assert first.attrs["climatology_source_mode"] == "synthetic_demonstration"
    assert first.attrs["scientific_use"] == "demonstration_only"
    assert first.attrs["represents_observed_climatology"] == "false"
    assert first.month_day.values[59] == "02-29"
    assert first.month_day.values[60] == "03-01"
