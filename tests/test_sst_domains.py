from copy import deepcopy

import pytest

from src.sst_domains import load_sst_domain_specs
from src.utils import load_config


def test_three_domain_specs_resolve_canonical_geography() -> None:
    registry = load_sst_domain_specs(load_config())
    assert registry.active_operational_domain == "nino12"
    assert set(registry.domains) == {"nino12", "pacific_context", "humboldt_coastal"}
    assert registry.domains["nino12"].role == "current_operational"
    assert registry.domains["pacific_context"].target_resolution_degrees == 0.25
    assert registry.domains["humboldt_coastal"].target_resolution_degrees == 0.05
    assert registry.domains["pacific_context"].bounds.contains(
        registry.domains["nino12"].bounds
    )


def test_domain_specs_reject_missing_geography_reference() -> None:
    config = deepcopy(load_config())
    config["sst"]["domains"]["pacific_context"]["geography_id"] = "missing"
    with pytest.raises(ValueError, match="Unknown analysis-domain"):
        load_sst_domain_specs(config)


def test_domain_specs_reject_invalid_resolution_and_active_id() -> None:
    config = deepcopy(load_config())
    config["sst"]["domains"]["nino12"]["target_resolution_degrees"] = 0
    with pytest.raises(ValueError, match="positive"):
        load_sst_domain_specs(config)
    config = deepcopy(load_config())
    config["sst"]["active_operational_domain"] = "missing"
    with pytest.raises(ValueError, match="active operational"):
        load_sst_domain_specs(config)
