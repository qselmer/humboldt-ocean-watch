from copy import deepcopy
from dataclasses import FrozenInstanceError

import pytest

from src.sst_climatology_spec import load_sst_climatology_specs
from src.utils import load_config


def test_three_independent_climatology_specs_preserve_operational_paths() -> None:
    config = load_config()
    specs = load_sst_climatology_specs(config)
    assert set(specs) == {"nino12", "pacific_context", "humboldt_coastal"}
    assert specs["nino12"].daily_path.as_posix() == config["climatology"]["daily_path"]
    assert specs["nino12"].monthly_path.as_posix() == config["climatology"]["monthly_path"]
    assert len({spec.daily_path for spec in specs.values()}) == 3
    assert all(spec.reference_period == (1991, 2020) for spec in specs.values())
    assert all(spec.source_product_family == "ostia" for spec in specs.values())
    assert all("nino12" not in specs[key].daily_path.name for key in ("pacific_context", "humboldt_coastal"))
    with pytest.raises(FrozenInstanceError):
        specs["pacific_context"].domain_id = "changed"  # type: ignore[misc]


@pytest.mark.parametrize(
    ("mutation", "message"),
    [
        (lambda c: c["sst"]["domains"]["pacific_context"]["climatology"].update(reference_period=[2020, 1991]), "reversed"),
        (lambda c: c["sst"]["domains"]["pacific_context"]["climatology"].update(method="invented"), "Unsupported"),
        (lambda c: c["sst"]["domains"]["pacific_context"]["climatology"].update(target_resolution_degrees=0), "resolution"),
        (lambda c: c["sst"]["domains"]["pacific_context"]["climatology"].update(minimum_valid_coverage=1.1), "coverage"),
        (lambda c: c["sst"]["domains"]["pacific_context"]["climatology"].update(source_product_family=""), "source_product_family"),
    ],
)
def test_invalid_spec_contracts_are_rejected(mutation, message) -> None:
    config = deepcopy(load_config())
    mutation(config)
    with pytest.raises(ValueError, match=message):
        load_sst_climatology_specs(config)
