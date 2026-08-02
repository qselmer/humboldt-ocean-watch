from copy import deepcopy

from src.multidomain_sst import load_multidomain_sst
from src.utils import load_config


def test_missing_contextual_domains_are_isolated_not_global_errors(tmp_path) -> None:
    config = deepcopy(load_config())
    for domain_id in ("pacific_context", "humboldt_coastal"):
        domain = config["sst"]["domains"][domain_id]
        domain["live_path"] = str(tmp_path / f"{domain_id}-live.nc")
        domain["demo_path"] = str(tmp_path / f"{domain_id}-demo.nc")
    results = load_multidomain_sst(config, allow_demo=True)
    assert set(results) == {"pacific_context", "humboldt_coastal"}
    assert all(result.status == "missing" for result in results.values())
    assert all(result.dataset is None for result in results.values())


def test_demo_fallback_is_never_generated_implicitly(tmp_path) -> None:
    config = deepcopy(load_config())
    demo = tmp_path / "absent.nc"
    config["sst"]["domains"]["pacific_context"]["live_path"] = str(tmp_path / "live.nc")
    config["sst"]["domains"]["pacific_context"]["demo_path"] = str(demo)
    result = load_multidomain_sst(config, ["pacific_context"], allow_demo=True)
    assert result["pacific_context"].status == "missing"
    assert not demo.exists()
