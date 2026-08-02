import numpy as np

from scripts.generate_multidomain_demo_data import build_synthetic_domain_dataset
from src.data_loader import load_sst_domain_dataset
from src.sst_domains import load_sst_domain_specs
from src.utils import load_config


def test_demo_is_deterministic_bounded_regular_and_kelvin(tmp_path) -> None:
    spec = load_sst_domain_specs(load_config()).domains["pacific_context"]
    first = build_synthetic_domain_dataset(spec, days=3, seed=7)
    second = build_synthetic_domain_dataset(spec, days=3, seed=7)
    different = build_synthetic_domain_dataset(spec, days=3, seed=8)
    np.testing.assert_array_equal(first.analysed_sst, second.analysed_sst)
    assert not np.array_equal(first.analysed_sst, different.analysed_sst)
    assert first.analysed_sst.attrs["units"] == "kelvin"
    assert first.attrs["product_status"] == "synthetic demonstration data"
    np.testing.assert_allclose(np.diff(first.longitude), 0.25)
    assert 270.0 < float(first.analysed_sst.min()) < 310.0
    path = tmp_path / "demo.nc"
    first.to_netcdf(path)
    normalized = load_sst_domain_dataset(path, spec, source_mode="demo")
    assert normalized.sst.attrs["units"] == "degrees_Celsius"
    assert normalized.attrs["domain_id"] == "pacific_context"
