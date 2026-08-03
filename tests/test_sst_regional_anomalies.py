import numpy as np
import pandas as pd
import xarray as xr

from src.sst_regional_indices import calculate_nino_region_sst
from src.utils import load_config


def _sst(mode="demo"):
    longitude = np.arange(-170.0, -69.9, 5.0)
    latitude = np.arange(-45.0, 10.1, 5.0)
    dataset = xr.Dataset(
        {"sst": (("time", "latitude", "longitude"), np.full((1, len(latitude), len(longitude)), 25.0))},
        coords={"time": ["2024-03-01"], "latitude": latitude, "longitude": longitude},
        attrs={"source_mode": mode, "source_product_family": "synthetic_demo" if mode == "demo" else "ostia"},
    )
    dataset.sst.attrs["units"] = "degrees_Celsius"
    return dataset


def _regional_climate(mode="synthetic_demonstration"):
    rows = []
    for region_id, label in (("nino34", "Niño 3.4"), ("nino3", "Niño 3"), ("nino12", "Niño 1+2")):
        rows.append(
            {
                "region_id": region_id,
                "region_label": label,
                "climatological_day": 61,
                "climatology_mean_c": 24.0,
                "climatology_std_c": 0.5,
                "threshold_p90_c": 24.5,
                "climatology_method": "daily_smoothed",
                "source_product_family": "synthetic_demo" if "synthetic" in mode else "ostia",
                "source_mode": mode,
            }
        )
    return pd.DataFrame(rows)


def test_demo_regional_anomaly_known_without_oni_or_classification() -> None:
    table = calculate_nino_region_sst(
        _sst(), load_config(), regional_climatology=_regional_climate()
    )
    np.testing.assert_allclose(table.anomaly_c, 1.0)
    np.testing.assert_allclose(table.standardized_anomaly, 2.0)
    assert table.exceeds_p90.all()
    assert set(table.anomaly_status) == {"calculated"}
    assert "oni" not in " ".join(table.columns).lower()
    assert "classification" not in " ".join(table.columns).lower()


def test_missing_or_demo_live_incompatible_climatology_stays_null() -> None:
    missing = calculate_nino_region_sst(_sst(), load_config())
    assert missing.anomaly_c.isna().all()
    live = calculate_nino_region_sst(
        _sst("live"), load_config(), regional_climatology=_regional_climate()
    )
    assert live.anomaly_c.isna().all()
    assert set(live.anomaly_status) == {"not_calculated_incompatible_climatology"}
