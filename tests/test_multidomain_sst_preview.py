from copy import deepcopy
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import scripts.preview_multidomain_sst as preview
from src.utils import load_config


class Foundation:
    warnings = ()
    corridor_status = {
        "available": True,
        "include_islands": False,
        "coastline_type": "continental_mainland_only",
    }


def test_preview_writes_png_and_validation_json_from_local_products(tmp_path, monkeypatch) -> None:
    config = deepcopy(load_config())
    outputs = config["sst"]["outputs"]
    outputs["processed_directory"] = str(tmp_path / "processed")
    outputs["snapshot"] = str(tmp_path / "snapshot.json")
    outputs["preview_figure"] = str(tmp_path / "preview.png")
    outputs["preview_report"] = str(tmp_path / "preview.json")
    config["sst"]["regional_indices"]["output"] = str(tmp_path / "indices.parquet")
    processed = tmp_path / "processed"
    processed.mkdir()
    data = xr.Dataset(
        {"sst": (("time", "latitude", "longitude"), np.ones((1, 2, 2)) * 25)},
        coords={"time": ["2026-01-01"], "latitude": [-1, 0], "longitude": [-90, -89]},
        attrs={"source_mode": "demo"},
    )
    data.sst.attrs["units"] = "degrees_Celsius"
    data.to_netcdf(processed / "pacific_context_latest.nc")
    data.to_netcdf(processed / "humboldt_coastal_latest.nc")
    pd.DataFrame({"date": [pd.Timestamp("2026-01-01")], "region_id": ["nino12"], "region_label": ["Niño 1+2"], "mean_sst_c": [25.0]}).to_parquet(tmp_path / "indices.parquet", index=False)
    (tmp_path / "snapshot.json").write_text(json.dumps({"domains_loaded": ["pacific_context", "humboldt_coastal"], "source_modes": {"pacific_context": "demo", "humboldt_coastal": "demo"}, "latest_dates": {}, "domains": {}, "nino_region_coverage": {}, "warnings": []}), encoding="utf-8")

    monkeypatch.setattr(
        preview,
        "create_preview_figure",
        lambda *args: (plt.figure(figsize=(19.2, 12), dpi=120), Foundation()),
    )
    report = preview.build_preview(config)
    assert report["validation_status"] == "valid"
    assert report["errors"] == []
    assert (tmp_path / "preview.png").exists()
    saved = json.loads((tmp_path / "preview.json").read_text(encoding="utf-8"))
    assert saved["preview_dimensions_pixels"][0] >= 1920
