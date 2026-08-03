from copy import deepcopy
import json

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import xarray as xr

import scripts.preview_multidomain_sst_anomalies as preview
from src.utils import load_config


class Foundation:
    warnings = ()
    corridor_status = {
        "available": True,
        "include_islands": False,
        "coastline_type": "continental_mainland_only",
    }


def test_anomaly_preview_writes_four_panel_png_and_complete_json(tmp_path, monkeypatch) -> None:
    config = deepcopy(load_config())
    outputs = config["sst"]["anomaly_outputs"]
    outputs.update(
        processed_directory=str(tmp_path / "processed"),
        status=str(tmp_path / "status.json"),
        regional_indices=str(tmp_path / "regional.parquet"),
        preview_figure=str(tmp_path / "preview.png"),
        preview_report=str(tmp_path / "preview.json"),
    )
    processed = tmp_path / "processed"
    processed.mkdir()
    dataset = xr.Dataset(
        {"sst_anomaly_c": (("time", "latitude", "longitude"), np.ones((1, 2, 2)))},
        coords={"time": ["2024-01-01"], "latitude": [-1.0, 0.0], "longitude": [-90.0, -89.0]},
    )
    dataset.to_netcdf(processed / "pacific_context_latest_anomaly.nc")
    dataset.to_netcdf(processed / "humboldt_coastal_latest_anomaly.nc")
    pd.DataFrame(
        {
            "date": [pd.Timestamp("2024-01-01")],
            "region_id": ["nino12"],
            "region_label": ["Niño 1+2"],
            "anomaly_c": [1.0],
        }
    ).to_parquet(tmp_path / "regional.parquet", index=False)
    status = {
        "domains": {
            "pacific_context": {
                "sst_mode": "demo",
                "climatology": {"compatibility": {"status": "compatible", "allowed_operations": ["anomaly"]}},
                "anomaly": {"status": "calculated"},
            }
        },
        "warnings": [],
        "errors": [],
        "regional_anomaly_status": {"nino12": ["calculated"]},
    }
    (tmp_path / "status.json").write_text(json.dumps(status), encoding="utf-8")
    monkeypatch.setattr(
        preview,
        "create_anomaly_preview_figure",
        lambda *args: (plt.figure(figsize=(19.2, 12.0), dpi=120), Foundation()),
    )
    report = preview.build_anomaly_preview(config)
    assert report["validation"] == "complete"
    assert report["four_panels"] is True
    assert report["diverging_scale_center_c"] == 0.0
    assert (tmp_path / "preview.png").exists()
    saved = json.loads((tmp_path / "preview.json").read_text(encoding="utf-8"))
    assert saved["coastal_corridor"]["include_islands"] is False
    assert saved["preview_dimensions_pixels"][0] >= 1920
