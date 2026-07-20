"""Build the operational Niño 1+2 daily metrics Parquet table."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.daily_diagnosis import load_climatology
from src.data_loader import load_active_sst_dataset
from src.temporal_metrics import build_metrics_table
from src.utils import load_config, resolve_project_path


def main() -> None:
    config = load_config()
    data = config["data"]
    region = config["region"]
    dataset, mode = load_active_sst_dataset(
        resolve_project_path(data["live_path"]),
        resolve_project_path(data["demo_path"]),
        aliases=data["sst_aliases"],
        demo_options={
            "longitude": tuple(region["longitude"]),
            "latitude": tuple(region["latitude"]),
            "days": data["demo_days"],
            "seed": data["demo_seed"],
        },
    )
    climatology_path = resolve_project_path(data["climatology_path"])
    climatology = load_climatology(str(climatology_path)) if climatology_path.exists() else None
    table = build_metrics_table(dataset, climatology)
    output = resolve_project_path(data["metrics_path"])
    output.parent.mkdir(parents=True, exist_ok=True)
    table.attrs = {"data_mode": mode, "climatology_available": climatology is not None}
    table.to_parquet(output, index=False)
    print(f"Mode: {mode}")
    print(f"Rows: {len(table)}")
    print(output)


if __name__ == "__main__":
    main()
