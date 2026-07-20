"""Build a normalized latest-day SST snapshot."""

from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import load_sst_dataset
from src.export_utils import export_netcdf
from src.utils import configure_logging, load_config, resolve_project_path


def main() -> None:
    """Load or create SST input and export its most recent day."""
    config = load_config()
    configure_logging(os.getenv("HOW_LOG_LEVEL", config["logging"]["level"]))
    region = config["region"]
    source = resolve_project_path(os.getenv("HOW_SST_PATH", config["data"]["input_path"]))
    dataset = load_sst_dataset(
        source,
        aliases=config["data"]["sst_aliases"],
        demo_options={
            "longitude": tuple(region["longitude"]),
            "latitude": tuple(region["latitude"]),
            "days": config["data"]["demo_days"],
            "seed": config["data"]["demo_seed"],
        },
    )
    snapshot = dataset.isel(time=[-1])
    output = export_netcdf(snapshot, resolve_project_path(config["data"]["processed_path"]))
    print(output)


if __name__ == "__main__":
    main()
