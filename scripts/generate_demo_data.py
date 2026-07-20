"""Generate the synthetic daily SST demo file."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.data_loader import generate_synthetic_sst
from src.utils import configure_logging, load_config, resolve_project_path


def main() -> None:
    """Generate demo data according to config.yaml."""
    config = load_config()
    configure_logging(config["logging"]["level"])
    region = config["region"]
    output = generate_synthetic_sst(
        resolve_project_path(config["data"]["input_path"]),
        longitude=tuple(region["longitude"]),
        latitude=tuple(region["latitude"]),
        days=config["data"]["demo_days"],
        seed=config["data"]["demo_seed"],
    )
    print(output)


if __name__ == "__main__":
    main()
