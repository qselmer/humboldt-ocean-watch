"""Preflight the real OSTIA reprocessed climatology source without exposing credentials."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.copernicus_sst_source import run_source_preflight
from src.real_climatology_builder import atomic_json
from src.sst_domains import load_sst_domain_specs
from src.utils import load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true", help="Check both production domains")
    selection.add_argument("--domain", choices=("pacific_context", "humboldt_coastal"))
    parser.add_argument("--offline", action="store_true", help="Check installed API and local contracts only")
    parser.add_argument("--json-output", type=Path, help="Override the configured JSON report path")
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    specs = load_sst_domain_specs(config).domains
    ids = ("pacific_context", "humboldt_coastal") if args.all else (args.domain,)
    domains = {item: specs[item].bounds for item in ids}
    result = run_source_preflight(config, domains, offline=args.offline)
    output = resolve_project_path(args.json_output or config["real_climatology_build"]["preflight_report"])
    atomic_json(output, result.to_dict())
    print(json.dumps(result.to_dict(), indent=2, sort_keys=True))
    return 0 if result.status == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
