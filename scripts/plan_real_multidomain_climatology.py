"""Plan resources and deterministic tiles before any real SST download."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.climatology_build_plan import build_climatology_plan
from src.real_climatology_builder import atomic_json
from src.utils import load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--domain", choices=("pacific_context", "humboldt_coastal"))
    parser.add_argument("--start-year", type=int, default=1991)
    parser.add_argument("--end-year", type=int, default=2020)
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--json-output", type=Path)
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    ids = ("pacific_context", "humboldt_coastal") if args.all else (args.domain,)
    plan = build_climatology_plan(
        config, ids, args.start_year, args.end_year, pilot=args.pilot,
    )
    output = resolve_project_path(args.json_output or config["real_climatology_build"]["plan_report"])
    atomic_json(output, plan.to_dict())
    print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
    return 0 if plan.status == "valid" else 2


if __name__ == "__main__":
    raise SystemExit(main())
