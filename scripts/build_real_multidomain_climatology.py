"""Dry-run or explicitly execute the resumable real multidomain SST builder."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.climatology_build_plan import build_climatology_plan
from src.copernicus_sst_source import CopernicusSSTAdapter
from src.real_climatology_builder import atomic_json, execute_build
from src.utils import load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--domain", choices=("pacific_context", "humboldt_coastal"))
    parser.add_argument("--start-year", type=int, default=1991)
    parser.add_argument("--end-year", type=int, default=2020)
    parser.add_argument("--dry-run", action="store_true", help="Print plan only; never use network")
    parser.add_argument("--pilot", action="store_true", help="Hard-limit execution to 1991-1992 pilot paths")
    parser.add_argument("--execute", action="store_true", help="Explicitly authorize network and writes")
    parser.add_argument("--confirm-full-build", action="store_true", help="Required in addition to --execute for 30 years")
    parser.add_argument("--resume", action="store_true")
    return parser.parse_args()


def _approved_preflight(config: dict) -> dict:
    path = resolve_project_path(config["real_climatology_build"]["preflight_report"])
    if not path.exists():
        raise RuntimeError("Online source preflight report is missing; no download was started")
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("status") != "valid" or payload.get("offline") or not payload.get("authentication_checked"):
        raise RuntimeError("Online source preflight is not valid and authenticated; no download was started")
    if payload.get("dataset_id") != "METOFFICE-GLO-SST-L4-REP-OBS-SST" or payload.get("source_variable") != "analysed_sst":
        raise RuntimeError("Source product contract mismatch; no download was started")
    return payload


def main() -> int:
    args = parse_args()
    config = load_config()
    ids = ("pacific_context", "humboldt_coastal") if args.all else (args.domain,)
    plan = build_climatology_plan(
        config, ids, args.start_year, args.end_year, pilot=args.pilot,
    )
    if args.dry_run:
        if args.execute:
            raise SystemExit("--dry-run and --execute are mutually exclusive")
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        return 0 if plan.status == "valid" else 2
    if not args.execute:
        raise SystemExit("Execution refused: pass --dry-run or explicitly pass --execute")
    if not args.pilot and not args.confirm_full_build:
        raise SystemExit("Full build refused: --execute and --confirm-full-build are both required")
    if args.pilot and args.confirm_full_build:
        raise SystemExit("Pilot and full-build confirmation cannot be combined")
    preflight = _approved_preflight(config)
    if plan.status != "valid":
        print(json.dumps(plan.to_dict(), indent=2, sort_keys=True))
        return 2
    import copernicusmarine
    adapter = CopernicusSSTAdapter(
        copernicusmarine,
        attempts=int(config["real_climatology_build"]["maximum_download_attempts"]),
        initial_backoff_seconds=float(config["real_climatology_build"]["retry_initial_seconds"]),
        timeout_seconds=float(config["real_climatology_build"]["network_timeout_seconds"]),
    )
    metadata = preflight.get("metadata") or {}
    report = execute_build(
        plan, config, adapter, resume=args.resume,
        source_version=metadata.get("dataset_version"),
    )
    print(json.dumps(report, indent=2, sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
