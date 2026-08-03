"""Validate assembled real or pilot multidomain daily climatologies."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.daily_climatology import validate_daily_climatology
from src.real_climatology_builder import atomic_json, project_relative
from src.sst_domains import load_sst_domain_specs
from src.utils import load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    selection = parser.add_mutually_exclusive_group(required=True)
    selection.add_argument("--all", action="store_true")
    selection.add_argument("--domain", choices=("pacific_context", "humboldt_coastal"))
    parser.add_argument("--pilot", action="store_true")
    parser.add_argument("--json-output", type=Path, default=Path("outputs/reports/real_climatology_validation.json"))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    config = load_config()
    specs = load_sst_domain_specs(config).domains
    ids = ("pacific_context", "humboldt_coastal") if args.all else (args.domain,)
    results = {}
    valid = True
    for domain_id in ids:
        path = resolve_project_path(
            f"data/climatology/pilot/{domain_id}_daily_1991_1992_pilot.nc"
            if args.pilot else specs[domain_id].climatology_daily_path
        )
        errors: list[str] = []
        if not path.exists():
            errors.append("missing output")
        else:
            try:
                with xr.open_dataset(path) as dataset:
                    errors.extend(validate_daily_climatology(dataset))
                    expected_pilot = "true" if args.pilot else "false"
                    if str(dataset.attrs.get("pilot_build", "false")).lower() != expected_pilot:
                        errors.append("pilot metadata mismatch")
                    if args.pilot and dataset.attrs.get("scientific_use") != "pipeline_validation_only":
                        errors.append("pilot scientific_use mismatch")
                    resolution = float(np.median(np.diff(dataset.longitude.values)))
                    if not np.isclose(resolution, specs[domain_id].target_resolution_degrees, atol=0.001):
                        errors.append("target resolution mismatch")
            except (OSError, ValueError, TypeError) as exc:
                errors.append(f"unreadable output ({type(exc).__name__})")
        results[domain_id] = {"path": project_relative(path), "status": "valid" if not errors else "invalid", "errors": errors}
        valid &= not errors
    regional_result = {"status": "not_checked", "errors": []}
    if args.pilot:
        regional_path = resolve_project_path(config["real_climatology_build"]["pilot_regional_output"])
        regional_errors: list[str] = []
        if not regional_path.exists():
            regional_errors.append("missing regional climatology")
        else:
            try:
                regional = pd.read_parquet(regional_path)
                required = {
                    "climatological_day", "region_id", "climatology_mean_c",
                    "climatology_median_c", "climatology_std_c", "threshold_p10_c",
                    "threshold_p90_c", "observation_count", "valid_coverage",
                }
                if required - set(regional):
                    regional_errors.append("regional columns are incomplete")
                elif len(regional) != 366 * 3 or set(regional.region_id) != {"nino34", "nino3", "nino12"}:
                    regional_errors.append("regional climatology must contain 366 days for three regions")
                else:
                    if (regional.threshold_p10_c > regional.threshold_p90_c).any():
                        regional_errors.append("regional P10 exceeds P90")
                    if (regional.climatology_std_c < 0).any():
                        regional_errors.append("regional standard deviation is negative")
                    if (regional.observation_count < 0).any():
                        regional_errors.append("regional observation count is negative")
                    if (regional.valid_coverage < float(config["real_climatology_build"]["minimum_valid_coverage"])).any():
                        regional_errors.append("regional coverage is insufficient")
                    if (regional.valid_coverage > 1.0 + 1.0e-9).any():
                        regional_errors.append("regional coverage exceeds one")
                    if any("oni" in column.lower() or "enso" in column.lower() for column in regional.columns):
                        regional_errors.append("official-index/classification columns are prohibited")
            except (OSError, ValueError, TypeError, KeyError) as exc:
                regional_errors.append(f"unreadable regional output ({type(exc).__name__})")
        regional_result = {
            "path": project_relative(regional_path),
            "status": "valid" if not regional_errors else "invalid",
            "errors": regional_errors,
        }
        valid &= not regional_errors
    validated_at = datetime.now(timezone.utc).isoformat()
    payload = {
        "status": "valid" if valid else "invalid", "pilot": args.pilot,
        "domains": results, "regional": regional_result, "validated_at": validated_at,
    }
    atomic_json(resolve_project_path(args.json_output), payload)
    status_path = resolve_project_path(config["real_climatology_build"]["build_status_report"])
    if args.pilot and status_path.exists():
        try:
            build_status = json.loads(status_path.read_text(encoding="utf-8"))
            build_status["last_validation"] = validated_at
            build_status["validation_status"] = payload["status"]
            build_status["validation_report"] = project_relative(resolve_project_path(args.json_output))
            atomic_json(status_path, build_status)
            atomic_json(resolve_project_path(config["real_climatology_build"]["pilot_report"]), build_status)
        except (OSError, ValueError, TypeError):
            pass
    print(json.dumps(payload, indent=2, sort_keys=True))
    return 0 if valid else 2


if __name__ == "__main__":
    raise SystemExit(main())
