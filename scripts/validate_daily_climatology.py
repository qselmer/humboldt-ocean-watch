"""Validate a Humboldt Ocean Watch daily climatology and report continuity."""

from __future__ import annotations

import argparse
import sys
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.daily_climatology import DAILY_VARIABLES, validate_daily_climatology
from src.export_utils import dumps_json_safe
from src.spatial_metrics import area_weighted_mean
from src.utils import resolve_project_path

REPORT_PATH = Path("outputs/reports/daily_climatology_validation.json")
FIGURE_PATH = Path("outputs/figures/daily_climatology_continuity.png")
BOUNDARIES = ("01-31", "02-28", "02-29", "06-30", "12-31")


def _scalar(value: xr.DataArray) -> float | None:
    number = float(value)
    return number if np.isfinite(number) else None


def validate_file(path: Path) -> tuple[dict[str, Any], xr.Dataset | None]:
    report: dict[str, Any] = {"input": str(path), "valid": False, "errors": [], "variables": {}}
    if not path.exists():
        report["errors"].append("file does not exist")
        return report, None
    try:
        dataset = xr.load_dataset(path)
    except (OSError, ValueError) as exc:
        report["errors"].append(f"file is unreadable: {exc}")
        return report, None
    errors = validate_daily_climatology(dataset)
    report["errors"].extend(errors)
    for name in DAILY_VARIABLES:
        if name not in dataset:
            continue
        values = dataset[name]
        finite = np.isfinite(values)
        report["variables"][name] = {
            "minimum": _scalar(values.min(skipna=True)),
            "maximum": _scalar(values.max(skipna=True)),
            "finite_fraction": float(finite.sum() / values.size),
        }
    if "climatology_mean" in dataset and "month_day" in dataset.coords:
        regional = area_weighted_mean(dataset.climatology_mean)
        jumps = abs(regional.diff("climatological_day"))
        lookup = {str(value): index for index, value in enumerate(dataset.month_day.values)}

        def jump(first: str, second: str) -> float | None:
            if first not in lookup or second not in lookup:
                return None
            return _scalar(abs(regional.isel(climatological_day=lookup[second]) - regional.isel(climatological_day=lookup[first])))

        continuity = {
            "maximum_regional_daily_climatology_jump_c": _scalar(jumps.max(skipna=True)),
            "december_31_to_january_1_jump_c": jump("12-31", "01-01"),
            "january_31_to_february_1_jump_c": jump("01-31", "02-01"),
            "february_28_to_february_29_jump_c": jump("02-28", "02-29"),
            "february_29_to_march_1_jump_c": jump("02-29", "03-01"),
            "june_30_to_july_1_jump_c": jump("06-30", "07-01"),
        }
        report["continuity"] = continuity
        if any(value is None for value in continuity.values()):
            report["errors"].append("calendar-boundary continuity cannot be calculated from finite values")
    report["valid"] = not report["errors"]
    return report, dataset


def create_continuity_figure(dataset: xr.Dataset, output: Path) -> Path:
    """Plot daily mean, optional raw mean, and optional monthly sensitivity."""
    figure, axis = plt.subplots(figsize=(12, 5), facecolor="white")
    day = dataset.climatological_day.values
    axis.plot(day, area_weighted_mean(dataset.climatology_mean), label="Smoothed daily mean", linewidth=2)
    if "climatology_mean_unsmoothed" in dataset:
        axis.plot(day, area_weighted_mean(dataset.climatology_mean_unsmoothed), label="Unsmoothed daily reference", alpha=0.55)
    monthly_path = resolve_project_path("data/climatology/nino12_monthly_climatology_1991_2020.nc")
    if monthly_path.exists():
        with xr.open_dataset(monthly_path) as monthly:
            monthly_mean = area_weighted_mean(monthly.climatological_mean).load()
        axis.step(day, monthly_mean.sel(month=dataset.month).values, where="mid", label="Monthly sensitivity", alpha=0.8)
    month_starts = np.flatnonzero(np.r_[True, np.diff(dataset.month.values) != 0]) + 1
    for boundary in month_starts[1:]:
        axis.axvline(boundary, color="0.8", linewidth=0.6, zorder=0)
    axis.set(xlabel="Climatological day", ylabel="Regional SST (°C)", title="Daily climatology continuity")
    axis.grid(True, color="0.9", linewidth=0.5)
    axis.legend()
    figure.tight_layout()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output, dpi=160, facecolor="white")
    plt.close(figure)
    return output


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    args = parser.parse_args(argv)
    report, dataset = validate_file(args.input)
    report_path = resolve_project_path(REPORT_PATH)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(dumps_json_safe(report), encoding="utf-8")
    if dataset is not None and report["valid"]:
        create_continuity_figure(dataset, resolve_project_path(FIGURE_PATH))
    print(dumps_json_safe(report))
    return 0 if report["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
