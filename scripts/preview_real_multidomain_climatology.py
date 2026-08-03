"""Render the explicitly non-operational 1991-1992 real climatology pilot."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_multidomain_sst_snapshot import _promote_atomically
from scripts.preview_multidomain_sst import _draw_lines, _fill_geometry
from src.geographic_foundation import prepare_geographic_foundation
from src.climatology_build_plan import build_climatology_plan
from src.sst_domains import load_sst_domain_specs
from src.utils import load_config, resolve_project_path


def _map(axis, dataset: xr.Dataset, field_name: str, title: str, figure, bounds):
    field = dataset[field_name].sel(climatological_day=1)
    mesh = axis.pcolormesh(
        dataset.longitude, dataset.latitude, field, shading="auto", cmap="turbo", rasterized=True,
    )
    axis.set(xlim=bounds.longitude, ylim=bounds.latitude, xlabel="Longitude (deg)", ylabel="Latitude (deg)")
    axis.grid(color="0.75", linestyle=":", linewidth=0.5)
    axis.set_title(title, loc="left", fontsize=10, weight="bold")
    figure.colorbar(mesh, ax=axis, pad=0.01, shrink=0.8).set_label("SST (degrees Celsius)")


def build_preview(config: dict, *, overwrite: bool) -> dict:
    if not overwrite:
        raise FileExistsError("Pilot preview publication requires --overwrite")
    build = config["real_climatology_build"]
    paths = {
        "pacific": resolve_project_path("data/climatology/pilot/pacific_context_daily_1991_1992_pilot.nc"),
        "coastal": resolve_project_path("data/climatology/pilot/humboldt_coastal_daily_1991_1992_pilot.nc"),
        "regional": resolve_project_path(build["pilot_regional_output"]),
        "status": resolve_project_path(build["build_status_report"]),
        "figure": resolve_project_path(build["pilot_preview"]),
        "report": resolve_project_path(build["pilot_report"]),
    }
    missing = [str(path) for key, path in paths.items() if key not in {"figure", "report", "regional"} and not path.exists()]
    if missing:
        raise FileNotFoundError("Pilot inputs are missing: " + ", ".join(missing))
    with xr.open_dataset(paths["pacific"]) as opened:
        pacific = opened.load()
    with xr.open_dataset(paths["coastal"]) as opened:
        coastal = opened.load()
    status = json.loads(paths["status"].read_text(encoding="utf-8"))
    pilot_plan = {
        item.domain_id: item
        for item in build_climatology_plan(
            config, ("pacific_context", "humboldt_coastal"), pilot=True
        ).domains
    }
    regional = pd.read_parquet(paths["regional"]) if paths["regional"].exists() else pd.DataFrame()
    specs = load_sst_domain_specs(config).domains
    foundation = prepare_geographic_foundation(config)
    figure = plt.figure(figsize=(19.2, 15.0), dpi=120, facecolor="white")
    grid = figure.add_gridspec(3, 2, height_ratios=(1.0, 0.8, 0.55), hspace=0.34, wspace=0.16)
    pacific_axis = figure.add_subplot(grid[0, 0])
    coastal_axis = figure.add_subplot(grid[0, 1])
    p90_axis = figure.add_subplot(grid[1, 0])
    regional_axis = figure.add_subplot(grid[1, 1])
    resource_axis = figure.add_subplot(grid[2, :])
    _map(pacific_axis, pacific, "climatology_mean", "A. Pacific context pilot mean — Jan 01", figure, specs["pacific_context"].bounds)
    for region in foundation.registry.standard_regions.values():
        pacific_axis.add_patch(Rectangle(
            (region.bounds.west, region.bounds.south), region.bounds.width, region.bounds.height,
            facecolor="none", edgecolor="#7C2D12", linewidth=1.2, zorder=8,
        ))
        pacific_axis.text(region.central_longitude, region.bounds.north + 0.7, region.label,
                          ha="center", fontsize=8, color="#7C2D12", weight="bold")
    _map(coastal_axis, coastal, "climatology_mean", "B. Humboldt coastal pilot mean — Jan 01", figure, specs["humboldt_coastal"].bounds)
    if foundation.corridor is not None:
        _fill_geometry(coastal_axis, foundation.corridor, facecolor="#10B981", edgecolor="#047857",
                       alpha=0.30, zorder=5, linewidth=0.8)
        coastal_axis.legend(handles=[Patch(facecolor="#10B981", edgecolor="#047857", alpha=0.30,
            label="60 nm mainland-only corridor; islands excluded")], loc="lower left", fontsize=8)
    if foundation.local_geometries is not None:
        local = foundation.local_geometries
        if local.all_land is not None:
            for axis in (pacific_axis, coastal_axis):
                _fill_geometry(axis, local.all_land, facecolor="#E7E5E4", edgecolor="#292524", alpha=1, zorder=7)
        _draw_lines(pacific_axis, local.display_coastline, zorder=9)
        _draw_lines(coastal_axis, local.display_coastline, zorder=9)

    for label, dataset, color in (("Pacific context", pacific, "#2563EB"), ("Humboldt coastal", coastal, "#DC2626")):
        values = np.asarray(dataset.threshold_p90.sel(climatological_day=1).values)
        finite = values[np.isfinite(values)]
        if finite.size:
            p90_axis.hist(finite, bins=25, density=True, histtype="step", linewidth=2, color=color, label=label)
    p90_axis.set_title("C. Exact pilot P90 comparison — Jan 01", loc="left", fontsize=10, weight="bold")
    p90_axis.set(xlabel="P90 SST (degrees Celsius)", ylabel="Density")
    p90_axis.grid(color="0.8", linestyle=":", linewidth=0.5)
    p90_axis.legend(frameon=False)

    if not regional.empty:
        colors = {"nino34": "#2563EB", "nino3": "#D97706", "nino12": "#DC2626"}
        for region_id, group in regional.groupby("region_id"):
            regional_axis.plot(group.climatological_day, group.climatology_mean_c,
                               color=colors.get(region_id), linewidth=1.5, label=region_id)
        regional_axis.legend(frameon=False, ncol=3)
    else:
        regional_axis.text(0.5, 0.5, "No complete regional series in this single-tile pilot",
                           ha="center", va="center", transform=regional_axis.transAxes)
    regional_axis.set_title("D. Experimental regional Nino pilot series", loc="left", fontsize=10, weight="bold")
    regional_axis.set(xlabel="Stable climatological day", ylabel="SST (degrees Celsius)")
    regional_axis.grid(color="0.8", linestyle=":", linewidth=0.5)

    rows = []
    observed_peak_gib = float(status.get("observed_peak_memory_bytes", 0)) / 1024**3
    cumulative_seconds = float(status.get("cumulative_duration_seconds", status.get("duration_seconds", 0)))
    for domain_id, details in status.get("domains", {}).items():
        planned_peak_gib = pilot_plan[domain_id].storage.peak_memory_bytes / 1024**3
        rows.append([
            domain_id, f"{details.get('cumulative_downloaded_bytes', 0):,}",
            f"{cumulative_seconds:.1f}" if not rows else "included above",
            f"{planned_peak_gib:.2f}", f"{observed_peak_gib:.2f}" if not rows else "included above",
            f"{details.get('checkpoint_bytes', 0) / 1024**2:.2f}", details.get("tiles"),
            details.get("resume_reused_checkpoints"), details.get("status"),
        ])
    resource_axis.axis("off")
    resource_axis.set_title("E. Pilot resource and validation record", loc="left", fontsize=10, weight="bold")
    table = resource_axis.table(cellText=rows, colLabels=[
        "Domain", "Downloaded bytes", "Cumulative s", "Peak GiB planned",
        "Peak GiB observed", "Checkpoint MiB", "Tiles", "Resumed", "Validation",
    ],
                                loc="center", cellLoc="left")
    table.auto_set_font_size(False); table.set_fontsize(8); table.scale(1, 1.5)
    figure.suptitle("Humboldt Ocean Watch — real climatology pipeline pilot", fontsize=16, weight="bold", y=0.985)
    figure.text(0.5, 0.958,
        "PILOT — 1991–1992 | NOT A 1991–2020 CLIMATOLOGY | NOT FOR OPERATIONAL INTERPRETATION",
        ha="center", color="#9A3412", weight="bold")
    width, height = figure.canvas.get_width_height()
    report = {
        **status, "preview_dimensions_pixels": [int(width), int(height)], "five_sections": True,
        "selected_climatological_day": 1, "coastal_corridor": foundation.corridor_status,
        "visual_labels": ["PILOT — 1991–1992", "NOT A 1991–2020 CLIMATOLOGY", "NOT FOR OPERATIONAL INTERPRETATION"],
        "warnings": list(status.get("warnings", [])) + list(foundation.warnings),
    }
    try:
        _promote_atomically([
            (paths["figure"], lambda path: figure.savefig(path, dpi=120, facecolor="white")),
            (paths["report"], lambda path: path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")),
        ], overwrite=True)
    finally:
        plt.close(figure)
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--pilot", action="store_true", required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    arguments = parse_args()
    print(json.dumps(build_preview(load_config(), overwrite=arguments.overwrite), indent=2))
