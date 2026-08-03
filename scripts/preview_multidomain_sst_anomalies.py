"""Render a four-panel offline preview of prepared multidomain anomalies."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from scripts.build_multidomain_sst_snapshot import _promote_atomically
from scripts.preview_multidomain_sst import _draw_lines, _fill_geometry
from src.geographic_foundation import prepare_geographic_foundation
from src.sst_domains import load_sst_domain_specs
from src.utils import load_config, resolve_project_path


def _anomaly_map(axis, dataset: xr.Dataset, *, title: str, bounds, figure):
    field = dataset.sst_anomaly_c.isel(time=-1)
    finite = np.asarray(field.values)[np.isfinite(field.values)]
    limit = max(0.5, float(np.nanpercentile(np.abs(finite), 98))) if finite.size else 1.0
    mesh = axis.pcolormesh(
        dataset.longitude,
        dataset.latitude,
        field,
        shading="auto",
        cmap="RdBu_r",
        norm=TwoSlopeNorm(vmin=-limit, vcenter=0.0, vmax=limit),
        rasterized=True,
        zorder=1,
    )
    axis.set(
        xlim=bounds.longitude,
        ylim=bounds.latitude,
        xlabel="Longitude (°E)",
        ylabel="Latitude (°N)",
    )
    axis.grid(color="0.7", linestyle=":", linewidth=0.5)
    axis.set_title(title, loc="left", fontsize=11, weight="bold")
    colorbar = figure.colorbar(mesh, ax=axis, pad=0.015, shrink=0.88)
    colorbar.set_label("SST anomaly (°C)")


def create_anomaly_preview_figure(
    config: dict,
    pacific: xr.Dataset,
    coastal: xr.Dataset,
    regional: pd.DataFrame,
    status: dict,
):
    specs = load_sst_domain_specs(config).domains
    foundation = prepare_geographic_foundation(config)
    figure, axes = plt.subplots(2, 2, figsize=(19.2, 12.0), dpi=120, facecolor="white")
    pacific_axis, coastal_axis, series_axis, summary_axis = axes.ravel()
    _anomaly_map(
        pacific_axis,
        pacific,
        title="A. Pacific context SST anomaly [synthetic demonstration]",
        bounds=specs["pacific_context"].bounds,
        figure=figure,
    )
    for region in foundation.registry.standard_regions.values():
        pacific_axis.add_patch(
            Rectangle(
                (region.bounds.west, region.bounds.south),
                region.bounds.width,
                region.bounds.height,
                facecolor="none",
                edgecolor="#7C2D12",
                linewidth=1.4,
                zorder=10,
            )
        )
        pacific_axis.text(
            region.central_longitude,
            region.bounds.north + 0.7,
            region.label,
            ha="center",
            fontsize=8,
            weight="bold",
            color="#7C2D12",
            zorder=11,
            bbox={"facecolor": "white", "edgecolor": "#7C2D12", "alpha": 0.78, "pad": 1.5},
        )
    _anomaly_map(
        coastal_axis,
        coastal,
        title="B. Humboldt coastal SST anomaly [synthetic demonstration]",
        bounds=specs["humboldt_coastal"].bounds,
        figure=figure,
    )
    if foundation.corridor is not None:
        _fill_geometry(
            coastal_axis,
            foundation.corridor,
            facecolor="#10B981",
            edgecolor="#047857",
            alpha=0.30,
            zorder=5,
            linewidth=0.8,
        )
        coastal_axis.legend(
            handles=[
                Patch(
                    facecolor="#10B981",
                    edgecolor="#047857",
                    alpha=0.30,
                    label="60 nm continental-mainland corridor; islands excluded",
                )
            ],
            loc="lower left",
            fontsize=8,
        )
    if foundation.local_geometries is not None:
        local = foundation.local_geometries
        if local.all_land is not None:
            for axis in (pacific_axis, coastal_axis):
                _fill_geometry(
                    axis,
                    local.all_land,
                    facecolor="#E7E5E4",
                    edgecolor="#292524",
                    alpha=1.0,
                    zorder=7,
                )
        _draw_lines(pacific_axis, local.display_coastline, zorder=9)
        _draw_lines(coastal_axis, local.display_coastline, zorder=9)

    colors = {"nino34": "#2563EB", "nino3": "#D97706", "nino12": "#DC2626"}
    for region_id, group in regional.groupby("region_id", sort=False):
        ordered = group.sort_values("date")
        series_axis.plot(
            pd.to_datetime(ordered.date),
            ordered.anomaly_c,
            marker="o",
            linewidth=1.7,
            markersize=4,
            color=colors.get(region_id),
            label=str(ordered.region_label.iloc[0]),
        )
    series_axis.axhline(0.0, color="black", linewidth=1.0)
    series_axis.set_title(
        "C. Niño regional SST anomalies [synthetic demonstration]",
        loc="left",
        fontsize=11,
        weight="bold",
    )
    series_axis.set(xlabel="Date", ylabel="SST anomaly (°C)")
    series_axis.grid(color="0.75", linestyle=":", linewidth=0.6)
    series_axis.legend(ncol=3, frameon=False)
    series_axis.text(
        0.0,
        -0.18,
        "Demonstration only — no ONI and no official ENSO classification.",
        transform=series_axis.transAxes,
        color="#7C2D12",
        fontsize=9,
    )

    summary_axis.axis("off")
    summary_axis.set_title(
        "D. Compatibility and coverage summary", loc="left", fontsize=11, weight="bold"
    )
    rows = []
    for domain_id, details in status.get("domains", {}).items():
        compatibility = details.get("climatology", {}).get("compatibility", {})
        anomaly = details.get("anomaly", {})
        rows.append(
            [
                {"pacific_context": "Pacific", "humboldt_coastal": "Humboldt"}.get(domain_id, domain_id),
                details.get("sst_mode"),
                str(compatibility.get("climatology_mode", "")).replace("synthetic_demonstration", "synthetic demo"),
                str(compatibility.get("method", "")).replace("daily_smoothed", "daily"),
                (
                    "—"
                    if not compatibility.get("source_resolution")
                    else f"{float(compatibility['source_resolution'][0]):.2f}°"
                ),
                (
                    "—"
                    if compatibility.get("coverage") is None
                    else f"{float(compatibility['coverage']):.0%}"
                ),
                anomaly.get("status"),
            ]
        )
    table = summary_axis.table(
        cellText=rows,
        colLabels=["Domain", "SST mode", "Climate mode", "Method", "Resolution", "Coverage", "Anomaly"],
        loc="center",
        cellLoc="left",
    )
    table.auto_set_font_size(False)
    table.set_fontsize(7.5)
    table.scale(1.0, 1.55)
    figure.suptitle(
        "Humboldt Ocean Watch — multidomain anomaly engine",
        fontsize=16,
        weight="bold",
        y=0.985,
    )
    figure.text(
        0.5,
        0.955,
        "SYNTHETIC DEMONSTRATION PRODUCTS — NOT OBSERVATIONS",
        ha="center",
        color="#9A3412",
        weight="bold",
    )
    figure.subplots_adjust(hspace=0.30, wspace=0.16, top=0.91, bottom=0.08)
    return figure, foundation


def build_anomaly_preview(config: dict, *, overwrite: bool = False) -> dict:
    outputs = config["sst"]["anomaly_outputs"]
    processed = resolve_project_path(outputs["processed_directory"])
    paths = {
        "pacific": processed / "pacific_context_latest_anomaly.nc",
        "coastal": processed / "humboldt_coastal_latest_anomaly.nc",
        "regional": resolve_project_path(outputs["regional_indices"]),
        "status": resolve_project_path(outputs["status"]),
        "figure": resolve_project_path(outputs["preview_figure"]),
        "report": resolve_project_path(outputs["preview_report"]),
    }
    missing = [
        str(path)
        for key, path in paths.items()
        if key not in {"figure", "report"} and not path.exists()
    ]
    if missing:
        raise FileNotFoundError("Required local anomaly products are missing: " + ", ".join(missing))
    with xr.open_dataset(paths["pacific"]) as opened:
        pacific = opened.load()
    with xr.open_dataset(paths["coastal"]) as opened:
        coastal = opened.load()
    regional = pd.read_parquet(paths["regional"])
    status = json.loads(paths["status"].read_text(encoding="utf-8"))
    figure, foundation = create_anomaly_preview_figure(
        config, pacific, coastal, regional, status
    )
    width, height = figure.canvas.get_width_height()
    report = {
        "validation": "complete",
        "offline": True,
        "compatibility": {
            key: value.get("climatology", {}).get("compatibility", {})
            for key, value in status.get("domains", {}).items()
        },
        "source_modes": {
            key: value.get("sst_mode") for key, value in status.get("domains", {}).items()
        },
        "methods": {
            key: value.get("anomaly", {}).get("climatology_method")
            for key, value in status.get("domains", {}).items()
        },
        "reference_periods": {
            key: value.get("anomaly", {}).get("reference_period")
            for key, value in status.get("domains", {}).items()
        },
        "resolutions": {
            key: value.get("climatology", {}).get("compatibility", {}).get("source_resolution")
            for key, value in status.get("domains", {}).items()
        },
        "coordinate_match": {
            key: value.get("climatology", {}).get("compatibility", {}).get("coordinate_match")
            for key, value in status.get("domains", {}).items()
        },
        "coverage": {
            key: value.get("climatology", {}).get("compatibility", {}).get("coverage")
            for key, value in status.get("domains", {}).items()
        },
        "missing_variables": {
            key: [
                operation
                for operation in ("z_score", "percentile_exceedance")
                if operation not in value.get("climatology", {}).get("compatibility", {}).get("allowed_operations", [])
            ]
            for key, value in status.get("domains", {}).items()
        },
        "regional_anomaly_status": status.get("regional_anomaly_status", {}),
        "coastal_corridor": foundation.corridor_status,
        "preview_dimensions_pixels": [int(width), int(height)],
        "four_panels": True,
        "diverging_scale_center_c": 0.0,
        "warnings": list(status.get("warnings", [])) + list(foundation.warnings),
        "errors": list(status.get("errors", [])),
    }
    try:
        _promote_atomically(
            [
                (paths["figure"], lambda path: figure.savefig(path, dpi=120, facecolor="white")),
                (
                    paths["report"],
                    lambda path: path.write_text(
                        json.dumps(report, indent=2, ensure_ascii=False) + "\n",
                        encoding="utf-8",
                    ),
                ),
            ],
            overwrite=overwrite,
        )
    finally:
        plt.close(figure)
    return {**report, "figure_path": str(paths["figure"]), "report_path": str(paths["report"])}


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render prepared offline anomaly products.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    report = build_anomaly_preview(load_config(), overwrite=args.overwrite)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
