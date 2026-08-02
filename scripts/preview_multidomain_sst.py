"""Render a fully offline, review-oriented multidomain SST preview."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
from uuid import uuid4

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Rectangle
import numpy as np
import pandas as pd
import xarray as xr

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.geographic_foundation import prepare_geographic_foundation
from src.sst_domains import load_sst_domain_specs
from src.utils import load_config, resolve_project_path


def _iter_parts(geometry, accepted: set[str]):
    if geometry.geom_type in accepted:
        yield geometry
    else:
        for part in getattr(geometry, "geoms", ()):
            yield from _iter_parts(part, accepted)


def _fill_geometry(axis, geometry, *, facecolor, edgecolor, alpha, zorder, linewidth=0.6):
    for polygon in _iter_parts(geometry, {"Polygon"}):
        x, y = polygon.exterior.xy
        axis.fill(
            x, y, facecolor=facecolor, edgecolor=edgecolor, alpha=alpha,
            zorder=zorder, linewidth=linewidth,
        )


def _draw_lines(axis, geometry, *, color="black", linewidth=0.6, zorder=8):
    for line in _iter_parts(geometry, {"LineString", "LinearRing"}):
        x, y = line.xy
        axis.plot(x, y, color=color, linewidth=linewidth, zorder=zorder)


def _plot_sst(axis, dataset: xr.Dataset, *, title: str, bounds, figure):
    field = dataset.sst.isel(time=-1)
    mesh = axis.pcolormesh(
        dataset.longitude,
        dataset.latitude,
        field,
        shading="auto",
        cmap="turbo",
        vmin=10.0,
        vmax=32.0,
        rasterized=True,
        zorder=1,
    )
    axis.set(xlim=bounds.longitude, ylim=bounds.latitude, xlabel="Longitude (°E)", ylabel="Latitude (°N)")
    axis.grid(color="0.7", linestyle=":", linewidth=0.5)
    axis.set_title(title, loc="left", fontsize=12, weight="bold")
    colorbar = figure.colorbar(mesh, ax=axis, pad=0.015, shrink=0.92)
    colorbar.set_label("SST (°C)")


def create_preview_figure(
    config: dict,
    pacific: xr.Dataset,
    coastal: xr.Dataset,
    indices: pd.DataFrame,
):
    specs = load_sst_domain_specs(config).domains
    foundation = prepare_geographic_foundation(config)
    figure = plt.figure(figsize=(19.2, 12.0), dpi=120, facecolor="white")
    grid = figure.add_gridspec(2, 2, height_ratios=(1.25, 0.75), hspace=0.28, wspace=0.16)
    pacific_axis = figure.add_subplot(grid[0, 0])
    coastal_axis = figure.add_subplot(grid[0, 1])
    series_axis = figure.add_subplot(grid[1, :])

    pacific_mode = str(pacific.attrs.get("source_mode", "unknown"))
    coastal_mode = str(coastal.attrs.get("source_mode", "unknown"))
    pacific_date = pd.Timestamp(pacific.time.values[-1]).date().isoformat()
    coastal_date = pd.Timestamp(coastal.time.values[-1]).date().isoformat()
    _plot_sst(
        pacific_axis,
        pacific,
        title=f"A. Pacific context latest SST — {pacific_date} [{pacific_mode}]",
        bounds=specs["pacific_context"].bounds,
        figure=figure,
    )
    geography = foundation.registry
    for region in geography.standard_regions.values():
        pacific_axis.add_patch(
            Rectangle(
                (region.bounds.west, region.bounds.south),
                region.bounds.width,
                region.bounds.height,
                facecolor="none",
                edgecolor="#9A3412",
                linewidth=1.5,
                zorder=10,
            )
        )
        pacific_axis.text(
            region.central_longitude,
            region.bounds.north + 0.8,
            region.label,
            ha="center",
            color="#9A3412",
            fontsize=8,
            weight="bold",
            zorder=11,
            bbox={"facecolor": "white", "edgecolor": "#9A3412", "alpha": 0.78, "pad": 1.5},
        )

    _plot_sst(
        coastal_axis,
        coastal,
        title=f"B. Humboldt coastal latest SST — {coastal_date} [{coastal_mode}]",
        bounds=specs["humboldt_coastal"].bounds,
        figure=figure,
    )
    if foundation.corridor is not None:
        _fill_geometry(
            coastal_axis,
            foundation.corridor,
            facecolor="#10B981",
            edgecolor="#047857",
            alpha=0.32,
            zorder=5,
            linewidth=0.8,
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
                    linewidth=0.45,
                )
        _draw_lines(pacific_axis, local.display_coastline, linewidth=0.65, zorder=9)
        _draw_lines(coastal_axis, local.display_coastline, linewidth=0.75, zorder=9)
    coastal_axis.legend(
        handles=[
            Patch(
                facecolor="#10B981",
                edgecolor="#047857",
                alpha=0.32,
                label="60 nm mainland corridor (islands excluded)",
            )
        ],
        loc="lower left",
        fontsize=8,
        framealpha=0.95,
    )

    colors = {"nino34": "#2563EB", "nino3": "#D97706", "nino12": "#DC2626"}
    for region_id, group in indices.groupby("region_id", sort=False):
        ordered = group.sort_values("date")
        series_axis.plot(
            pd.to_datetime(ordered.date),
            ordered.mean_sst_c,
            marker="o",
            linewidth=1.8,
            markersize=4,
            label=str(ordered.region_label.iloc[0]),
            color=colors.get(region_id),
        )
    series_axis.set_title("C. Niño regional mean SST series (daily source-grid means)", loc="left", fontsize=12, weight="bold")
    series_axis.set_xlabel("Date")
    series_axis.set_ylabel("Mean SST (°C)")
    series_axis.grid(color="0.75", linestyle=":", linewidth=0.6)
    series_axis.legend(ncol=3, frameon=False)
    series_axis.text(
        0.0,
        -0.23,
        "Anomalies and official ENSO classifications are not shown: compatible regional climatologies are not available.",
        transform=series_axis.transAxes,
        fontsize=9,
        color="#7C2D12",
    )

    modes = {pacific_mode, coastal_mode}
    demo_note = (
        "SYNTHETIC DEMONSTRATION DATA — NOT CURRENT OCEAN CONDITIONS"
        if "demo" in modes
        else "LOCAL CACHED SST DATA"
    )
    figure.suptitle("Humboldt Ocean Watch — multidomain SST foundation", fontsize=16, weight="bold", y=0.985)
    figure.text(0.5, 0.955, demo_note, ha="center", color="#9A3412", fontsize=11, weight="bold")
    return figure, foundation


def build_preview(config: dict, *, overwrite: bool = False) -> dict:
    outputs = config["sst"]["outputs"]
    processed = resolve_project_path(outputs["processed_directory"])
    paths = {
        "pacific": processed / "pacific_context_latest.nc",
        "coastal": processed / "humboldt_coastal_latest.nc",
        "indices": resolve_project_path(config["sst"]["regional_indices"]["output"]),
        "snapshot": resolve_project_path(outputs["snapshot"]),
        "figure": resolve_project_path(outputs["preview_figure"]),
        "report": resolve_project_path(outputs["preview_report"]),
    }
    missing = [str(path) for key, path in paths.items() if key not in {"figure", "report"} and not path.exists()]
    if missing:
        raise FileNotFoundError("Required local products are missing: " + ", ".join(missing))
    existing = [paths[key] for key in ("figure", "report") if paths[key].exists()]
    if existing and not overwrite:
        raise FileExistsError("Preview products exist; use --overwrite: " + ", ".join(map(str, existing)))
    with xr.open_dataset(paths["pacific"]) as opened:
        pacific = opened.load()
    with xr.open_dataset(paths["coastal"]) as opened:
        coastal = opened.load()
    indices = pd.read_parquet(paths["indices"])
    snapshot = json.loads(paths["snapshot"].read_text(encoding="utf-8"))
    figure, foundation = create_preview_figure(config, pacific, coastal, indices)
    width, height = figure.canvas.get_width_height()
    warnings = list(snapshot.get("warnings", [])) + list(foundation.warnings)
    report = {
        "validation_status": "valid",
        "domains_loaded": snapshot.get("domains_loaded", []),
        "source_modes": snapshot.get("source_modes", {}),
        "latest_dates": snapshot.get("latest_dates", {}),
        "domains": snapshot.get("domains", {}),
        "nino_region_coverage": snapshot.get("nino_region_coverage", {}),
        "anomalies_calculated": False,
        "anomaly_status": "not_calculated_no_compatible_climatology",
        "coastal_corridor": foundation.corridor_status,
        "preview_dimensions_pixels": [int(width), int(height)],
        "synthetic_demo_warning_visible": "demo" in set(snapshot.get("source_modes", {}).values()),
        "warnings": warnings,
        "errors": [],
    }
    token = uuid4().hex
    figure_path = paths["figure"]
    report_path = paths["report"]
    figure_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    temp_figure = figure_path.with_name(f".{figure_path.stem}.{token}.tmp.png")
    temp_report = report_path.with_name(f".{report_path.stem}.{token}.tmp.json")
    try:
        figure.savefig(temp_figure, dpi=120, facecolor="white")
        temp_report.write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        temp_figure.replace(figure_path)
        temp_report.replace(report_path)
    finally:
        plt.close(figure)
        for temporary in (temp_figure, temp_report):
            if temporary.exists():
                temporary.unlink()
    report["figure_path"] = str(figure_path)
    report["report_path"] = str(report_path)
    return report


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Render an offline multidomain SST preview.")
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> dict:
    args = parse_args(argv)
    report = build_preview(load_config(), overwrite=args.overwrite)
    print(json.dumps(report, indent=2, ensure_ascii=False))
    return report


if __name__ == "__main__":
    main()
