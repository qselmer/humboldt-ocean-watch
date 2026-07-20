"""Build a date-specific snapshot from cached live SST or demo fallback."""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.daily_diagnosis import available_dates, diagnose_date, load_climatology
from src.data_loader import load_active_sst_dataset
from src.export_utils import export_json, export_netcdf
from src.plotting import plot_thermal_field, save_figure
from src.utils import load_config, resolve_project_path


def main() -> None:
    config = load_config()
    data = config["data"]
    region = config["region"]
    dataset, mode = load_active_sst_dataset(
        resolve_project_path(data["live_path"]),
        resolve_project_path(data["demo_path"]),
        aliases=data["sst_aliases"],
        demo_options={
            "longitude": tuple(region["longitude"]),
            "latitude": tuple(region["latitude"]),
            "days": data["demo_days"],
            "seed": data["demo_seed"],
        },
    )
    climatology_path = resolve_project_path(data["climatology_path"])
    climatology = load_climatology(str(climatology_path)) if climatology_path.exists() else None
    fields, report = diagnose_date(
        dataset,
        available_dates(dataset)[-1],
        climatology=climatology,
        data_mode=mode,
    )
    snapshot = export_netcdf(fields, resolve_project_path(data["processed_path"]))
    figures = config["outputs"]["figures"]
    save_figure(
        plot_thermal_field(fields.sst_current, title=f"Current SST — {report['date']}", colorbar_label="SST (°C)", cmap="turbo", vmin=18, vmax=32),
        resolve_project_path(figures["sst_current"]),
    )
    if "sst_anomaly" in fields:
        save_figure(plot_thermal_field(fields.sst_anomaly, title=f"SST anomaly — {report['date']}", colorbar_label="Anomaly (°C)", cmap="RdBu_r", vmin=-5, vmax=5), resolve_project_path(figures["sst_anomaly"]))
        save_figure(plot_thermal_field(fields.sst_z_score, title=f"Standardized anomaly — {report['date']}", colorbar_label="Z-score", cmap="RdBu_r", vmin=-3, vmax=3), resolve_project_path(figures["sst_zscore"]))
    report_path = export_json(report, resolve_project_path(config["outputs"]["metrics"]))
    print(f"Mode: {mode}")
    print(f"Snapshot: {snapshot}")
    print(f"Metrics: {report_path}")
    if report["warning"]:
        print(f"Warning: {report['warning']}")


if __name__ == "__main__":
    main()
