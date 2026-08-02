"""Validate and render the offline Increment 6A geographic foundation."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from datetime import datetime, timezone
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.coastline_source import LocalCoastlineSource, LocalGeometries
from src.export_utils import dumps_json_safe
from src.geographic_foundation import prepare_geographic_foundation
from src.geographic_overlays import plot_geographic_foundation
from src.utils import configure_logging, load_config, resolve_project_path


DEFAULT_REPORT = Path("outputs/reports/geographic_foundation_preview.json")


@dataclass(frozen=True)
class GeographicFoundationArtifacts:
    output_path: Path
    report_path: Path
    metadata: dict[str, Any]


def _public_path(path: Path) -> str:
    try:
        return path.resolve().relative_to(PROJECT_ROOT.resolve()).as_posix()
    except ValueError:
        return path.name


def _temporary(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.stem}-", suffix=destination.suffix, dir=destination.parent
    )
    os.close(descriptor)
    return Path(name)


def _contains_absolute_path(value: Any) -> bool:
    """Return whether serializable metadata exposes a host-specific path."""
    if isinstance(value, dict):
        return any(_contains_absolute_path(item) for item in value.values())
    if isinstance(value, (list, tuple, set)):
        return any(_contains_absolute_path(item) for item in value)
    return isinstance(value, (str, Path)) and Path(str(value)).is_absolute()


def build_foundation_artifacts(
    *,
    config_path: Path,
    output_path: Path,
    report_path: Path = DEFAULT_REPORT,
    overwrite: bool = False,
    source: LocalCoastlineSource | None = None,
    local_geometries: LocalGeometries | None = None,
) -> GeographicFoundationArtifacts:
    """Build PNG and metadata atomically without network access."""
    config = load_config(resolve_project_path(config_path))
    configure_logging(config["logging"]["level"])
    output = resolve_project_path(output_path)
    report = resolve_project_path(report_path)
    existing = [path for path in (output, report) if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Geographic preview output exists; use --overwrite: "
            + ", ".join(_public_path(path) for path in existing)
        )

    foundation = prepare_geographic_foundation(
        config, source=source, local_geometries=local_geometries
    )
    figure = plot_geographic_foundation(
        foundation.registry,
        local_geometries=foundation.local_geometries,
        corridor=foundation.corridor,
        source_status=foundation.source_status,
    )

    metadata = {
        "generated_at": datetime.now(timezone.utc).isoformat(),
        "display_domain": foundation.registry.display_domain.to_dict(),
        "analysis_domains": {
            key: value.to_dict()
            for key, value in foundation.registry.analysis_domains.items()
        },
        "standard_regions": {
            key: value.to_dict()
            for key, value in foundation.registry.standard_regions.items()
        },
        "coastal_corridor_spec": foundation.registry.coastal_corridors[
            "humboldt_60nm"
        ].to_dict(),
        "coastline_source_status": foundation.source_status.to_dict(),
        "coastal_corridor_geometry_status": foundation.corridor_status,
        "output_path": _public_path(output),
        "validation_status": "passed",
        "warnings": list(foundation.warnings),
        "experimental_product": True,
        "scientific_calculation_domain_unchanged": True,
    }
    if _contains_absolute_path(metadata):
        raise RuntimeError("Geographic preview metadata contains a private absolute path")
    temporary_output = _temporary(output)
    temporary_report = _temporary(report)
    try:
        figure.savefig(
            temporary_output,
            dpi=170,
            facecolor="white",
            format="png",
        )
        plt.close(figure)
        temporary_report.write_text(dumps_json_safe(metadata) + "\n", encoding="utf-8")
        if temporary_output.stat().st_size <= 0:
            raise RuntimeError("Temporary geographic preview PNG is empty")
        temporary_output.replace(output)
        temporary_report.replace(report)
    finally:
        plt.close(figure)
        temporary_output.unlink(missing_ok=True)
        temporary_report.unlink(missing_ok=True)
    return GeographicFoundationArtifacts(output, report, metadata)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("outputs/figures/geographic_foundation_preview.png"),
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifacts = build_foundation_artifacts(
        config_path=args.config,
        output_path=args.output,
        overwrite=args.overwrite,
    )
    print(f"Geographic preview: {artifacts.output_path}")
    print(f"Metadata: {artifacts.report_path}")
    print(f"Validation: {artifacts.metadata['validation_status']}")
    for warning in artifacts.metadata["warnings"]:
        print(f"Warning: {warning}")


if __name__ == "__main__":
    main()
