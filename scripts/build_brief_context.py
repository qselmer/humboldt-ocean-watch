"""Build an offline, validated scientific-brief context from cached products."""

from __future__ import annotations

import argparse
from dataclasses import dataclass
import json
import logging
import os
from pathlib import Path
import sys
import tempfile
from typing import Any

import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.brief_context import BriefBuildInputs, build_brief_context, input_paths_from_config
from src.brief_validation import BriefValidationReport, validate_brief_context
from src.export_utils import dumps_json_safe
from src.utils import configure_logging, load_config, resolve_project_path


LOGGER = logging.getLogger("humboldt_ocean_watch.brief_context")


@dataclass(frozen=True)
class BriefArtifacts:
    context_path: Path
    latest_path: Path | None
    validation_path: Path
    facts_path: Path
    context: dict[str, Any]
    validation: BriefValidationReport


def _temporary(destination: Path) -> Path:
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(
        prefix=f".{destination.stem}-", suffix=destination.suffix, dir=destination.parent,
    )
    os.close(descriptor)
    return Path(name)


def _atomic_outputs(
    context: dict[str, Any],
    report: BriefValidationReport,
    *,
    context_path: Path,
    latest_path: Path | None,
    validation_path: Path,
    facts_path: Path,
    overwrite: bool,
) -> None:
    destinations = [context_path, validation_path, facts_path]
    if latest_path is not None and latest_path != context_path:
        destinations.append(latest_path)
    existing = [path for path in destinations if path.exists()]
    if existing and not overwrite:
        raise FileExistsError(
            "Brief output exists; use --overwrite: " + ", ".join(str(path) for path in existing)
        )
    temporary = {path: _temporary(path) for path in destinations}
    try:
        context_text = dumps_json_safe(context) + "\n"
        temporary[context_path].write_text(context_text, encoding="utf-8")
        temporary[validation_path].write_text(dumps_json_safe(report.to_dict()) + "\n", encoding="utf-8")
        pd.DataFrame(context["fact_registry"]).to_parquet(temporary[facts_path], index=False)
        if latest_path is not None and latest_path != context_path:
            temporary[latest_path].write_text(context_text, encoding="utf-8")

        parsed = json.loads(temporary[context_path].read_text(encoding="utf-8"))
        if parsed.get("analysis", {}).get("resolved_analysis_date") != context["analysis"]["resolved_analysis_date"]:
            raise RuntimeError("Temporary brief context failed date validation")
        if len(pd.read_parquet(temporary[facts_path])) != len(context["fact_registry"]):
            raise RuntimeError("Temporary fact table failed row-count validation")
        for destination in destinations:
            temporary[destination].replace(destination)
    finally:
        for path in temporary.values():
            path.unlink(missing_ok=True)


def build(
    *,
    config_path: Path,
    analysis_date: str | None,
    output_directory: Path | None,
    input_overrides: dict[str, Path | None] | None = None,
    strict: bool | None = None,
    overwrite: bool = False,
) -> BriefArtifacts:
    config = load_config(resolve_project_path(config_path))
    configure_logging(config["logging"]["level"])
    settings = config["brief_context"]
    requested = analysis_date or str(settings["default_analysis_date"])
    effective_strict = bool(settings["strict_validation"] if strict is None else strict)
    inputs: BriefBuildInputs = input_paths_from_config(config, PROJECT_ROOT, input_overrides)
    LOGGER.info("Building local brief context for %s", requested)
    context, _ = build_brief_context(
        config, analysis_date=requested, root=PROJECT_ROOT, inputs=inputs,
    )
    report = validate_brief_context(
        context,
        maximum_payload_kb=int(settings["maximum_payload_kb"]),
        strict=effective_strict,
    )
    if report.result == "failed":
        messages = "; ".join(issue["message"] for issue in report.errors)
        qualifier = "Strict " if effective_strict else ""
        raise ValueError(f"{qualifier}brief-context validation failed: {messages}")

    directory = resolve_project_path(output_directory or settings["output_directory"])
    selected = context["analysis"]["resolved_analysis_date"]
    dated = directory / f"brief_context_{selected}.json"
    validation = directory / f"brief_context_{selected}_validation.json"
    facts = directory / f"brief_facts_{selected}.parquet"
    latest = (
        directory / "brief_context_latest.json"
        if selected == context["analysis"]["available_end_date"]
        else None
    )
    _atomic_outputs(
        context, report,
        context_path=dated, latest_path=latest, validation_path=validation,
        facts_path=facts, overwrite=overwrite,
    )
    LOGGER.info(
        "Wrote %s (%d bytes, %d facts)",
        dated,
        report.payload_bytes,
        len(context["fact_registry"]),
    )
    return BriefArtifacts(dated, latest, validation, facts, context, report)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--analysis-date", default=None)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--daily-metrics", type=Path)
    parser.add_argument("--qc-report", type=Path)
    parser.add_argument("--representativeness", type=Path)
    parser.add_argument("--series-bank", type=Path)
    parser.add_argument("--temporal-features", type=Path)
    parser.add_argument("--spatial-features", type=Path)
    parser.add_argument("--univariate-events", type=Path)
    parser.add_argument("--univariate-flags", type=Path)
    parser.add_argument("--daily-patches", type=Path)
    parser.add_argument("--daily-patch-summary", type=Path)
    parser.add_argument("--patch-observations", type=Path)
    parser.add_argument("--tracks", type=Path)
    parser.add_argument("--event-families", type=Path)
    parser.add_argument("--lineage-edges", type=Path)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=None)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    override_names = (
        "daily_metrics", "qc_report", "representativeness", "series_bank",
        "temporal_features", "spatial_features", "univariate_events",
        "univariate_flags", "daily_patches", "daily_patch_summary",
        "patch_observations", "tracks", "event_families", "lineage_edges",
    )
    overrides = {name: getattr(args, name) for name in override_names}
    artifacts = build(
        config_path=args.config,
        analysis_date=args.analysis_date,
        output_directory=args.output_directory,
        input_overrides=overrides,
        strict=args.strict,
        overwrite=args.overwrite,
    )
    print(f"Resolved analysis date: {artifacts.context['analysis']['resolved_analysis_date']}")
    print(f"Context: {artifacts.context_path}")
    if artifacts.latest_path:
        print(f"Latest: {artifacts.latest_path}")
    print(f"Validation: {artifacts.validation.result}")
    print(f"Payload: {artifacts.validation.payload_bytes} bytes")
    print(f"Facts: {len(artifacts.context['fact_registry'])}")


if __name__ == "__main__":
    main()
