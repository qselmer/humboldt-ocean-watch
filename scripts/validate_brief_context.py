"""Validate a Humboldt Ocean Watch scientific-brief context JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.brief_validation import validate_brief_context
from src.export_utils import dumps_json_safe
from src.utils import load_config, resolve_project_path


def validate_file(
    input_path: Path,
    *,
    config_path: Path = Path("config.yaml"),
    output_path: Path | None = None,
    strict: bool | None = None,
) -> tuple[dict, Path]:
    source = resolve_project_path(input_path)
    if not source.exists():
        raise FileNotFoundError(f"Brief context not found: {source}")
    context = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(context, dict):
        raise ValueError("Brief context JSON must contain an object")
    config = load_config(resolve_project_path(config_path))
    settings = config["brief_context"]
    effective_strict = bool(settings["strict_validation"] if strict is None else strict)
    report = validate_brief_context(
        context,
        maximum_payload_kb=int(settings["maximum_payload_kb"]),
        strict=effective_strict,
    )
    selected = context.get("analysis", {}).get("resolved_analysis_date", "unknown-date")
    destination = resolve_project_path(
        output_path
        or Path(settings["output_directory"]) / f"brief_context_{selected}_validation.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    temporary = destination.with_name(f".{destination.name}.tmp")
    try:
        temporary.write_text(dumps_json_safe(report.to_dict()) + "\n", encoding="utf-8")
        json.loads(temporary.read_text(encoding="utf-8"))
        temporary.replace(destination)
    finally:
        temporary.unlink(missing_ok=True)
    return report.to_dict(), destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, required=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--output", type=Path)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=None)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    report, destination = validate_file(
        args.input, config_path=args.config, output_path=args.output, strict=args.strict,
    )
    print(f"Validation result: {report['result']}")
    print(f"Report: {destination}")
    print(f"Payload: {report['payload_bytes']} bytes")
    if report["result"] == "failed" and report["strict"]:
        for issue in report["errors"]:
            print(f"ERROR {issue['code']}: {issue['message']}")
        raise SystemExit(1)


if __name__ == "__main__":
    main()
