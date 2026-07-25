"""Validate one structured scientific brief without calling OpenAI."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.brief_claim_validation import validate_scientific_brief
from src.brief_generator import load_validated_context
from src.export_utils import dumps_json_safe
from src.utils import load_config, resolve_project_path


def validate_file(
    brief_path: Path,
    context_path: Path,
    *,
    language: str,
    config_path: Path = Path("config.yaml"),
    output_path: Path | None = None,
) -> tuple[dict, Path]:
    config = load_config(resolve_project_path(config_path))
    settings = config["brief_generation"]
    context = load_validated_context(resolve_project_path(context_path), settings=settings)
    source = resolve_project_path(brief_path)
    if not source.exists():
        raise FileNotFoundError(f"Scientific brief not found: {source}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Scientific brief JSON must contain an object")
    report = validate_scientific_brief(
        payload,
        context.payload,
        language=language,
        settings=settings,
    ).to_dict()
    destination = resolve_project_path(
        output_path or source.with_name(f"{source.stem}_validation.json")
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.stem}-", suffix=".json", dir=destination.parent)
    os.close(descriptor)
    try:
        Path(name).write_text(dumps_json_safe(report) + "\n", encoding="utf-8")
        Path(name).replace(destination)
        return report, destination
    finally:
        Path(name).unlink(missing_ok=True)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brief", type=Path, required=True)
    parser.add_argument("--context", type=Path, default=Path("outputs/briefs/brief_context_latest.json"))
    parser.add_argument("--language", choices=("es", "en"), required=True)
    parser.add_argument("--validation-output", type=Path)
    parser.add_argument("--strict", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        report, destination = validate_file(
            args.brief,
            args.context,
            language=args.language,
            config_path=args.config,
            output_path=args.validation_output,
        )
    except (FileNotFoundError, ValueError, json.JSONDecodeError) as exc:
        print(f"Validation could not run: {exc}")
        raise SystemExit(1) from None
    except OSError:
        print("Validation input or output could not be read or written")
        raise SystemExit(1) from None
    print(f"Validation status: {report['final_status']}")
    print(f"Report: {destination}")
    if report["final_status"] == "failed" and args.strict:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
