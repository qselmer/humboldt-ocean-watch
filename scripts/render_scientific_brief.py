"""Render deterministic Markdown from a schema-valid scientific brief JSON."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys
import tempfile

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.brief_renderer import render_scientific_brief_markdown
from src.utils import resolve_project_path


def render_file(source_path: Path, output_path: Path, *, overwrite: bool = False) -> Path:
    source = resolve_project_path(source_path)
    destination = resolve_project_path(output_path)
    if destination.exists() and not overwrite:
        raise FileExistsError(f"Output exists; use --overwrite: {destination}")
    payload = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise ValueError("Scientific brief JSON must contain an object")
    markdown = render_scientific_brief_markdown(payload)
    destination.parent.mkdir(parents=True, exist_ok=True)
    descriptor, name = tempfile.mkstemp(prefix=f".{destination.stem}-", suffix=".md", dir=destination.parent)
    try:
        import os
        os.close(descriptor)
        temporary = Path(name)
        temporary.write_text(markdown, encoding="utf-8")
        temporary.replace(destination)
    finally:
        Path(name).unlink(missing_ok=True)
    return destination


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--brief-json", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    try:
        path = render_file(args.brief_json, args.output, overwrite=args.overwrite)
    except (FileNotFoundError, FileExistsError, ValueError, json.JSONDecodeError) as exc:
        print(f"Markdown rendering failed: {exc}")
        raise SystemExit(1) from None
    except OSError:
        print("Markdown input or output could not be read or written")
        raise SystemExit(1) from None
    print(f"Markdown: {path}")


if __name__ == "__main__":
    main()
