"""Check OpenAI SDK, credential, model, Responses API, and output access."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.openai_preflight import run_openai_preflight
from src.utils import load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    parser.add_argument("--model")
    parser.add_argument("--output-directory", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(resolve_project_path(args.config))
    settings = config["brief_generation"]
    provider_settings = settings["providers"]["openai"]
    directory = resolve_project_path(args.output_directory or settings["output_directory"])
    report = run_openai_preflight(
        configured_model=str(provider_settings["model"]),
        model_override=args.model,
        output_directory=directory,
    )
    print(f"Preflight status: {report.status}")
    print(f"OpenAI SDK: {report.sdk_version or 'unavailable'}")
    print(f"API key: {report.api_key_status}")
    print(f"Selected model: {report.selected_model or args.model or provider_settings['model']}")
    print(f"Model accessible: {report.model_accessible}")
    print(f"Responses API available: {report.responses_api_available}")
    print(f"Output directory writable: {report.output_directory_writable}")
    for message in report.messages:
        print(message)
    if report.status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
