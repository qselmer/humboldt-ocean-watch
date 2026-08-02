"""Run a provider-neutral scientific-brief access preflight."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.llm_provider import get_llm_provider
from src.utils import load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--provider", choices=("gemini", "ollama", "openai"), default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(resolve_project_path(args.config))
    settings = config["brief_generation"]
    provider = get_llm_provider(args.provider, args.model, settings, base_url=args.base_url)
    output = resolve_project_path(settings["output_directory"])
    report = provider.preflight(output_directory=output)
    data = report.to_dict()
    print(f"Provider: {provider.provider_name}")
    print(f"Model: {provider.model_name}")
    print(f"Base URL: {provider.base_url or 'not applicable'}")
    if provider.provider_name == "gemini":
        print(f"Gemini package available: {data['package_available']}")
        print(f"API key available: {data['api_key_available']}")
        print(f"API reachable: {data['api_reachable']}")
        print(f"Model accessible: {data['model_accessible']}")
        print(f"Basic generation available: {data['basic_generation_available']}")
        print(f"Structured generation available: {data['structured_generation_available']}")
        print(f"Output directory writable: {data['output_directory_writable']}")
    elif provider.provider_name == "ollama":
        print(f"Ollama package available: {data['package_available']}")
        print(f"Ollama reachable: {data['server_reachable']}")
        print(f"Model installed: {data['model_installed']}")
        print(f"Basic generation available: {data['basic_generation_available']}")
        print(f"Structured generation available: {data['structured_generation_available']}")
        print(f"Output directory writable: {data['output_directory_writable']}")
        print(f"Configured context length: {data['configured_context_length']}")
    else:
        print(f"Model accessible: {data['model_accessible']}")
        print(f"Structured generation available: {data['responses_api_available']}")
        print(f"Output directory writable: {data['output_directory_writable']}")
    print(f"Preflight status: {report.status}")
    for item in data.get("errors", []):
        print(f"{item['code']}: {item['message']}")
    for message in data.get("messages", []):
        print(message)
    if report.status != "passed":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
