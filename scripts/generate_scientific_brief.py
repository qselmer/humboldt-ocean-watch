"""Generate validated bilingual scientific briefs with a selected LLM provider."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from src.brief_generator import (
    ScientificBriefGenerationError,
    dry_run_scientific_brief,
    generate_scientific_briefs,
    load_validated_context,
)
from src.utils import configure_logging, load_config, resolve_project_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--context", type=Path, default=Path("outputs/briefs/brief_context_latest.json"))
    parser.add_argument("--analysis-date")
    parser.add_argument("--language", choices=("es", "en", "both"), default=None)
    parser.add_argument("--model", default=None)
    parser.add_argument("--provider", choices=("gemini", "ollama", "openai"), default=None)
    parser.add_argument("--base-url", default=None)
    parser.add_argument("--reasoning-effort", default=None)
    parser.add_argument("--text-verbosity", choices=("low", "medium", "high"), default=None)
    parser.add_argument("--output-directory", type=Path)
    parser.add_argument("--dry-run", action="store_true")
    parser.add_argument("--validate-only", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no-repair", action="store_true")
    parser.add_argument("--config", type=Path, default=Path("config.yaml"))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config = load_config(resolve_project_path(args.config))
    configure_logging(config["logging"]["level"])
    settings = config["brief_generation"]
    language = args.language or str(settings["default_language"])
    context_path = resolve_project_path(args.context)
    output = resolve_project_path(args.output_directory) if args.output_directory else None
    try:
        if args.validate_only:
            validated = load_validated_context(
                context_path,
                settings=settings,
                expected_analysis_date=args.analysis_date,
            )
            print(f"Context validation: {validated.validation_result}")
            print(f"Analysis date: {validated.analysis_date}")
            print(f"Context SHA-256: {validated.sha256}")
            return
        if args.dry_run:
            preview, path = dry_run_scientific_brief(
                context_path=context_path,
                config=config,
                root=PROJECT_ROOT,
                language=language,
                analysis_date=args.analysis_date,
                provider_name=args.provider,
                model=args.model,
                base_url=args.base_url,
                reasoning_effort=args.reasoning_effort,
                text_verbosity=args.text_verbosity,
                output_directory=output,
                overwrite=args.overwrite,
            )
            print("Dry run: no provider request was made")
            print(f"Provider: {preview['provider']}")
            print(f"Model: {preview['model']}")
            print(f"Base URL: {preview['base_url'] or 'not applicable'}")
            print(f"Configured context length: {preview['num_ctx'] or 'provider managed'}")
            print(
                "Request timeout: "
                f"{preview['request_timeout_seconds'] or 'provider managed'} seconds"
            )
            print(f"Maximum transport attempts: {preview['maximum_transport_attempts']}")
            print(
                "Transport retry delays: initial="
                f"{preview['retry_initial_seconds']} seconds; maximum="
                f"{preview['retry_maximum_seconds']} seconds"
            )
            print(
                "Automatic function calling enabled: "
                f"{preview['automatic_function_calling_enabled']}"
            )
            print(f"Analysis date: {preview['analysis_date']}")
            print(f"Context bytes: {preview['estimated_context_bytes']}")
            print(f"Strict schema bytes: {preview['strict_schema_bytes']}")
            print(f"Provider schema bytes: {preview['provider_schema_bytes']}")
            print(f"Provider schema supplied: {preview['provider_schema_supplied']}")
            print(f"Strict local schema validation: {preview['strict_local_schema_validation']}")
            print(
                "Compact response contract included: "
                f"{preview['compact_response_contract_included']}"
            )
            print(
                "Compact response contract bytes: "
                f"{preview['compact_response_contract_bytes']}"
            )
            print(f"Disclaimer source: {preview['disclaimer_source']}")
            print(
                "Disclaimer inserted before strict validation: "
                f"{preview['disclaimer_inserted_before_strict_validation']}"
            )
            print(
                "Model generates disclaimer: "
                f"{preview['model_generates_disclaimer']}"
            )
            print(f"facts_used source: {preview['facts_used_source']}")
            print(
                "Mandatory operational coverage plan included: "
                f"{preview['mandatory_operational_coverage_plan_included']}"
            )
            print(
                "Mandatory coverage matrix included: "
                f"{preview['mandatory_coverage_matrix_included']}"
            )
            print(
                "Mandatory operational fact count: "
                f"{preview['mandatory_operational_fact_count']}"
            )
            print(
                "Mandatory operational fact IDs: "
                + ", ".join(preview["mandatory_operational_fact_ids"])
            )
            print(
                "Final mandatory coverage checklist included: "
                f"{preview['final_mandatory_coverage_checklist_included']}"
            )
            print(f"Removed schema keywords: {preview['removed_keyword_counts']}")
            print(f"Converted const keywords: {preview['converted_const_count']}")
            print(f"Resolved schema references: {preview['resolved_reference_count']}")
            print(f"Unresolved schema references: {preview['unresolved_reference_count']}")
            print(f"Provider schema validation: {preview['provider_schema_validation_result']}")
            print(f"Structured-output mode: {preview['structured_output_mode']}")
            print(f"Schema fallback used: {preview['schema_fallback_used']}")
            print(f"Fallback policy: {preview['schema_fallback_policy']}")
            for code, size in preview["estimated_request_bytes"].items():
                print(f"Estimated request bytes ({code}): {size}")
                print(
                    f"Approximate request tokens ({code}): "
                    f"{preview['approximate_request_tokens'][code]}"
                )
            print(f"Preview: {path}")
            return
        artifacts = generate_scientific_briefs(
            context_path=context_path,
            config=config,
            root=PROJECT_ROOT,
            language=language,
            analysis_date=args.analysis_date,
            provider_name=args.provider,
            model=args.model,
            base_url=args.base_url,
            reasoning_effort=args.reasoning_effort,
            text_verbosity=args.text_verbosity,
            output_directory=output,
            overwrite=args.overwrite,
            allow_repair=not args.no_repair,
        )
        print(f"Analysis date: {artifacts.analysis_date}")
        print(f"Languages: {', '.join(artifacts.languages)}")
        for code, result in artifacts.generations.items():
            metadata = result.metadata
            print(
                f"{code}: {result.validation.final_status}; "
                f"input={metadata.input_tokens}; output={metadata.output_tokens}; total={metadata.total_tokens}"
            )
        for name, path in artifacts.paths.items():
            print(f"{name}: {path}")
    except (ScientificBriefGenerationError, FileNotFoundError, FileExistsError, ValueError) as exc:
        logging.getLogger("humboldt_ocean_watch.scientific_brief").error("%s", exc)
        raise SystemExit(1) from None
    except OSError:
        logging.getLogger("humboldt_ocean_watch.scientific_brief").error(
            "A scientific-brief file could not be read or written"
        )
        raise SystemExit(1) from None
    except KeyboardInterrupt:
        logging.getLogger("humboldt_ocean_watch.scientific_brief").warning(
            "Scientific-brief generation interrupted; no partial brief was published"
        )
        raise SystemExit(130) from None


if __name__ == "__main__":
    main()
