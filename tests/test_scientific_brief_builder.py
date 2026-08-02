import json
from types import SimpleNamespace
from pathlib import Path

import pytest

import scripts.generate_scientific_brief as generation_cli
from scripts.render_scientific_brief import render_file
from scripts.validate_scientific_brief import validate_file
from src.brief_generator import ScientificBriefGenerationError, dry_run_scientific_brief
from src.utils import load_config
from tests.scientific_brief_test_data import ROOT, valid_brief


def test_standalone_validator_and_renderer_never_need_api_key(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    brief = tmp_path / "brief.json"
    brief.write_text(json.dumps(valid_brief("en"), ensure_ascii=False), encoding="utf-8")
    report, validation_path = validate_file(
        brief,
        ROOT / "outputs/briefs/brief_context_latest.json",
        language="en",
        output_path=tmp_path / "validation.json",
    )
    assert report["final_status"] == "passed"
    assert validation_path.exists()
    markdown = render_file(brief, tmp_path / "brief.md")
    assert markdown.exists()
    assert "## Executive summary" in markdown.read_text(encoding="utf-8")


def test_dry_run_preview_contains_sanitized_request_package(tmp_path) -> None:
    config = load_config(ROOT / "config.yaml")
    config["brief_generation"]["debug_directory"] = str(tmp_path / "debug")
    preview, path = dry_run_scientific_brief(
        context_path=ROOT / "outputs/briefs/brief_context_latest.json",
        config=config, root=ROOT, language="both", analysis_date=None,
        provider_name="openai",
        model="gpt-5.6", reasoning_effort="medium", text_verbosity="medium",
        output_directory=tmp_path, overwrite=False,
    )
    assert set(preview["requests"]) == {"es", "en"}
    assert preview["structured_output"]["strict"] is True
    assert "disclaimer" in preview["structured_output"]["strict_local_schema"]["required"]
    assert "disclaimer" not in preview["structured_output"]["schema"]["required"]
    assert preview["disclaimer_source"] == "application_constant"
    assert preview["disclaimer_inserted_before_strict_validation"] is True
    assert preview["model_generates_disclaimer"] is False
    assert preview["mandatory_operational_coverage_plan_included"] is True
    assert preview["mandatory_coverage_matrix_included"] is True
    assert preview["facts_used_source"] == "derived_from_claim_support"
    assert preview["final_mandatory_coverage_checklist_included"] is True
    assert preview["mandatory_operational_fact_count"] == 5
    assert len(preview["mandatory_operational_fact_ids"]) == 5
    assert preview["estimated_context_bytes"] > 0
    assert Path(path).exists()


def _generation_args():
    return SimpleNamespace(
        context=ROOT / "outputs/briefs/brief_context_latest.json",
        analysis_date=None,
        language="es",
        model="qwen3:4b",
        provider="ollama",
        base_url=None,
        reasoning_effort=None,
        text_verbosity=None,
        output_directory=None,
        dry_run=False,
        validate_only=False,
        overwrite=True,
        no_repair=False,
        config=ROOT / "config.yaml",
    )


def test_cli_returns_nonzero_after_json_validation_failure(monkeypatch) -> None:
    monkeypatch.setattr(
        generation_cli,
        "parse_args",
        _generation_args,
    )

    def fail_generation(**kwargs):
        raise ScientificBriefGenerationError(
            "output_validation_failed",
            "The JSON fallback response failed strict local validation",
        )

    monkeypatch.setattr(generation_cli, "generate_scientific_briefs", fail_generation)
    with pytest.raises(SystemExit) as caught:
        generation_cli.main()
    assert caught.value.code == 1


def test_cli_handles_ctrl_c_cleanly(monkeypatch) -> None:
    monkeypatch.setattr(generation_cli, "parse_args", _generation_args)

    def interrupt(**kwargs):
        raise KeyboardInterrupt()

    monkeypatch.setattr(generation_cli, "generate_scientific_briefs", interrupt)
    with pytest.raises(SystemExit) as caught:
        generation_cli.main()
    assert caught.value.code == 130
