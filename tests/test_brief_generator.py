from copy import deepcopy
import os
from pathlib import Path

import pytest

from src.brief_generator import (
    ScientificBriefGenerationError,
    _atomic_write_group,
    dry_run_scientific_brief,
    generate_scientific_briefs,
)
from src.brief_disclaimer import (
    MandatoryDisclaimerConfigurationError,
    get_mandatory_disclaimer,
)
from tests.scientific_brief_test_data import (
    FakeClient,
    ROOT,
    fake_response,
    valid_brief,
)
from src.utils import load_config
from src.llm_provider import get_llm_provider, resolve_provider_settings
from src.ollama_client import OllamaProvider
import json


CONTEXT = ROOT / "outputs/briefs/brief_context_latest.json"
LLM_ENVIRONMENT_VARIABLES = (
    "LLM_PROVIDER",
    "GEMINI_API_KEY",
    "GEMINI_MODEL",
    "GEMINI_REQUEST_TIMEOUT_SECONDS",
    "GEMINI_MAXIMUM_OUTPUT_TOKENS",
    "GEMINI_STRUCTURED_OUTPUT_MODE",
    "GEMINI_MAXIMUM_TRANSPORT_ATTEMPTS",
    "GEMINI_RETRY_INITIAL_SECONDS",
    "GEMINI_RETRY_MAXIMUM_SECONDS",
    "OLLAMA_MODEL",
    "OLLAMA_BASE_URL",
    "OLLAMA_NUM_CTX",
    "OLLAMA_REQUEST_TIMEOUT_SECONDS",
    "OLLAMA_STRUCTURED_OUTPUT_MODE",
    "OPENAI_MODEL",
    "OPENAI_API_KEY",
)


@pytest.fixture(autouse=True)
def _isolate_llm_environment(monkeypatch):
    """Keep ambient provider configuration out of generator unit tests."""
    for variable in LLM_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(variable, raising=False)


def _config():
    config = deepcopy(load_config(ROOT / "config.yaml"))
    config["brief_generation"]["provider"] = "openai"
    return config


def test_ambient_ollama_provider_cannot_override_isolated_openai_config() -> None:
    assert all(os.getenv(variable) is None for variable in LLM_ENVIRONMENT_VARIABLES)
    settings = _config()["brief_generation"]
    provider = get_llm_provider(None, None, settings, client=FakeClient([]))
    assert provider.provider_name == "openai"
    assert provider.model_name == "gpt-5.6"


def test_generation_writes_valid_dated_latest_and_metadata_files(tmp_path) -> None:
    client = FakeClient([fake_response(valid_brief("es"), response_id="resp_es"), fake_response(valid_brief("en"), response_id="resp_en")])
    artifacts = generate_scientific_briefs(
        context_path=CONTEXT, config=_config(), root=ROOT, language="both",
        output_directory=tmp_path, client=client,
    )
    assert len(client.responses.calls) == 2
    assert all(path.exists() for path in artifacts.paths.values())
    assert artifacts.cross_language_report["final_status"] == "passed"
    for call in client.responses.calls:
        assert call["store"] is False
        assert call["tools"] == []
        assert call["model"] == "gpt-5.6"
        assert "previous_response_id" not in call
    assert artifacts.generations["en"].metadata.total_tokens == 150


def test_one_bounded_repair_succeeds(tmp_path) -> None:
    invalid = deepcopy(valid_brief("en"))
    invalid.pop("title")
    client = FakeClient([fake_response(invalid), fake_response(valid_brief("en"))])
    artifacts = generate_scientific_briefs(
        context_path=CONTEXT, config=_config(), root=ROOT, language="en",
        output_directory=tmp_path, client=client,
    )
    assert len(client.responses.calls) == 2
    assert artifacts.generations["en"].metadata.repair_count == 1


def test_missing_operational_facts_trigger_one_content_repair_with_exact_details(
    tmp_path,
) -> None:
    missing = {
        "activity.active_family_count",
        "event.event_day",
        "patches.patch_count",
    }
    invalid = deepcopy(valid_brief("en"))
    paragraph = invalid["event_status"]["paragraphs"][0]
    paragraph["supporting_fact_ids"] = [
        fact_id
        for fact_id in paragraph["supporting_fact_ids"]
        if fact_id not in missing
    ]
    invalid["facts_used"] = [
        fact_id for fact_id in invalid["facts_used"] if fact_id not in missing
    ]
    client = FakeClient(
        [fake_response(invalid), fake_response(valid_brief("en"))]
    )

    artifacts = generate_scientific_briefs(
        context_path=CONTEXT,
        config=_config(),
        root=ROOT,
        language="en",
        output_directory=tmp_path,
        client=client,
    )

    assert len(client.responses.calls) == 2
    assert artifacts.generations["en"].metadata.repair_count == 1
    repair_request = client.responses.calls[1]["input"][1]["content"]
    for fact_id in missing:
        assert fact_id in repair_request
    assert '"value": 69' in repair_request
    assert '"unit": "day"' in repair_request
    assert '"precision": 3' in repair_request
    assert '"claim_support_field": "supporting_fact_ids"' in repair_request
    assert "complete corrected JSON document" in repair_request
    assert "Do not return a JSON Patch" in repair_request

    diagnostic_path = next(
        (tmp_path / "debug").glob("*_validation_attempt_1.json")
    )
    diagnostic_text = diagnostic_path.read_text(encoding="utf-8")
    diagnostic = json.loads(diagnostic_text)
    coverage_issues = [
        issue
        for issue in diagnostic["validation_issues"]
        if issue["code"] == "required_operational_fact_not_represented"
    ]
    assert {issue["fact_id"] for issue in coverage_issues} == missing
    assert all(issue["present_in_claim_support"] is False for issue in coverage_issues)
    assert "The univariate event, daily patch" not in diagnostic_text


def test_claim_supported_fact_is_added_to_derived_facts_used_without_repair(
    tmp_path,
) -> None:
    raw = deepcopy(valid_brief("en"))
    raw["facts_used"].remove("activity.active_family_count")
    original_text = raw["event_status"]["paragraphs"][0]["text"]
    original_claim_types = list(
        raw["event_status"]["paragraphs"][0]["claim_types"]
    )
    client = FakeClient([fake_response(raw)])

    artifacts = generate_scientific_briefs(
        context_path=CONTEXT,
        config=_config(),
        root=ROOT,
        language="en",
        output_directory=tmp_path,
        client=client,
    )
    result = artifacts.generations["en"]

    assert len(client.responses.calls) == 1
    assert "activity.active_family_count" in result.payload["facts_used"]
    assert result.payload["event_status"]["paragraphs"][0]["text"] == original_text
    assert (
        result.payload["event_status"]["paragraphs"][0]["claim_types"]
        == original_claim_types
    )
    assert result.metadata.repair_count == 0
    assert result.metadata.facts_used_source == "derived_from_claim_support"


def test_fact_only_in_root_index_is_removed_and_triggers_one_repair(tmp_path) -> None:
    invalid = deepcopy(valid_brief("en"))
    invalid["event_status"]["paragraphs"][0]["supporting_fact_ids"].remove(
        "activity.active_family_count"
    )
    assert "activity.active_family_count" in invalid["facts_used"]
    client = FakeClient(
        [fake_response(invalid), fake_response(valid_brief("en"))]
    )

    artifacts = generate_scientific_briefs(
        context_path=CONTEXT,
        config=_config(),
        root=ROOT,
        language="en",
        output_directory=tmp_path,
        client=client,
    )

    assert len(client.responses.calls) == 2
    assert artifacts.generations["en"].metadata.repair_count == 1
    repair_request = client.responses.calls[1]["input"][1]["content"]
    assert "activity.active_family_count" in repair_request
    assert "currently_present_in_claim_support" in repair_request
    diagnostic_path = next(
        (tmp_path / "debug").glob("*_validation_attempt_1.json")
    )
    diagnostic = json.loads(diagnostic_path.read_text(encoding="utf-8"))
    issue = next(
        item
        for item in diagnostic["validation_issues"]
        if item.get("fact_id") == "activity.active_family_count"
    )
    assert issue["present_in_facts_used"] is False
    assert issue["derived_facts_used_membership"] is False
    assert issue["repair_instruction_generated"] is True
    assert issue["claim_support_locations_inspected"]
    assert not any(issue["present_in_each_inspected_location"].values())


def test_disclaimer_only_mismatch_is_inserted_without_scientific_repair(tmp_path) -> None:
    raw = deepcopy(valid_brief("en"))
    paraphrase = "Experimental monitoring product; not an official category."
    raw["disclaimer"] = paraphrase
    client = FakeClient([fake_response(raw)])
    artifacts = generate_scientific_briefs(
        context_path=CONTEXT,
        config=_config(),
        root=ROOT,
        language="en",
        output_directory=tmp_path,
        client=client,
    )
    result = artifacts.generations["en"]
    metadata = result.metadata.to_dict()
    assert len(client.responses.calls) == 1
    assert result.payload["disclaimer"] == get_mandatory_disclaimer("en")
    assert result.validation.final_status == "passed"
    assert result.metadata.repair_count == 0
    assert metadata["scientific_repair_count"] == 0
    assert metadata["disclaimer_source"] == "application_constant"
    assert metadata["disclaimer_inserted"] is True
    assert metadata["disclaimer_language"] == "en"
    assert metadata["raw_model_disclaimer_present"] is True
    assert metadata["raw_model_disclaimer_matched"] is False
    assert paraphrase not in json.dumps(metadata)
    provider_schema = client.responses.calls[0]["text"]["format"]["schema"]
    assert "disclaimer" not in provider_schema["required"]
    assert "disclaimer" not in provider_schema["properties"]


def test_missing_disclaimer_is_inserted_before_validation(tmp_path) -> None:
    raw = deepcopy(valid_brief("es"))
    raw.pop("disclaimer")
    client = FakeClient([fake_response(raw)])
    artifacts = generate_scientific_briefs(
        context_path=CONTEXT, config=_config(), root=ROOT, language="es",
        output_directory=tmp_path, client=client,
    )
    result = artifacts.generations["es"]
    assert result.payload["disclaimer"] == get_mandatory_disclaimer("es")
    assert result.metadata.raw_model_disclaimer_present is False
    assert result.metadata.repair_count == 0
    assert len(client.responses.calls) == 1


def test_repaired_output_receives_disclaimer_before_final_validation(tmp_path) -> None:
    first = deepcopy(valid_brief("en"))
    first.pop("title")
    first["disclaimer"] = "wrong first disclaimer"
    repaired = deepcopy(valid_brief("en"))
    repaired["disclaimer"] = "wrong repaired disclaimer"
    client = FakeClient([fake_response(first), fake_response(repaired)])
    artifacts = generate_scientific_briefs(
        context_path=CONTEXT, config=_config(), root=ROOT, language="en",
        output_directory=tmp_path, client=client,
    )
    result = artifacts.generations["en"]
    assert len(client.responses.calls) == 2
    assert result.metadata.repair_count == 1
    assert result.payload["disclaimer"] == get_mandatory_disclaimer("en")
    assert result.validation.final_status == "passed"


def test_unavailable_disclaimer_fails_before_provider_call(monkeypatch, tmp_path) -> None:
    client = FakeClient([])
    existing = tmp_path / "scientific_brief_2026-07-19_en.md"
    existing.write_text("previous validated brief", encoding="utf-8")

    def unavailable(language: str) -> str:
        raise MandatoryDisclaimerConfigurationError(
            "The mandatory disclaimer configuration is unavailable"
        )

    monkeypatch.setattr("src.brief_generator.get_mandatory_disclaimer", unavailable)
    with pytest.raises(ScientificBriefGenerationError) as caught:
        generate_scientific_briefs(
            context_path=CONTEXT, config=_config(), root=ROOT, language="en",
            output_directory=tmp_path, client=client, overwrite=True,
        )
    assert caught.value.code == "mandatory_disclaimer_configuration_error"
    assert client.responses.calls == []
    assert not list(tmp_path.glob("scientific_brief_*.json"))
    assert existing.read_text(encoding="utf-8") == "previous validated brief"


def test_structurally_invalid_section_is_repaired_once_with_sanitized_diagnostic(tmp_path) -> None:
    invalid = deepcopy(valid_brief("en"))
    invalid["executive_summary"] = "regression payload matching the observed failure"
    client = FakeClient([fake_response(invalid), fake_response(valid_brief("en"))])

    artifacts = generate_scientific_briefs(
        context_path=CONTEXT,
        config=_config(),
        root=ROOT,
        language="en",
        output_directory=tmp_path,
        client=client,
    )

    assert len(client.responses.calls) == 2
    assert artifacts.generations["en"].metadata.repair_count == 1
    repair_request = client.responses.calls[1]["input"][1]["content"]
    assert "invalid_section_type" in repair_request
    assert "executive_summary" in repair_request

    diagnostic_path = next(
        (tmp_path / "debug").glob("*_validation_attempt_1.json")
    )
    diagnostic_text = diagnostic_path.read_text(encoding="utf-8")
    diagnostic = json.loads(diagnostic_text)
    assert diagnostic["provider"] == "openai"
    assert diagnostic["model"] == "gpt-5.6"
    assert diagnostic["validation_issue_codes"] == ["invalid_section_type"]
    assert diagnostic["json_paths"] == ["executive_summary"]
    assert diagnostic["repair_attempted"] is True
    assert len(diagnostic["response_sha256"]) == 64
    assert {
        (item["path"], item["type"])
        for item in diagnostic["response_structure_summary"]["entries"]
    } >= {("root.executive_summary", "string")}
    assert "regression payload matching the observed failure" not in diagnostic_text


def test_shape_diagnostic_and_repair_contract_are_specific_but_do_not_leak_prose(tmp_path) -> None:
    invalid = deepcopy(valid_brief("en"))
    sensitive_text = "generated prose credential prompt context must stay out of diagnostics"
    invalid["regional_state"] = {"text": sensitive_text}
    client = FakeClient([fake_response(invalid), fake_response(valid_brief("en"))])

    generate_scientific_briefs(
        context_path=CONTEXT,
        config=_config(),
        root=ROOT,
        language="en",
        output_directory=tmp_path,
        client=client,
    )

    diagnostic_path = next((tmp_path / "debug").glob("*_validation_attempt_1.json"))
    diagnostic_text = diagnostic_path.read_text(encoding="utf-8")
    diagnostic = json.loads(diagnostic_text)
    issue = next(
        item for item in diagnostic["validation_issues"]
        if item.get("unexpected_field") == "text"
    )
    assert issue["allowed_fields"] == ["heading", "paragraphs", "reason", "status"]
    assert issue["expected_shape"]["required"] == [
        "heading", "paragraphs", "status", "reason"
    ]
    assert sensitive_text not in diagnostic_text
    assert "SCIENTIFIC_CONTEXT_JSON" not in diagnostic_text

    repair_request = client.responses.calls[1]["input"][1]["content"]
    assert '"unexpected_field": "text"' in repair_request
    assert '"missing_field": "paragraphs"' in repair_request
    assert '"expected_type": "array"' in repair_request
    assert "complete corrected JSON document" in repair_request


def test_failed_structural_repair_preserves_output_and_publishes_no_markdown(tmp_path) -> None:
    existing = tmp_path / "scientific_brief_2026-07-19_en.md"
    existing.write_text("previous valid brief", encoding="utf-8")
    invalid = deepcopy(valid_brief("en"))
    invalid["executive_summary"] = "unexpected text"
    client = FakeClient([fake_response(invalid), fake_response(invalid)])

    with pytest.raises(ScientificBriefGenerationError) as caught:
        generate_scientific_briefs(
            context_path=CONTEXT,
            config=_config(),
            root=ROOT,
            language="en",
            output_directory=tmp_path,
            client=client,
            overwrite=True,
        )

    assert caught.value.code == "output_validation_failed"
    assert len(client.responses.calls) == 2
    assert existing.read_text(encoding="utf-8") == "previous valid brief"
    assert list(tmp_path.glob("scientific_brief_*.md")) == [existing]
    assert not (tmp_path / "scientific_brief_2026-07-19_en.json").exists()


def test_one_transient_retry_is_bounded(tmp_path) -> None:
    error_type = type("RateLimitError", (Exception,), {})
    transient = error_type("server detail must not be exposed")
    transient.status_code = 429
    client = FakeClient([transient, fake_response(valid_brief("en"))])
    delays: list[float] = []
    artifacts = generate_scientific_briefs(
        context_path=CONTEXT, config=_config(), root=ROOT, language="en",
        output_directory=tmp_path, client=client, sleep=delays.append,
    )
    assert len(client.responses.calls) == 2
    assert delays == [2.0]
    assert artifacts.generations["en"].metadata.retry_count == 1


def test_failed_repair_preserves_previous_valid_output(tmp_path) -> None:
    existing = tmp_path / "scientific_brief_2026-07-19_en.md"
    existing.write_text("previous valid brief", encoding="utf-8")
    invalid = deepcopy(valid_brief("en"))
    invalid.pop("title")
    client = FakeClient([fake_response(invalid), fake_response(invalid)])
    with pytest.raises(ScientificBriefGenerationError) as error:
        generate_scientific_briefs(
            context_path=CONTEXT, config=_config(), root=ROOT, language="en",
            output_directory=tmp_path, client=client, overwrite=True,
        )
    assert error.value.code == "output_validation_failed"
    assert len(client.responses.calls) == 2
    assert existing.read_text(encoding="utf-8") == "previous valid brief"
    assert list((tmp_path / "debug").glob("*_failure.json"))


def test_existing_output_prevents_any_api_call(tmp_path) -> None:
    (tmp_path / "scientific_brief_generation_summary.json").write_text("{}", encoding="utf-8")
    client = FakeClient([])
    with pytest.raises(FileExistsError):
        generate_scientific_briefs(
            context_path=CONTEXT, config=_config(), root=ROOT, language="en",
            output_directory=tmp_path, client=client,
        )
    assert client.responses.calls == []


def test_dry_run_does_not_require_key_or_call_client(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = _config()
    config["brief_generation"]["debug_directory"] = str(tmp_path / "debug")
    preview, path = dry_run_scientific_brief(
        context_path=CONTEXT, config=config, root=ROOT, language="both",
        analysis_date=None, model=None, reasoning_effort=None, text_verbosity=None,
        output_directory=tmp_path, overwrite=False,
    )
    assert preview["dry_run"] is True
    assert preview["model"] == "gpt-5.6"
    assert preview["store"] is False
    assert preview["mandatory_operational_coverage_plan_included"] is True
    assert preview["mandatory_coverage_matrix_included"] is True
    assert preview["facts_used_source"] == "derived_from_claim_support"
    assert preview["final_mandatory_coverage_checklist_included"] is True
    assert preview["mandatory_operational_fact_count"] == 5
    assert set(preview["mandatory_operational_fact_ids"]) == {
        "regional.valid_coverage",
        "event.event_day",
        "patches.patch_count",
        "activity.active_track_count",
        "activity.active_family_count",
    }
    assert path.exists()
    serialized = path.read_text(encoding="utf-8")
    assert "OPENAI_API_KEY" not in serialized
    assert str(ROOT) not in serialized


def test_ollama_dry_run_has_context_diagnostics_without_provider_call(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    config = deepcopy(load_config(ROOT / "config.yaml"))
    config["brief_generation"]["debug_directory"] = str(tmp_path / "debug")
    preview, path = dry_run_scientific_brief(
        context_path=CONTEXT, config=config, root=ROOT, language="es",
        analysis_date=None, provider_name="ollama", model=None, base_url=None,
        output_directory=tmp_path, overwrite=False,
    )
    assert preview["provider"] == "ollama"
    assert preview["model"] == "qwen3:4b"
    assert preview["base_url"] == "http://localhost:11434"
    assert preview["local_generation"] is True
    assert preview["think"] is False
    assert preview["num_ctx"] == 65536
    assert preview["num_predict"] == 3500
    assert preview["request_timeout_seconds"] == 600
    assert preview["configured_num_ctx"] == 65536
    assert preview["structured_output_status"] == "json_mode_with_strict_local_validation"
    assert preview["structured_output_mode"] == "json"
    assert preview["provider_schema_supplied"] is False
    assert preview["strict_local_schema_validation"] is True
    assert preview["schema_fallback_used"] is False
    assert preview["schema_fallback_policy"].startswith("disabled")
    assert preview["strict_schema_bytes"] > 0
    assert preview["provider_schema_bytes"] == 0
    assert preview["removed_keyword_counts"] == {}
    assert preview["converted_const_count"] == 0
    assert preview["unresolved_reference_count"] == 0
    assert preview["provider_schema_validation_result"] == "not_supplied"
    assert preview["facts_used_source"] == "derived_from_claim_support"
    assert preview["mandatory_coverage_matrix_included"] is True
    assert preview["final_mandatory_coverage_checklist_included"] is True
    assert preview["approximate_request_tokens"]["es"] > 0
    assert path.exists()


def test_wrong_analysis_date_context_is_rejected_before_call(tmp_path) -> None:
    client = FakeClient([])
    with pytest.raises(ScientificBriefGenerationError, match="does not match"):
        generate_scientific_briefs(
            context_path=CONTEXT, config=_config(), root=ROOT, language="en",
            analysis_date="2026-07-18", output_directory=tmp_path, client=client,
        )
    assert client.responses.calls == []


def test_invalid_context_is_rejected_before_call(tmp_path) -> None:
    import json

    context = json.loads(CONTEXT.read_text(encoding="utf-8"))
    context["schema_version"] = "unsupported"
    invalid_path = tmp_path / "invalid-context.json"
    invalid_path.write_text(json.dumps(context), encoding="utf-8")
    client = FakeClient([])
    with pytest.raises(ScientificBriefGenerationError):
        generate_scientific_briefs(
            context_path=invalid_path, config=_config(), root=ROOT, language="en",
            output_directory=tmp_path / "out", client=client,
        )
    assert client.responses.calls == []


def test_cross_language_failure_publishes_nothing(tmp_path) -> None:
    english = deepcopy(valid_brief("en"))
    english["recent_evolution"]["paragraphs"][0]["text"] = "The anomaly is increasing."
    client = FakeClient([fake_response(valid_brief("es")), fake_response(english)])
    with pytest.raises(ScientificBriefGenerationError, match="inconsistent"):
        generate_scientific_briefs(
            context_path=CONTEXT, config=_config(), root=ROOT, language="both",
            output_directory=tmp_path, client=client,
        )
    assert not list(tmp_path.glob("scientific_brief_*.md"))
    assert list((tmp_path / "debug").glob("*_failure.json"))


def test_atomic_writer_restores_previous_outputs_after_failure(monkeypatch, tmp_path) -> None:
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("old first", encoding="utf-8")
    second.write_text("old second", encoding="utf-8")
    original_replace = Path.replace

    def fail_second_temporary(source: Path, target: Path):
        destination = Path(target)
        if destination == second and source.name.startswith(".second-"):
            raise OSError("simulated atomic replacement failure")
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", fail_second_temporary)
    with pytest.raises(OSError):
        _atomic_write_group({first: "new first", second: "new second"}, overwrite=True)
    assert first.read_text(encoding="utf-8") == "old first"
    assert second.read_text(encoding="utf-8") == "old second"


class _OllamaClient:
    def __init__(self, payloads):
        self.payloads = list(payloads)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        payload = self.payloads.pop(0)
        if isinstance(payload, BaseException):
            raise payload
        content = payload if isinstance(payload, str) else json.dumps(payload)
        return {
            "model": "qwen3:4b", "done": True,
            "message": {"content": content, "thinking": "not exported"},
            "load_duration": 10, "total_duration": 20,
            "prompt_eval_count": 100, "prompt_eval_duration": 11,
            "eval_count": 50, "eval_duration": 12,
        }


class _OllamaResponseError(Exception):
    def __init__(self, error: str, status_code: int = 400):
        super().__init__(error)
        self.error = error
        self.status_code = status_code


def _ollama_provider(client):
    config = deepcopy(load_config(ROOT / "config.yaml"))
    settings = config["brief_generation"]
    return config, OllamaProvider(resolve_provider_settings("ollama", None, None, settings), client=client)


def test_ollama_generation_runs_identical_validation_and_metadata(tmp_path) -> None:
    raw = valid_brief("es")
    raw["facts_used"].remove("activity.active_family_count")
    client = _OllamaClient([raw])
    config, provider = _ollama_provider(client)
    artifacts = generate_scientific_briefs(
        context_path=CONTEXT, config=config, root=ROOT, language="es",
        output_directory=tmp_path, llm_provider=provider,
    )
    metadata = artifacts.generations["es"].metadata.to_dict()
    assert artifacts.generations["es"].validation.final_status == "passed"
    assert metadata["provider"] == "ollama"
    assert metadata["local_generation"] is True
    assert metadata["external_network_used"] is False
    assert metadata["prompt_eval_count"] == 100
    assert metadata["eval_count"] == 50
    assert "thinking" not in json.dumps(metadata).casefold()
    assert client.calls[0]["think"] is False
    assert client.calls[0]["format"] == "json"
    assert metadata["structured_output_mode"] == "json"
    assert metadata["provider_schema_supplied"] is False
    assert metadata["strict_local_schema_validation"] is True
    assert metadata["facts_used_source"] == "derived_from_claim_support"
    assert "activity.active_family_count" in artifacts.generations["es"].payload[
        "facts_used"
    ]


def test_ollama_uses_application_owned_disclaimer_without_repair(tmp_path) -> None:
    raw = deepcopy(valid_brief("es"))
    raw["disclaimer"] = "descargo parafraseado"
    client = _OllamaClient([raw])
    config, provider = _ollama_provider(client)
    artifacts = generate_scientific_briefs(
        context_path=CONTEXT, config=config, root=ROOT, language="es",
        output_directory=tmp_path, llm_provider=provider,
    )
    result = artifacts.generations["es"]
    assert len(client.calls) == 1
    assert result.metadata.repair_count == 0
    assert result.payload["disclaimer"] == get_mandatory_disclaimer("es")
    assert result.metadata.disclaimer_source == "application_constant"
    assert result.metadata.raw_model_disclaimer_matched is False
    assert "disclaimer" not in client.calls[0]["messages"][1]["content"].split(
        "Required top-level fields: ", 1
    )[1].split(".", 1)[0]


def test_ollama_repair_is_bounded_and_uses_same_provider(tmp_path) -> None:
    invalid = deepcopy(valid_brief("en"))
    invalid.pop("title")
    client = _OllamaClient([invalid, valid_brief("en")])
    config, provider = _ollama_provider(client)
    artifacts = generate_scientific_briefs(
        context_path=CONTEXT, config=config, root=ROOT, language="en",
        output_directory=tmp_path, llm_provider=provider,
    )
    assert len(client.calls) == 2
    assert artifacts.generations["en"].metadata.repair_count == 1
    assert all(call["model"] == "qwen3:4b" for call in client.calls)
    assert all(call["format"] == "json" for call in client.calls)


def test_ollama_json_mode_runs_complete_strict_validation_without_openai(tmp_path, monkeypatch) -> None:
    def reject_openai_call(*args, **kwargs):
        raise AssertionError("OpenAI must not be called by Ollama JSON mode")

    monkeypatch.setattr(
        "src.openai_client.OpenAIProvider.generate_structured",
        reject_openai_call,
    )
    client = _OllamaClient([valid_brief("es")])
    config, provider = _ollama_provider(client)
    artifacts = generate_scientific_briefs(
        context_path=CONTEXT, config=config, root=ROOT, language="es",
        output_directory=tmp_path, llm_provider=provider,
    )
    result = artifacts.generations["es"]
    assert result.validation.final_status == "passed"
    assert result.metadata.structured_output_mode == "json"
    assert result.metadata.schema_fallback_used is False
    assert result.metadata.strict_local_schema_validation is True
    assert client.calls[0]["format"] == "json"
    assert len(client.calls) == 1


def test_failed_json_repair_is_rejected_and_previous_output_preserved(tmp_path) -> None:
    existing = tmp_path / "scientific_brief_2026-07-19_en.md"
    existing.write_text("previous validated brief", encoding="utf-8")
    invalid = deepcopy(valid_brief("en"))
    invalid.pop("title")
    client = _OllamaClient([invalid, invalid])
    config, provider = _ollama_provider(client)
    with pytest.raises(ScientificBriefGenerationError) as caught:
        generate_scientific_briefs(
            context_path=CONTEXT, config=config, root=ROOT, language="en",
            output_directory=tmp_path, llm_provider=provider, overwrite=True,
        )
    assert caught.value.code == "output_validation_failed"
    assert existing.read_text(encoding="utf-8") == "previous validated brief"
    assert not (tmp_path / "scientific_brief_2026-07-19_en.json").exists()
    assert len(client.calls) == 2
    assert all(call["format"] == "json" for call in client.calls)
    report = json.loads(
        (tmp_path / "debug/scientific_brief_2026-07-19_en_failure.json").read_text(
            encoding="utf-8"
        )
    )
    assert report["final_status"] == "failed"
    assert report["error_code"] == "output_validation_failed"


def test_json_provider_failure_writes_sanitized_report_and_returns(tmp_path) -> None:
    existing = tmp_path / "scientific_brief_2026-07-19_es.md"
    existing.write_text("previous validated brief", encoding="utf-8")
    client = _OllamaClient([_OllamaResponseError("invalid JSON mode option")])
    config, provider = _ollama_provider(client)
    with pytest.raises(ScientificBriefGenerationError) as caught:
        generate_scientific_briefs(
            context_path=CONTEXT, config=config, root=ROOT, language="es",
            output_directory=tmp_path, llm_provider=provider, overwrite=True,
        )
    assert caught.value.code == "invalid_local_request"
    assert existing.read_text(encoding="utf-8") == "previous validated brief"
    report_path = tmp_path / "debug/scientific_brief_2026-07-19_es_failure.json"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["attempted_modes"] == ["json"]
    assert report["http_status_code"] == 400
    assert report["ollama_error"] == "invalid JSON mode option"
    assert not (tmp_path / "scientific_brief_2026-07-19_es.json").exists()


def _assert_json_mode_validation_failure(tmp_path, payload) -> ScientificBriefGenerationError:
    client = _OllamaClient([payload])
    config, provider = _ollama_provider(client)
    with pytest.raises(ScientificBriefGenerationError) as caught:
        generate_scientific_briefs(
            context_path=CONTEXT,
            config=config,
            root=ROOT,
            language="en",
            output_directory=tmp_path,
            llm_provider=provider,
            allow_repair=False,
        )
    assert len(client.calls) == 1
    assert client.calls[0]["format"] == "json"
    assert not list(tmp_path.glob("scientific_brief_*.md"))
    return caught.value


def test_malformed_ollama_json_fails_without_partial_markdown(tmp_path) -> None:
    error = _assert_json_mode_validation_failure(tmp_path, "{not-json")
    assert error.code == "malformed_structured_output"


def test_json_mode_missing_required_field_fails_strict_schema(tmp_path) -> None:
    invalid = deepcopy(valid_brief("en"))
    invalid.pop("title")
    error = _assert_json_mode_validation_failure(tmp_path, invalid)
    assert error.code == "output_validation_failed"


def test_json_mode_unsupported_fact_id_fails(tmp_path) -> None:
    invalid = deepcopy(valid_brief("en"))
    invalid["executive_summary"]["paragraphs"][0]["supporting_fact_ids"].append(
        "unsupported.fact"
    )
    invalid["facts_used"].append("unsupported.fact")
    error = _assert_json_mode_validation_failure(tmp_path, invalid)
    assert error.code == "output_validation_failed"


def test_json_mode_unsupported_number_fails(tmp_path) -> None:
    invalid = deepcopy(valid_brief("en"))
    invalid["executive_summary"]["paragraphs"][0]["text"] = (
        "The regional SST anomaly was 999.99 degrees Celsius."
    )
    error = _assert_json_mode_validation_failure(tmp_path, invalid)
    assert error.code == "output_validation_failed"


def test_json_mode_prohibited_claim_fails(tmp_path) -> None:
    invalid = deepcopy(valid_brief("en"))
    invalid["executive_summary"]["paragraphs"][0]["text"] = (
        "This is an official classification."
    )
    error = _assert_json_mode_validation_failure(tmp_path, invalid)
    assert error.code == "output_validation_failed"


def test_ollama_timeout_returns_after_one_request(tmp_path) -> None:
    client = _OllamaClient([TimeoutError("request timed out")])
    config, provider = _ollama_provider(client)
    with pytest.raises(ScientificBriefGenerationError) as caught:
        generate_scientific_briefs(
            context_path=CONTEXT,
            config=config,
            root=ROOT,
            language="en",
            output_directory=tmp_path,
            llm_provider=provider,
        )
    assert caught.value.code == "local_timeout"
    assert len(client.calls) == 1
    assert not list(tmp_path.glob("scientific_brief_*.md"))


def test_ctrl_c_preserves_previous_output_and_leaves_no_temporary_markdown(tmp_path) -> None:
    existing = tmp_path / "scientific_brief_2026-07-19_en.md"
    existing.write_text("previous validated brief", encoding="utf-8")
    client = _OllamaClient([KeyboardInterrupt()])
    config, provider = _ollama_provider(client)
    with pytest.raises(KeyboardInterrupt):
        generate_scientific_briefs(
            context_path=CONTEXT,
            config=config,
            root=ROOT,
            language="en",
            output_directory=tmp_path,
            llm_provider=provider,
            overwrite=True,
        )
    assert existing.read_text(encoding="utf-8") == "previous validated brief"
    assert not list(tmp_path.glob(".*.md"))


def test_atomic_writer_restores_outputs_after_keyboard_interrupt(monkeypatch, tmp_path) -> None:
    first = tmp_path / "first.md"
    second = tmp_path / "second.md"
    first.write_text("old first", encoding="utf-8")
    second.write_text("old second", encoding="utf-8")
    original_replace = Path.replace

    def interrupt_second_temporary(source: Path, target: Path):
        destination = Path(target)
        if destination == second and source.name.startswith(".second-"):
            raise KeyboardInterrupt()
        return original_replace(source, target)

    monkeypatch.setattr(Path, "replace", interrupt_second_temporary)
    with pytest.raises(KeyboardInterrupt):
        _atomic_write_group({first: "new first", second: "new second"}, overwrite=True)
    assert first.read_text(encoding="utf-8") == "old first"
    assert second.read_text(encoding="utf-8") == "old second"
    assert not list(tmp_path.glob(".*.md"))
