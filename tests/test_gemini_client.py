from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

from src.brief_generator import (
    ScientificBriefGenerationError,
    dry_run_scientific_brief,
    generate_scientific_briefs,
)
from src.brief_disclaimer import get_mandatory_disclaimer
from src.brief_output_schema import scientific_brief_json_schema
from src.brief_prompts import build_prompt_package
from src.gemini_client import (
    GeminiSDK,
    GeminiProvider,
    create_gemini_client,
    extract_gemini_text,
    gemini_api_key_from_environment,
    load_gemini_sdk,
    normalize_gemini_error,
)
from src.llm_errors import LLMConfigurationError, LLMProviderError
from src.llm_provider import get_llm_provider, resolve_provider_settings
from src.utils import load_config
from tests.scientific_brief_test_data import ROOT, generation_settings, valid_brief, validated_context


CONTEXT = ROOT / "outputs/briefs/brief_context_latest.json"


class FakeConfig:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeTypes:
    GenerateContentConfig = FakeConfig
    ThinkingConfig = FakeConfig
    AutomaticFunctionCallingConfig = FakeConfig
    HttpOptions = FakeConfig
    HttpRetryOptions = FakeConfig


def fake_response(payload, *, finish_reason="STOP", response_id="gemini-response"):
    text = payload if isinstance(payload, str) else json.dumps(payload, ensure_ascii=False)
    return SimpleNamespace(
        response_id=response_id,
        model_version="gemini-3.5-flash-test",
        candidates=[
            SimpleNamespace(
                finish_reason=finish_reason,
                content=SimpleNamespace(
                    parts=[
                        SimpleNamespace(text="private thought", thought=True),
                        SimpleNamespace(text=text, thought=False),
                    ]
                ),
            )
        ],
        prompt_feedback=SimpleNamespace(block_reason="BLOCK_REASON_UNSPECIFIED"),
        usage_metadata=SimpleNamespace(
            prompt_token_count=100,
            cached_content_token_count=10,
            candidates_token_count=50,
            total_token_count=150,
        ),
    )


class FakeModels:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def generate_content(self, **kwargs):
        self.calls.append(kwargs)
        if not self.responses:
            raise AssertionError("Unexpected Gemini call")
        result = self.responses.pop(0)
        if isinstance(result, BaseException):
            raise result
        return result


class FakeClient:
    def __init__(self, responses):
        self.models = FakeModels(responses)


@pytest.fixture(autouse=True)
def _isolate_provider_environment(monkeypatch):
    for variable in (
        "LLM_PROVIDER",
        "GEMINI_API_KEY",
        "GEMINI_MODEL",
        "GEMINI_REQUEST_TIMEOUT_SECONDS",
        "GEMINI_MAXIMUM_OUTPUT_TOKENS",
        "GEMINI_STRUCTURED_OUTPUT_MODE",
        "GEMINI_MAXIMUM_TRANSPORT_ATTEMPTS",
        "GEMINI_RETRY_INITIAL_SECONDS",
        "GEMINI_RETRY_MAXIMUM_SECONDS",
    ):
        monkeypatch.delenv(variable, raising=False)


def gemini_config():
    return deepcopy(load_config(ROOT / "config.yaml"))


def gemini_provider(responses, *, mode="json_schema", sleep=lambda _: None):
    config = gemini_config()
    config["brief_generation"]["providers"]["gemini"]["structured_output_mode"] = mode
    settings = resolve_provider_settings("gemini", None, None, config["brief_generation"])
    client = FakeClient(responses)
    return config, client, GeminiProvider(
        settings, client=client, types_module=FakeTypes, sleep=sleep
    )


def test_native_non_streaming_request_uses_json_schema_and_no_tools() -> None:
    config, client, provider = gemini_provider([fake_response(valid_brief("en"))])
    prompt = build_prompt_package(validated_context(), "en", config["brief_generation"])
    generated = provider.generate_structured(prompt, scientific_brief_json_schema())

    assert generated.content
    assert generated.provider == "gemini"
    assert generated.usage == {
        "input_tokens": 100,
        "cached_input_tokens": 10,
        "output_tokens": 50,
        "reasoning_tokens": None,
        "total_tokens": 150,
    }
    assert "private thought" not in generated.content
    assert len(client.models.calls) == 1
    request = client.models.calls[0]
    assert set(request) == {"model", "contents", "config"}
    assert request["model"] == "gemini-3.5-flash"
    native_config = request["config"]
    assert native_config.response_mime_type == "application/json"
    assert native_config.temperature == 0
    assert native_config.max_output_tokens == 3500
    assert native_config.response_json_schema["required"]
    assert "$ref" not in json.dumps(native_config.response_json_schema)
    assert native_config.system_instruction == prompt.developer_text
    assert not hasattr(native_config, "tools")
    assert native_config.automatic_function_calling.disable is True
    assert native_config.thinking_config.include_thoughts is False
    assert generated.provider_metadata["search_grounding_enabled"] is False
    assert generated.provider_metadata["provider_schema_supplied"] is True
    assert len(generated.provider_metadata["response_sha256"]) == 64


def test_provider_factory_accepts_an_offline_injected_client_without_sdk() -> None:
    config = gemini_config()
    client = FakeClient([fake_response(valid_brief("en"))])
    provider = get_llm_provider(
        "gemini",
        None,
        config["brief_generation"],
        client=client,
    )
    prompt = build_prompt_package(
        validated_context(), "en", config["brief_generation"]
    )
    generated = provider.generate_structured(prompt, scientific_brief_json_schema())
    assert generated.provider == "gemini"
    assert len(client.models.calls) == 1


def test_native_client_uses_millisecond_timeout_and_disables_sdk_retries() -> None:
    construction = {}

    class GenAI:
        @staticmethod
        def Client(**kwargs):
            construction.update(kwargs)
            return SimpleNamespace(created=True)

    client = create_gemini_client(
        api_key="secret-test-key",
        request_timeout_seconds=300,
        sdk=GeminiSDK(genai=GenAI, types=FakeTypes, version="test"),
    )
    assert client.created is True
    assert construction["api_key"] == "secret-test-key"
    assert construction["http_options"].timeout == 300_000
    assert construction["http_options"].retry_options.attempts == 1


def test_explicit_json_mode_omits_provider_schema_but_keeps_local_contract() -> None:
    config, client, provider = gemini_provider([fake_response(valid_brief("en"))], mode="json")
    prompt = build_prompt_package(validated_context(), "en", config["brief_generation"])
    generated = provider.generate_structured(prompt, scientific_brief_json_schema())

    request = client.models.calls[0]
    assert not hasattr(request["config"], "response_json_schema")
    assert "Return exactly one JSON object" in request["contents"]
    assert generated.provider_metadata["provider_schema_supplied"] is False
    assert generated.provider_metadata["strict_local_schema_validation"] is True


@pytest.mark.parametrize(
    ("class_name", "status", "message", "code", "retryable"),
    [
        ("AuthenticationError", 401, "secret-value", "invalid_api_key", False),
        ("PermissionDeniedError", 403, "secret-value", "inaccessible_model", False),
        ("ResourceExhausted", 429, "quota exhausted secret-value", "quota_exceeded", True),
        ("RateLimitError", 429, "rate limit secret-value", "rate_limit", True),
        ("ServerError", 500, "internal secret-value", "provider_internal_error", True),
        ("ServerError", 502, "gateway secret-value", "service_unavailable", True),
        ("ServerError", 503, "capacity secret-value", "service_unavailable", True),
        ("ServerError", 504, "deadline secret-value", "request_timeout", True),
        ("TimeoutError", None, "timed out secret-value", "request_timeout", True),
        ("ConnectionError", None, "network secret-value", "network_failure", True),
        ("BadRequestError", 400, "response_json_schema rejected secret-value", "structured_output_failure", False),
    ],
)
def test_gemini_errors_are_stable_sanitized_and_bounded(
    class_name: str, status, message: str, code: str, retryable: bool,
) -> None:
    error_type = type(class_name, (Exception,), {})
    error = error_type(message)
    error.status_code = status
    normalized = normalize_gemini_error(error, request_mode="json_schema")
    assert normalized.code == code
    assert normalized.retryable is retryable
    assert "secret-value" not in str(normalized)
    assert normalized.diagnostics["request_mode"] == "json_schema"


def _provider_error(status: int, provider_status: str, message: str = "secret"):
    error = type("GeminiAPIError", (Exception,), {})(message)
    error.code = status
    error.status = provider_status
    error.message = message
    return error


def test_http_503_retries_once_and_succeeds_without_scientific_repair(tmp_path) -> None:
    delays: list[float] = []
    config, client, provider = gemini_provider(
        [
            _provider_error(503, "UNAVAILABLE", "provider capacity secret"),
            fake_response(valid_brief("en")),
        ],
        mode="json",
        sleep=delays.append,
    )
    artifacts = generate_scientific_briefs(
        context_path=CONTEXT,
        config=config,
        root=ROOT,
        language="en",
        provider_name="gemini",
        output_directory=tmp_path,
        llm_provider=provider,
    )
    metadata = artifacts.generations["en"].metadata
    assert len(client.models.calls) == 2
    assert delays == [15.0]
    assert metadata.transport_attempt_count == 2
    assert metadata.transient_retry_count == 1
    assert metadata.transport_retry_delays_seconds == [15.0]
    assert metadata.repair_count == 0
    assert metadata.scientific_repair_count == 0
    assert metadata.final_provider_status == "completed"


def test_repeated_503_stops_after_configured_maximum_with_bounded_backoff() -> None:
    delays: list[float] = []
    _, client, provider = gemini_provider(
        [_provider_error(503, "UNAVAILABLE", "secret") for _ in range(3)],
        mode="json",
        sleep=delays.append,
    )
    prompt = build_prompt_package(validated_context(), "en", gemini_config()["brief_generation"])
    with pytest.raises(LLMProviderError) as caught:
        provider.generate_structured(prompt, scientific_brief_json_schema())
    assert caught.value.code == "service_unavailable"
    assert caught.value.retryable is True
    assert str(caught.value) == (
        "Gemini is temporarily unavailable because provider capacity is constrained"
    )
    assert len(client.models.calls) == 3
    assert delays == [15.0, 30.0]
    assert caught.value.diagnostics["transport_attempt_count"] == 3
    assert caught.value.diagnostics["transient_retry_count"] == 2
    assert caught.value.diagnostics["transport_retry_delays_seconds"] == [15.0, 30.0]


@pytest.mark.parametrize("status", [400, 401, 403, 404])
def test_non_transient_http_errors_are_not_retried(status: int) -> None:
    delays: list[float] = []
    _, client, provider = gemini_provider(
        [_provider_error(status, "INVALID_ARGUMENT", "request secret")],
        mode="json",
        sleep=delays.append,
    )
    prompt = build_prompt_package(validated_context(), "en", gemini_config()["brief_generation"])
    with pytest.raises(LLMProviderError):
        provider.generate_structured(prompt, scientific_brief_json_schema())
    assert len(client.models.calls) == 1
    assert delays == []


@pytest.mark.parametrize(
    "failure",
    [TimeoutError("timed out secret"), ConnectionError("network secret")],
)
def test_timeout_and_network_interruption_are_retried(failure: Exception) -> None:
    delays: list[float] = []
    _, client, provider = gemini_provider(
        [failure, fake_response(valid_brief("en"))],
        mode="json",
        sleep=delays.append,
    )
    prompt = build_prompt_package(validated_context(), "en", gemini_config()["brief_generation"])
    generated = provider.generate_structured(prompt, scientific_brief_json_schema())
    assert generated.status == "completed"
    assert len(client.models.calls) == 2
    assert delays == [15.0]


def test_exhausted_transport_retries_preserve_output_and_write_sanitized_failure(tmp_path) -> None:
    existing = tmp_path / "scientific_brief_2026-07-19_es.md"
    existing.write_text("previous validated brief", encoding="utf-8")
    secret = "AIza-secret-key prompt-and-context-secret"
    config, client, provider = gemini_provider(
        [_provider_error(503, "UNAVAILABLE", secret) for _ in range(3)],
        mode="json",
    )
    with pytest.raises(ScientificBriefGenerationError) as caught:
        generate_scientific_briefs(
            context_path=CONTEXT,
            config=config,
            root=ROOT,
            language="es",
            provider_name="gemini",
            output_directory=tmp_path,
            llm_provider=provider,
            overwrite=True,
        )
    assert caught.value.code == "service_unavailable"
    assert len(client.models.calls) == 3
    assert existing.read_text(encoding="utf-8") == "previous validated brief"
    assert not (tmp_path / "scientific_brief_2026-07-19_es.json").exists()
    assert not list(tmp_path.glob(".*.md"))
    report = json.loads(
        (tmp_path / "debug/scientific_brief_2026-07-19_es_failure.json").read_text(
            encoding="utf-8"
        )
    )
    serialized = json.dumps(report)
    assert secret not in serialized
    assert "prompt-and-context-secret" not in serialized
    assert report["transport_attempt_count"] == 3
    assert report["transient_retry_count"] == 2
    assert report["transport_retry_delays_seconds"] == [15.0, 30.0]
    assert report["final_provider_status"] == "service_unavailable"
    assert report["scientific_repair_count"] == 0
    assert report["automatic_function_calling_enabled"] is False
    assert report["request_duration_seconds"] >= 0


def test_missing_package_and_key_have_clear_local_errors(monkeypatch) -> None:
    import builtins

    original_import = builtins.__import__

    def missing_google(name, *args, **kwargs):
        if name == "google" or name.startswith("google.genai"):
            raise ImportError("simulated missing package")
        return original_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", missing_google)
    with pytest.raises(LLMConfigurationError) as missing_package:
        load_gemini_sdk()
    assert missing_package.value.code == "gemini_package_missing"
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    with pytest.raises(LLMConfigurationError) as missing_key:
        gemini_api_key_from_environment()
    assert missing_key.value.code == "missing_api_key"


def test_blocked_empty_and_token_limited_responses_are_explicit() -> None:
    blocked = fake_response({"status": "blocked"})
    blocked.prompt_feedback.block_reason = "SAFETY"
    with pytest.raises(LLMProviderError) as refusal:
        extract_gemini_text(blocked)
    assert refusal.value.code == "provider_refusal"

    limited = fake_response({"status": "partial"}, finish_reason="MAX_TOKENS")
    with pytest.raises(LLMProviderError) as token_limit:
        extract_gemini_text(limited)
    assert token_limit.value.code == "output_token_limit"

    empty = fake_response("")
    empty.candidates[0].content.parts = []
    empty.text = ""
    with pytest.raises(LLMProviderError) as malformed:
        extract_gemini_text(empty)
    assert malformed.value.code == "malformed_structured_output"


def test_valid_gemini_generation_writes_artifacts_and_metadata(tmp_path) -> None:
    raw = valid_brief("en")
    raw["facts_used"].remove("activity.active_family_count")
    config, client, provider = gemini_provider([fake_response(raw)])
    artifacts = generate_scientific_briefs(
        context_path=CONTEXT,
        config=config,
        root=ROOT,
        language="en",
        provider_name="gemini",
        output_directory=tmp_path,
        llm_provider=provider,
    )
    result = artifacts.generations["en"]
    metadata = result.metadata.to_dict()
    assert result.validation.final_status == "passed"
    assert metadata["provider"] == "gemini"
    assert metadata["local_generation"] is False
    assert metadata["external_network_used"] is True
    assert metadata["tools_enabled"] is False
    assert metadata["automatic_function_calling_enabled"] is False
    assert metadata["search_grounding_enabled"] is False
    assert metadata["generation_call_count"] == 1
    assert metadata["transport_attempt_count"] == 1
    assert metadata["transient_retry_count"] == 0
    assert metadata["scientific_repair_count"] == 0
    assert metadata["final_provider_status"] == "completed"
    assert metadata["request_duration_seconds"] is not None
    assert len(metadata["response_sha256"]) == 64
    assert metadata["input_tokens"] == 100
    assert metadata["facts_used_source"] == "derived_from_claim_support"
    assert "activity.active_family_count" in result.payload["facts_used"]
    assert all(path.exists() for path in artifacts.paths.values())
    assert len(client.models.calls) == 1


def test_gemini_uses_application_owned_disclaimer_without_repair(tmp_path) -> None:
    raw = deepcopy(valid_brief("en"))
    raw["disclaimer"] = "A provider-authored paraphrase that must not be retained."
    config, client, provider = gemini_provider([fake_response(raw)], mode="json")
    artifacts = generate_scientific_briefs(
        context_path=CONTEXT,
        config=config,
        root=ROOT,
        language="en",
        provider_name="gemini",
        output_directory=tmp_path,
        llm_provider=provider,
    )
    result = artifacts.generations["en"]
    assert len(client.models.calls) == 1
    assert result.metadata.repair_count == 0
    assert result.payload["disclaimer"] == get_mandatory_disclaimer("en")
    assert result.metadata.disclaimer_source == "application_constant"
    assert result.metadata.disclaimer_inserted is True
    assert result.metadata.raw_model_disclaimer_present is True
    assert result.metadata.raw_model_disclaimer_matched is False
    assert not hasattr(client.models.calls[0]["config"], "response_json_schema")


def test_gemini_dry_run_needs_no_key_and_makes_no_provider_call(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    def unexpected_call(*args, **kwargs):
        raise AssertionError("A dry-run must not instantiate or call any provider")

    monkeypatch.setattr(GeminiProvider, "generate_structured", unexpected_call)
    monkeypatch.setattr(
        "src.openai_client.OpenAIProvider.generate_structured", unexpected_call
    )
    monkeypatch.setattr(
        "src.ollama_client.OllamaProvider.generate_structured", unexpected_call
    )
    preview, path = dry_run_scientific_brief(
        context_path=CONTEXT,
        config=gemini_config(),
        root=ROOT,
        language="es",
        analysis_date=None,
        provider_name="gemini",
        model="gemini-3.5-flash",
        output_directory=tmp_path,
        overwrite=False,
    )
    assert preview["provider"] == "gemini"
    assert preview["model"] == "gemini-3.5-flash"
    assert preview["request_timeout_seconds"] == 300
    assert preview["structured_output_mode"] == "json"
    assert preview["provider_schema_supplied"] is False
    assert preview["strict_local_schema_validation"] is True
    assert preview["compact_response_contract_included"] is True
    assert preview["compact_response_contract_bytes"] > 0
    assert preview["compact_response_contract_bytes"] < preview["strict_schema_bytes"]
    assert preview["disclaimer_source"] == "application_constant"
    assert preview["disclaimer_inserted_before_strict_validation"] is True
    assert preview["model_generates_disclaimer"] is False
    assert preview["mandatory_operational_coverage_plan_included"] is True
    assert preview["mandatory_coverage_matrix_included"] is True
    assert preview["facts_used_source"] == "derived_from_claim_support"
    assert preview["final_mandatory_coverage_checklist_included"] is True
    assert preview["mandatory_operational_fact_count"] == 5
    assert "event.event_day" in preview["mandatory_operational_fact_ids"]
    assert "patches.patch_count" in preview["mandatory_operational_fact_ids"]
    assert "activity.active_family_count" in preview[
        "mandatory_operational_fact_ids"
    ]
    assert preview["provider_schema_bytes"] == 0
    assert preview["external_network_used"] is False
    assert preview["maximum_transport_attempts"] == 3
    assert preview["retry_initial_seconds"] == 15
    assert preview["retry_maximum_seconds"] == 60
    assert preview["tools_enabled"] is False
    assert preview["automatic_function_calling_enabled"] is False
    assert path.exists()


def test_normal_generation_uses_local_checks_without_full_preflight(monkeypatch, tmp_path) -> None:
    config, client, provider = gemini_provider(
        [fake_response(valid_brief("en"))], mode="json"
    )
    activity = {"local_checks": 0, "preflight": 0}

    def local_check(*, output_directory):
        activity["local_checks"] += 1
        output_directory.mkdir(parents=True, exist_ok=True)

    def forbidden_preflight(*, output_directory):
        activity["preflight"] += 1
        raise AssertionError("Normal generation must not run the full preflight")

    provider.validate_local_environment = local_check
    provider.preflight = forbidden_preflight
    monkeypatch.setattr("src.brief_generator.get_llm_provider", lambda *args, **kwargs: provider)
    artifacts = generate_scientific_briefs(
        context_path=CONTEXT,
        config=config,
        root=ROOT,
        language="en",
        provider_name="gemini",
        output_directory=tmp_path,
    )
    assert artifacts.generations["en"].validation.final_status == "passed"
    assert activity == {"local_checks": 1, "preflight": 0}
    assert len(client.models.calls) == 1


def test_malformed_gemini_json_fails_without_partial_markdown(tmp_path) -> None:
    config, client, provider = gemini_provider([fake_response("{not-json")])
    with pytest.raises(ScientificBriefGenerationError) as caught:
        generate_scientific_briefs(
            context_path=CONTEXT,
            config=config,
            root=ROOT,
            language="en",
            provider_name="gemini",
            output_directory=tmp_path,
            llm_provider=provider,
            allow_repair=False,
        )
    assert caught.value.code == "malformed_structured_output"
    assert len(client.models.calls) == 1
    assert not list(tmp_path.glob("scientific_brief_*.md"))


def test_gemini_generation_never_falls_back_to_openai_or_ollama(monkeypatch, tmp_path) -> None:
    def forbidden_fallback(*args, **kwargs):
        raise AssertionError("Cross-provider fallback is forbidden")

    monkeypatch.setattr(
        "src.openai_client.OpenAIProvider.generate_structured", forbidden_fallback
    )
    monkeypatch.setattr(
        "src.ollama_client.OllamaProvider.generate_structured", forbidden_fallback
    )
    config, client, provider = gemini_provider([fake_response(valid_brief("en"))])
    artifacts = generate_scientific_briefs(
        context_path=CONTEXT,
        config=config,
        root=ROOT,
        language="en",
        provider_name="gemini",
        output_directory=tmp_path,
        llm_provider=provider,
    )
    assert artifacts.generations["en"].metadata.provider == "gemini"
    assert len(client.models.calls) == 1


def test_structural_failure_enters_exactly_one_same_provider_repair(tmp_path) -> None:
    invalid = deepcopy(valid_brief("en"))
    invalid["executive_summary"] = "invalid section"
    config, client, provider = gemini_provider(
        [fake_response(invalid), fake_response(valid_brief("en"))]
    )
    artifacts = generate_scientific_briefs(
        context_path=CONTEXT,
        config=config,
        root=ROOT,
        language="en",
        provider_name="gemini",
        output_directory=tmp_path,
        llm_provider=provider,
    )
    assert len(client.models.calls) == 2
    assert all(call["model"] == "gemini-3.5-flash" for call in client.models.calls)
    assert artifacts.generations["en"].metadata.repair_count == 1
    assert artifacts.generations["en"].metadata.generation_call_count == 2
    assert "invalid_section_type" in client.models.calls[1]["contents"]


def test_json_mode_repairs_missing_operational_fact_coverage_once(tmp_path) -> None:
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
    config, client, provider = gemini_provider(
        [fake_response(invalid), fake_response(valid_brief("en"))],
        mode="json",
    )

    artifacts = generate_scientific_briefs(
        context_path=CONTEXT,
        config=config,
        root=ROOT,
        language="en",
        provider_name="gemini",
        output_directory=tmp_path,
        llm_provider=provider,
    )

    assert len(client.models.calls) == 2
    assert all(
        call["model"] == "gemini-3.5-flash" for call in client.models.calls
    )
    assert all(
        not hasattr(call["config"], "response_json_schema")
        for call in client.models.calls
    )
    repair_contents = client.models.calls[1]["contents"]
    for fact_id in missing:
        assert fact_id in repair_contents
    assert '"value": 69' in repair_contents
    assert '"unit": "day"' in repair_contents
    assert '"claim_support_field": "supporting_fact_ids"' in repair_contents
    result = artifacts.generations["en"]
    assert result.metadata.repair_count == 1
    assert result.metadata.scientific_repair_count == 1
    assert result.validation.final_status == "passed"
    assert result.metadata.facts_used_source == "derived_from_claim_support"


def test_flattened_first_response_and_guessed_repair_are_both_rejected(tmp_path) -> None:
    flattened = deepcopy(valid_brief("en"))
    for section in (
        "executive_summary", "regional_state", "recent_evolution",
        "spatial_structure", "event_status", "data_quality",
    ):
        flattened[section] = "flattened"
    flattened["key_messages"] = ["flattened"]
    flattened["limitations"] = "flattened"
    flattened["generation_notes"] = "flattened"

    guessed = deepcopy(valid_brief("en"))
    for section in (
        "executive_summary", "regional_state", "recent_evolution",
        "spatial_structure", "event_status", "data_quality",
    ):
        guessed[section] = {"text": "guessed wrapper"}
    guessed["key_messages"] = [{"message": "guessed wrapper"}]
    guessed["limitations"] = [{"limitation_id": "limit-1", "statement": "guessed wrapper"}]

    config, client, provider = gemini_provider(
        [fake_response(flattened), fake_response(guessed)], mode="json"
    )
    with pytest.raises(ScientificBriefGenerationError) as caught:
        generate_scientific_briefs(
            context_path=CONTEXT,
            config=config,
            root=ROOT,
            language="en",
            provider_name="gemini",
            output_directory=tmp_path,
            llm_provider=provider,
        )
    assert caught.value.code == "output_validation_failed"
    assert len(client.models.calls) == 2
    assert not list(tmp_path.glob("scientific_brief_*.json"))
    assert not list(tmp_path.glob("scientific_brief_*.md"))


def test_failed_repair_preserves_previous_output_and_publishes_no_json(tmp_path) -> None:
    existing = tmp_path / "scientific_brief_2026-07-19_en.md"
    existing.write_text("previous valid brief", encoding="utf-8")
    invalid = deepcopy(valid_brief("en"))
    invalid["executive_summary"] = "invalid section"
    config, client, provider = gemini_provider([fake_response(invalid), fake_response(invalid)])
    with pytest.raises(ScientificBriefGenerationError) as caught:
        generate_scientific_briefs(
            context_path=CONTEXT,
            config=config,
            root=ROOT,
            language="en",
            provider_name="gemini",
            output_directory=tmp_path,
            llm_provider=provider,
            overwrite=True,
        )
    assert caught.value.code == "output_validation_failed"
    assert len(client.models.calls) == 2
    assert existing.read_text(encoding="utf-8") == "previous valid brief"
    assert not (tmp_path / "scientific_brief_2026-07-19_en.json").exists()


@pytest.mark.parametrize(
    "mutation",
    [
        lambda payload: (
            payload["executive_summary"]["paragraphs"][0]["supporting_fact_ids"].append("unsupported.fact"),
            payload["facts_used"].append("unsupported.fact"),
        ),
        lambda payload: payload["executive_summary"]["paragraphs"][0].update(
            {"text": "The SST anomaly was 999.99 degrees Celsius."}
        ),
        lambda payload: payload["executive_summary"]["paragraphs"][0].update(
            {"text": "This is an official classification."}
        ),
        lambda payload: payload.update({"language": "es"}),
    ],
)
def test_gemini_has_no_provider_specific_scientific_exceptions(tmp_path, mutation) -> None:
    invalid = deepcopy(valid_brief("en"))
    mutation(invalid)
    config, client, provider = gemini_provider([fake_response(invalid)])
    with pytest.raises(ScientificBriefGenerationError) as caught:
        generate_scientific_briefs(
            context_path=CONTEXT,
            config=config,
            root=ROOT,
            language="en",
            provider_name="gemini",
            output_directory=tmp_path,
            llm_provider=provider,
            allow_repair=False,
        )
    assert caught.value.code == "output_validation_failed"
    assert len(client.models.calls) == 1
    assert not list(tmp_path.glob("scientific_brief_*.md"))
