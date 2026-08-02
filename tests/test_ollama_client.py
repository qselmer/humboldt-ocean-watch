import json
from copy import deepcopy
from types import SimpleNamespace

import pytest

import src.ollama_client as ollama_client
from src.brief_output_schema import scientific_brief_json_schema
from src.brief_prompts import build_prompt_package
from src.llm_errors import LLMProviderError
from src.llm_provider import resolve_provider_settings
from src.ollama_client import OllamaProvider, build_ollama_request, normalize_ollama_error
from tests.scientific_brief_test_data import generation_settings, valid_brief, validated_context


@pytest.fixture(autouse=True)
def _isolate_ollama_environment(monkeypatch) -> None:
    """Keep ambient production provider settings out of client unit tests."""
    for variable in (
        "LLM_PROVIDER",
        "OLLAMA_MODEL",
        "OLLAMA_BASE_URL",
        "OLLAMA_NUM_CTX",
        "OLLAMA_REQUEST_TIMEOUT_SECONDS",
        "OLLAMA_STRUCTURED_OUTPUT_MODE",
    ):
        monkeypatch.delenv(variable, raising=False)


class FakeOllamaClient:
    def __init__(self, responses):
        self.responses = list(responses)
        self.calls = []

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        response = self.responses.pop(0)
        if isinstance(response, Exception):
            raise response
        return response


class ResponseError(Exception):
    def __init__(self, error: str, status_code: int = -1):
        super().__init__(error)
        self.error = error
        self.status_code = status_code


def response(payload, **metadata):
    return {
        "model": "qwen3:4b",
        "done": True,
        "message": {"content": json.dumps(payload), "thinking": "must never be used"},
        "prompt_eval_count": 120,
        "eval_count": 40,
        **metadata,
    }


def provider(client, *, mode="json"):
    settings = generation_settings()
    settings = deepcopy(settings)
    settings["providers"]["ollama"]["structured_output_mode"] = mode
    settings["providers"]["ollama"]["attempt_provider_schema"] = mode == "json_schema"
    return OllamaProvider(resolve_provider_settings("ollama", None, None, settings), client=client)


def test_native_non_streaming_json_request_and_role_separation() -> None:
    settings = generation_settings()
    resolved = resolve_provider_settings("ollama", None, None, settings)
    prompt = build_prompt_package(validated_context(), "es", settings)
    schema = scientific_brief_json_schema()
    request = build_ollama_request(
        prompt, "json", resolved, request_mode="json", strict_schema=schema,
    )
    assert request["stream"] is False
    assert request["think"] is False
    assert "think" not in request["options"]
    assert request["format"] == "json"
    assert request["model"] == "qwen3:4b"
    assert request["options"] == {"temperature": 0, "num_ctx": 65536, "num_predict": 3500}
    assert "tools" not in request
    assert [message["role"] for message in request["messages"]] == ["system", "user"]
    assert "SCIENTIFIC_CONTEXT_JSON" not in request["messages"][0]["content"]
    assert "SCIENTIFIC_CONTEXT_JSON" in request["messages"][1]["content"]
    assert "exactly one JSON object" in request["messages"][1]["content"]
    assert "Compact top-level structural template" in request["messages"][1]["content"]


def test_provider_uses_message_content_and_excludes_thinking() -> None:
    client = FakeOllamaClient([response(valid_brief("es"), total_duration=123)])
    active = provider(client)
    prompt = build_prompt_package(validated_context(), "es", generation_settings())
    generated = active.generate_structured(prompt, scientific_brief_json_schema())
    assert json.loads(generated.content)["language"] == "es"
    assert "thinking" not in str(generated.provider_metadata).casefold()
    assert generated.provider_metadata["prompt_eval_count"] == 120
    assert generated.provider_metadata["eval_count"] == 40


def test_empty_content_and_errors_are_sanitized() -> None:
    client = FakeOllamaClient([{"message": {"content": ""}, "done": True}])
    with pytest.raises(LLMProviderError) as error:
        provider(client).generate_structured(SimpleNamespace(developer_text="rules", user_text="data"), {})
    assert error.value.code == "malformed_local_response"
    secret = ConnectionError("private prompt text")
    normalized = normalize_ollama_error(secret)
    assert normalized.code == "ollama_server_unavailable"
    assert "private prompt" not in str(normalized)


def test_response_error_retains_safe_status_and_ollama_detail(monkeypatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "must-not-leak")
    error = ResponseError(
        "invalid schema grammar; OPENAI_API_KEY=must-not-leak",
        status_code=400,
    )
    normalized = normalize_ollama_error(
        error,
        model="qwen3:4b",
        request_mode="json_schema",
        json_schema_supplied=True,
    )
    assert normalized.code == "structured_output_failure"
    assert normalized.diagnostics["exception_type"] == "ResponseError"
    assert normalized.diagnostics["http_status_code"] == 400
    assert normalized.diagnostics["provider"] == "ollama"
    assert normalized.diagnostics["model"] == "qwen3:4b"
    assert normalized.diagnostics["request_mode"] == "json_schema"
    assert normalized.diagnostics["json_schema_supplied"] is True
    serialized = json.dumps(normalized.diagnostics)
    assert "invalid schema grammar" in serialized
    assert "must-not-leak" not in serialized


def test_logged_response_error_does_not_leak_request_or_credentials(caplog) -> None:
    client = FakeOllamaClient([
        ResponseError(
            "schema rejected SCIENTIFIC_CONTEXT_JSON OPENAI_API_KEY=must-not-leak "
            "LLM_PROVIDER=ollama",
            400,
        ),
        ResponseError("invalid request body", 400),
    ])
    active = provider(client)
    prompt = build_prompt_package(validated_context(), "en", generation_settings())
    with caplog.at_level("WARNING", logger="src.ollama_client"):
        with pytest.raises(LLMProviderError):
            active.generate_structured(prompt, scientific_brief_json_schema())
    messages = "\n".join(record.getMessage() for record in caplog.records)
    assert "SCIENTIFIC_CONTEXT_JSON" not in messages
    assert "must-not-leak" not in messages
    assert "LLM_PROVIDER" not in messages
    assert "redacted request data" in messages


def test_provider_defaults_to_json_without_supplying_complete_schema(caplog) -> None:
    client = FakeOllamaClient([response(valid_brief("en"))])
    active = provider(client)
    prompt = build_prompt_package(validated_context(), "en", generation_settings())
    with caplog.at_level("INFO", logger="src.ollama_client"):
        generated = active.generate_structured(prompt, scientific_brief_json_schema())
    assert client.calls[0]["format"] == "json"
    assert scientific_brief_json_schema() != client.calls[0]["format"]
    assert generated.provider_metadata["structured_output_mode"] == "json"
    assert generated.provider_metadata["provider_schema_supplied"] is False
    assert generated.provider_metadata["strict_local_schema_validation"] is True
    assert generated.provider_metadata["provider_schema_bytes"] == 0
    assert generated.provider_metadata["schema_fallback_used"] is False
    assert generated.provider_metadata["attempted_modes"] == ["json"]
    assert "Structured output mode: json" in "\n".join(
        record.getMessage() for record in caplog.records
    )


def test_explicit_json_schema_mode_passes_adapted_schema_without_fallback() -> None:
    client = FakeOllamaClient([response(valid_brief("en"))])
    active = provider(client, mode="json_schema")
    prompt = build_prompt_package(validated_context(), "en", generation_settings())
    generated = active.generate_structured(prompt, scientific_brief_json_schema())
    provider_schema = client.calls[0]["format"]
    serialized = json.dumps(provider_schema)
    assert isinstance(provider_schema, dict)
    assert "$schema" not in provider_schema
    assert "$defs" not in provider_schema
    assert "$ref" not in serialized
    assert len(client.calls) == 1
    assert generated.provider_metadata["structured_output_mode"] == "json_schema"
    assert generated.provider_metadata["provider_schema_supplied"] is True


def test_explicit_schema_failure_does_not_switch_modes() -> None:
    client = FakeOllamaClient([ResponseError("failed to parse grammar", 400)])
    active = provider(client, mode="json_schema")
    prompt = build_prompt_package(validated_context(), "en", generation_settings())
    with pytest.raises(LLMProviderError) as caught:
        active.generate_structured(prompt, scientific_brief_json_schema())
    assert caught.value.code == "structured_output_failure"
    assert len(client.calls) == 1
    assert caught.value.diagnostics["attempted_modes"] == ["json_schema"]


@pytest.mark.parametrize(
    ("error", "expected_code"),
    [
        (ResponseError("invalid num_ctx option", 400), "invalid_local_request"),
        (ResponseError("invalid request body", 400), "invalid_local_request"),
        (ResponseError("model missing-model not found", 404), "model_not_installed"),
        (ResponseError("not enough memory to load model", 500), "model_load_failure"),
        (ResponseError("context length exceeded", 400), "context_length_exceeded"),
        (TimeoutError("timed out while loading"), "local_timeout"),
        (ConnectionError("server refused connection"), "ollama_server_unavailable"),
    ],
)
def test_provider_failures_return_after_one_json_request(error, expected_code) -> None:
    client = FakeOllamaClient([error])
    active = provider(client)
    prompt = build_prompt_package(validated_context(), "en", generation_settings())
    with pytest.raises(LLMProviderError) as caught:
        active.generate_structured(prompt, scientific_brief_json_schema())
    assert caught.value.code == expected_code
    assert len(client.calls) == 1
    assert caught.value.diagnostics["attempted_modes"] == ["json"]


def test_json_mode_failure_has_single_attempt_history() -> None:
    client = FakeOllamaClient([ResponseError("invalid JSON mode request", 400)])
    active = provider(client)
    prompt = build_prompt_package(validated_context(), "en", generation_settings())
    with pytest.raises(LLMProviderError) as caught:
        active.generate_structured(prompt, scientific_brief_json_schema())
    assert len(client.calls) == 1
    assert caught.value.diagnostics["attempted_modes"] == ["json"]
    assert caught.value.diagnostics["http_status_code"] == 400


def test_owned_client_uses_timeout_and_closes_after_provider_failure(monkeypatch) -> None:
    class ClosableClient(FakeOllamaClient):
        def __init__(self):
            super().__init__([ResponseError("invalid request body", 400)])
            self.close_count = 0

        def close(self):
            self.close_count += 1

    client = ClosableClient()
    construction = {}

    def create(base_url, *, request_timeout_seconds=None):
        construction.update({"base_url": base_url, "timeout": request_timeout_seconds})
        return client

    monkeypatch.setattr(ollama_client, "create_ollama_client", create)
    settings = resolve_provider_settings("ollama", None, None, generation_settings())
    active = OllamaProvider(settings)
    prompt = build_prompt_package(validated_context(), "en", generation_settings())
    with pytest.raises(LLMProviderError, match="invalid request body"):
        active.generate_structured(prompt, scientific_brief_json_schema())
    assert construction == {
        "base_url": "http://localhost:11434",
        "timeout": 600.0,
    }
    assert client.close_count == 1
    assert len(client.calls) == 1
