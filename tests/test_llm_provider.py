from copy import deepcopy

import pytest

from src.llm_errors import LLMConfigurationError
from src.llm_provider import (
    get_llm_provider,
    resolve_provider_name,
    resolve_provider_settings,
)
from src.utils import load_config
from tests.scientific_brief_test_data import ROOT


def settings():
    return deepcopy(load_config(ROOT / "config.yaml")["brief_generation"])


def test_gemini_is_default_and_other_providers_remain_selectable(monkeypatch) -> None:
    monkeypatch.delenv("LLM_PROVIDER", raising=False)
    assert settings()["provider"] == "gemini"
    config = settings()
    config.pop("provider")
    assert resolve_provider_name(None, config) == "gemini"
    assert get_llm_provider("gemini", None, config).provider_name == "gemini"
    assert get_llm_provider("ollama", None, config).provider_name == "ollama"
    assert get_llm_provider("openai", None, config).provider_name == "openai"


def test_provider_cli_precedes_environment_and_config(monkeypatch) -> None:
    monkeypatch.setenv("LLM_PROVIDER", "openai")
    assert resolve_provider_name("ollama", settings()) == "ollama"
    assert resolve_provider_name(None, settings()) == "openai"


def test_gemini_configuration_and_environment_overrides(monkeypatch) -> None:
    for name in (
        "GEMINI_MODEL",
        "GEMINI_REQUEST_TIMEOUT_SECONDS",
        "GEMINI_MAXIMUM_OUTPUT_TOKENS",
        "GEMINI_STRUCTURED_OUTPUT_MODE",
        "GEMINI_MAXIMUM_TRANSPORT_ATTEMPTS",
        "GEMINI_RETRY_INITIAL_SECONDS",
        "GEMINI_RETRY_MAXIMUM_SECONDS",
    ):
        monkeypatch.delenv(name, raising=False)
    resolved = resolve_provider_settings("gemini", None, None, settings())
    assert resolved.model_name == "gemini-3.1-flash-lite"
    assert resolved.local_generation is False
    assert resolved.request_timeout_seconds == 300
    assert resolved.num_predict == 3500
    assert resolved.structured_output_mode == "json"
    assert resolved.api_key_environment_variable == "GEMINI_API_KEY"
    assert resolved.use_tools is False
    assert resolved.use_search_grounding is False
    assert resolved.include_thoughts is False
    assert resolved.maximum_transport_attempts == 3
    assert resolved.retry_initial_seconds == 15
    assert resolved.retry_maximum_seconds == 60

    monkeypatch.setenv("GEMINI_MODEL", "environment-model")
    monkeypatch.setenv("GEMINI_REQUEST_TIMEOUT_SECONDS", "45.5")
    monkeypatch.setenv("GEMINI_MAXIMUM_OUTPUT_TOKENS", "1024")
    monkeypatch.setenv("GEMINI_STRUCTURED_OUTPUT_MODE", "json")
    monkeypatch.setenv("GEMINI_MAXIMUM_TRANSPORT_ATTEMPTS", "2")
    monkeypatch.setenv("GEMINI_RETRY_INITIAL_SECONDS", "4.5")
    monkeypatch.setenv("GEMINI_RETRY_MAXIMUM_SECONDS", "12")
    environment = resolve_provider_settings("gemini", None, None, settings())
    assert environment.model_name == "environment-model"
    assert environment.request_timeout_seconds == 45.5
    assert environment.num_predict == 1024
    assert environment.structured_output_mode == "json"
    assert environment.maximum_transport_attempts == 2
    assert environment.retry_initial_seconds == 4.5
    assert environment.retry_maximum_seconds == 12
    assert resolve_provider_settings("gemini", "cli-model", None, settings()).model_name == "cli-model"


def test_invalid_gemini_structured_output_mode_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("GEMINI_STRUCTURED_OUTPUT_MODE", "markdown")
    with pytest.raises(LLMConfigurationError, match="json_schema"):
        resolve_provider_settings("gemini", None, None, settings())


@pytest.mark.parametrize(
    ("variable", "value"),
    [
        ("GEMINI_REQUEST_TIMEOUT_SECONDS", "0"),
        ("GEMINI_REQUEST_TIMEOUT_SECONDS", "not-a-number"),
        ("GEMINI_MAXIMUM_OUTPUT_TOKENS", "-1"),
        ("GEMINI_MAXIMUM_TRANSPORT_ATTEMPTS", "4"),
        ("GEMINI_RETRY_INITIAL_SECONDS", "0"),
        ("GEMINI_RETRY_MAXIMUM_SECONDS", "not-a-number"),
    ],
)
def test_invalid_gemini_numeric_overrides_are_rejected(monkeypatch, variable, value) -> None:
    monkeypatch.setenv(variable, value)
    with pytest.raises(LLMConfigurationError):
        resolve_provider_settings("gemini", None, None, settings())


def test_ollama_configuration_resolution(monkeypatch) -> None:
    for name in (
        "OLLAMA_MODEL", "OLLAMA_BASE_URL", "OLLAMA_NUM_CTX",
        "OLLAMA_REQUEST_TIMEOUT_SECONDS", "OLLAMA_STRUCTURED_OUTPUT_MODE",
    ):
        monkeypatch.delenv(name, raising=False)
    resolved = resolve_provider_settings("ollama", None, None, settings())
    assert resolved.model_name == "qwen3:4b"
    assert resolved.base_url == "http://localhost:11434"
    assert resolved.num_ctx == 65536
    assert resolved.num_predict == 3500
    assert resolved.think is False
    assert resolved.temperature == 0
    assert resolved.local_generation is True
    assert resolved.request_timeout_seconds == 600
    assert resolved.structured_output_mode == "json"
    assert resolved.attempt_provider_schema is False


def test_ollama_environment_and_explicit_overrides(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_MODEL", "env-model")
    monkeypatch.setenv("OLLAMA_BASE_URL", "http://env:11434/")
    monkeypatch.setenv("OLLAMA_NUM_CTX", "32768")
    monkeypatch.setenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", "45.5")
    environment = resolve_provider_settings("ollama", None, None, settings())
    assert (environment.model_name, environment.base_url, environment.num_ctx) == (
        "env-model", "http://env:11434", 32768,
    )
    assert environment.request_timeout_seconds == 45.5
    explicit = resolve_provider_settings("ollama", "cli-model", "http://cli:11434/", settings())
    assert explicit.model_name == "cli-model"
    assert explicit.base_url == "http://cli:11434"


def test_invalid_provider_never_falls_back() -> None:
    with pytest.raises(LLMConfigurationError, match="Unsupported"):
        get_llm_provider("unknown", None, settings())


def test_environment_can_explicitly_select_json_schema(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_STRUCTURED_OUTPUT_MODE", "json_schema")
    resolved = resolve_provider_settings("ollama", None, None, settings())
    assert resolved.structured_output_mode == "json_schema"
    assert resolved.attempt_provider_schema is True


def test_invalid_ollama_structured_output_mode_is_rejected(monkeypatch) -> None:
    monkeypatch.setenv("OLLAMA_STRUCTURED_OUTPUT_MODE", "text")
    with pytest.raises(LLMConfigurationError, match="json_schema"):
        resolve_provider_settings("ollama", None, None, settings())


@pytest.mark.parametrize("value", ["0", "-1", "not-a-number"])
def test_invalid_ollama_request_timeout_is_rejected(monkeypatch, value) -> None:
    monkeypatch.setenv("OLLAMA_REQUEST_TIMEOUT_SECONDS", value)
    with pytest.raises(LLMConfigurationError, match="positive number"):
        resolve_provider_settings("ollama", None, None, settings())
