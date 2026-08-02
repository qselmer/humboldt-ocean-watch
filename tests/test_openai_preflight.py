from types import SimpleNamespace

import pytest

from src.openai_client import (
    OpenAIConfigurationError,
    api_key_from_environment,
    classify_openai_error,
    resolve_model,
    validate_installed_sdk_contract,
)
from src.openai_preflight import run_openai_preflight


class Models:
    def __init__(self, error=None) -> None:
        self.error = error
        self.requested = None

    def retrieve(self, model: str):
        self.requested = model
        if self.error:
            raise self.error
        return {"id": model}


class Client:
    def __init__(self, error=None) -> None:
        self.models = Models(error)
        self.responses = SimpleNamespace(create=lambda **kwargs: None)


def test_sdk_contract_and_model_defaults(monkeypatch) -> None:
    assert validate_installed_sdk_contract()
    monkeypatch.delenv("OPENAI_MODEL", raising=False)
    assert resolve_model("gpt-5.6") == "gpt-5.6"
    monkeypatch.setenv("OPENAI_MODEL", "gpt-5.6-project")
    assert resolve_model("gpt-5.6") == "gpt-5.6-project"
    assert resolve_model("gpt-5.6", "gpt-5.6-explicit") == "gpt-5.6-explicit"


@pytest.mark.parametrize("value", [None, "", "   "])
def test_missing_and_empty_key_fail_without_disclosure(monkeypatch, value) -> None:
    if value is None:
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
    else:
        monkeypatch.setenv("OPENAI_API_KEY", value)
    with pytest.raises(OpenAIConfigurationError):
        api_key_from_environment()


def test_accessible_model_preflight(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-value")
    client = Client()
    report = run_openai_preflight(
        configured_model="gpt-5.6", output_directory=tmp_path, client=client
    )
    assert report.status == "passed"
    assert report.api_key_status == "present"
    assert client.models.requested == "gpt-5.6"
    assert "secret-test-value" not in str(report.to_dict())


@pytest.mark.parametrize(
    ("class_name", "status", "expected"),
    [
        ("AuthenticationError", 401, "invalid_api_key"),
        ("PermissionDeniedError", 403, "inaccessible_model"),
        ("RateLimitError", 429, "rate_limit"),
        ("APIConnectionError", None, "network_failure"),
        ("APITimeoutError", None, "request_timeout"),
    ],
)
def test_sanitized_openai_errors(class_name: str, status, expected: str) -> None:
    error_type = type(class_name, (Exception,), {})
    error = error_type("do not echo secret-test-value")
    error.status_code = status
    safe = classify_openai_error(error)
    assert safe.code == expected
    assert "secret-test-value" not in str(safe)


def test_inaccessible_and_rate_limited_preflight_are_nonpassing(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", "secret-test-value")
    for class_name, status, message in (
        ("PermissionDeniedError", 403, "not accessible"),
        ("RateLimitError", 429, "rate limit"),
        ("APIConnectionError", None, "could not be reached"),
    ):
        error_type = type(class_name, (Exception,), {})
        error = error_type("secret-test-value")
        error.status_code = status
        report = run_openai_preflight(
            configured_model="gpt-5.6", output_directory=tmp_path, client=Client(error)
        )
        assert report.status == "failed"
        assert message in " ".join(report.messages)
        assert "secret-test-value" not in str(report.to_dict())
