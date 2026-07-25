from __future__ import annotations

from copy import deepcopy
import json
from types import SimpleNamespace

import pytest

import scripts.check_llm_access as access_cli
from src.gemini_preflight import run_gemini_preflight
from src.llm_provider import resolve_provider_settings
from src.utils import load_config
from tests.scientific_brief_test_data import ROOT


GEMINI_ENVIRONMENT_VARIABLES = (
    "GEMINI_MODEL",
    "GEMINI_API_KEY",
    "GEMINI_REQUEST_TIMEOUT_SECONDS",
    "GEMINI_MAXIMUM_OUTPUT_TOKENS",
    "GEMINI_STRUCTURED_OUTPUT_MODE",
    "GEMINI_MAXIMUM_TRANSPORT_ATTEMPTS",
    "GEMINI_RETRY_INITIAL_SECONDS",
    "GEMINI_RETRY_MAXIMUM_SECONDS",
    "LLM_PROVIDER",
)


@pytest.fixture(autouse=True)
def _isolate_gemini_environment(monkeypatch):
    """Keep ambient provider settings out of deterministic preflight tests."""
    for name in GEMINI_ENVIRONMENT_VARIABLES:
        monkeypatch.delenv(name, raising=False)


class FakeConfig:
    def __init__(self, **kwargs):
        self.kwargs = dict(kwargs)
        for key, value in kwargs.items():
            setattr(self, key, value)


class FakeTypes:
    GenerateContentConfig = FakeConfig
    ThinkingConfig = FakeConfig
    AutomaticFunctionCallingConfig = FakeConfig


def response(text: str):
    return SimpleNamespace(
        candidates=[
            SimpleNamespace(
                finish_reason="STOP",
                content=SimpleNamespace(
                    parts=[SimpleNamespace(text=text, thought=False)]
                ),
            )
        ],
        prompt_feedback=SimpleNamespace(block_reason="BLOCK_REASON_UNSPECIFIED"),
    )


class Models:
    def __init__(self, *, model_error=None, generation_error=None):
        self.model_error = model_error
        self.generation_error = generation_error
        self.get_calls = []
        self.generate_calls = []

    def get(self, **kwargs):
        self.get_calls.append(kwargs)
        if self.model_error:
            raise self.model_error
        return {"name": kwargs["model"]}

    def generate_content(self, **kwargs):
        self.generate_calls.append(kwargs)
        if self.generation_error:
            raise self.generation_error
        if len(self.generate_calls) == 1:
            return response("OK")
        return response(json.dumps({"status": "OK"}))


class Client:
    def __init__(self, **kwargs):
        self.models = Models(**kwargs)


def settings():
    config = deepcopy(load_config(ROOT / "config.yaml"))["brief_generation"]
    return resolve_provider_settings("gemini", None, None, config)


def test_all_preflight_checks_pass_with_small_fake_calls(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-test-key")
    client = Client()
    report = run_gemini_preflight(
        settings=settings(),
        output_directory=tmp_path,
        client=client,
        types_module=FakeTypes,
        package_validator=lambda: "test-version",
    )
    data = report.to_dict()
    assert report.status == "passed"
    assert data["package_available"] is True
    assert data["api_key_available"] is True
    assert data["api_reachable"] is True
    assert data["model_accessible"] is True
    assert data["basic_generation_available"] is True
    assert data["structured_generation_available"] is True
    assert data["output_directory_writable"] is True
    assert client.models.get_calls == [{"model": "gemini-3.5-flash"}]
    assert len(client.models.generate_calls) == 2
    assert all(
        call["config"].automatic_function_calling.disable is True
        for call in client.models.generate_calls
    )
    assert all(not hasattr(call["config"], "tools") for call in client.models.generate_calls)
    sent = json.dumps(client.models.generate_calls, default=lambda value: value.kwargs)
    assert "SCIENTIFIC_CONTEXT_JSON" not in sent
    assert "secret-test-key" not in str(data)


def test_check_llm_access_retains_explicit_full_preflight(monkeypatch, tmp_path) -> None:
    calls = []
    report_data = {
        "package_available": True,
        "api_key_available": True,
        "api_reachable": True,
        "model_accessible": True,
        "basic_generation_available": True,
        "structured_generation_available": True,
        "output_directory_writable": True,
        "errors": [],
        "messages": [],
    }
    report = SimpleNamespace(status="passed", to_dict=lambda: report_data)

    class Provider:
        provider_name = "gemini"
        model_name = "gemini-3.1-flash-lite"
        base_url = None

        def preflight(self, *, output_directory):
            calls.append(output_directory)
            return report

    monkeypatch.setattr(
        access_cli,
        "parse_args",
        lambda: SimpleNamespace(
            provider="gemini",
            model="gemini-3.1-flash-lite",
            base_url=None,
            config=tmp_path / "config.yaml",
        ),
    )
    monkeypatch.setattr(
        access_cli,
        "load_config",
        lambda _: {"brief_generation": {"output_directory": str(tmp_path)}},
    )
    monkeypatch.setattr(access_cli, "get_llm_provider", lambda *args, **kwargs: Provider())
    access_cli.main()
    assert calls == [tmp_path]


def test_missing_key_fails_without_any_provider_call(monkeypatch, tmp_path) -> None:
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    client = Client()
    report = run_gemini_preflight(
        settings=settings(),
        output_directory=tmp_path,
        client=client,
        types_module=FakeTypes,
        package_validator=lambda: "test-version",
    )
    assert report.status == "failed"
    assert "missing_api_key" in {item["code"] for item in report.errors}
    assert client.models.get_calls == []
    assert client.models.generate_calls == []


def test_package_model_and_generation_checks_fail_independently(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-test-key")
    package_failure = run_gemini_preflight(
        settings=settings(),
        output_directory=tmp_path / "package",
        client=Client(),
        types_module=FakeTypes,
        package_validator=lambda: (_ for _ in ()).throw(ImportError("missing")),
    )
    assert "gemini_package_missing" in {item["code"] for item in package_failure.errors}

    permission_error = type("PermissionDeniedError", (Exception,), {})
    denied = permission_error("secret-test-key")
    denied.status_code = 403
    model_failure_client = Client(model_error=denied)
    model_failure = run_gemini_preflight(
        settings=settings(),
        output_directory=tmp_path / "model",
        client=model_failure_client,
        types_module=FakeTypes,
        package_validator=lambda: "test-version",
    )
    assert model_failure.model_accessible is False
    assert model_failure.basic_generation_available is True
    assert model_failure.structured_generation_available is True
    assert "inaccessible_model" in {item["code"] for item in model_failure.errors}
    assert "secret-test-key" not in str(model_failure.to_dict())


def test_generation_failure_does_not_expose_key(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("GEMINI_API_KEY", "secret-test-key")
    timeout_error = TimeoutError("secret-test-key request timed out")
    client = Client(generation_error=timeout_error)
    report = run_gemini_preflight(
        settings=settings(),
        output_directory=tmp_path,
        client=client,
        types_module=FakeTypes,
        package_validator=lambda: "test-version",
    )
    assert report.status == "failed"
    assert "request_timeout" in {item["code"] for item in report.errors}
    assert "secret-test-key" not in str(report.to_dict())
