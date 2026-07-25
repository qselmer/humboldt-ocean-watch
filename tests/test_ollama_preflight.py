import json

from src.llm_provider import resolve_provider_settings
from src.ollama_preflight import run_ollama_preflight
from tests.scientific_brief_test_data import generation_settings


class Client:
    def __init__(self, *, installed=True, malformed=False, error=None):
        self.installed = installed
        self.malformed = malformed
        self.error = error
        self.calls = []
        self.close_count = 0

    def list(self):
        if self.error:
            raise self.error
        return {"models": [{"model": "qwen3:4b"}] if self.installed else []}

    def chat(self, **kwargs):
        self.calls.append(kwargs)
        if self.error:
            raise self.error
        if "format" in kwargs:
            content = "not-json" if self.malformed else json.dumps({"status": "OK"})
        else:
            content = "OK"
        return {"message": {"content": content}, "done": True}

    def close(self):
        self.close_count += 1


def settings():
    return resolve_provider_settings("ollama", None, None, generation_settings())


def test_successful_independent_preflight(tmp_path) -> None:
    client = Client()
    report = run_ollama_preflight(
        settings=settings(), output_directory=tmp_path, client=client,
        package_validator=lambda: "test-version",
    )
    assert report.status == "passed"
    assert report.package_available and report.server_reachable and report.model_installed
    assert report.basic_generation_available and report.structured_generation_available
    assert report.thinking_excluded and report.output_directory_writable
    assert report.configured_context_length == 65536
    assert all(call["think"] is False for call in client.calls)


def test_package_missing_does_not_hide_independent_checks(tmp_path) -> None:
    def missing():
        raise ImportError("private path")

    report = run_ollama_preflight(
        settings=settings(), output_directory=tmp_path, client=Client(), package_validator=missing,
    )
    assert report.status == "failed"
    assert not report.package_available
    assert report.server_reachable and report.model_installed
    assert report.output_directory_writable
    assert report.errors[0]["code"] == "ollama_package_missing"
    assert "private path" not in str(report.to_dict())


def test_model_missing_and_malformed_structured_response(tmp_path) -> None:
    report = run_ollama_preflight(
        settings=settings(), output_directory=tmp_path, client=Client(installed=False, malformed=True),
        package_validator=lambda: "test-version",
    )
    codes = {item["code"] for item in report.errors}
    assert "model_not_installed" in codes
    assert "malformed_local_response" in codes
    assert report.basic_generation_available
    assert not report.structured_generation_available


def test_server_failure_does_not_make_output_check_fail(tmp_path) -> None:
    report = run_ollama_preflight(
        settings=settings(), output_directory=tmp_path,
        client=Client(error=ConnectionError("private endpoint")),
        package_validator=lambda: "test-version",
    )
    assert report.status == "failed"
    assert report.output_directory_writable
    assert not report.server_reachable
    assert "private endpoint" not in str(report.to_dict())


def test_internally_created_preflight_client_uses_timeout_and_closes(monkeypatch, tmp_path) -> None:
    client = Client()
    construction = {}

    def create(base_url, *, request_timeout_seconds=None):
        construction.update({"base_url": base_url, "timeout": request_timeout_seconds})
        return client

    monkeypatch.setattr("src.ollama_preflight.create_ollama_client", create)
    report = run_ollama_preflight(
        settings=settings(),
        output_directory=tmp_path,
        package_validator=lambda: "test-version",
    )
    assert report.status == "passed"
    assert construction == {
        "base_url": "http://localhost:11434",
        "timeout": 600.0,
    }
    assert client.close_count == 1
