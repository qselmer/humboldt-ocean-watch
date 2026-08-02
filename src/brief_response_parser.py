"""Safe parsing of final structured Responses API output."""

from __future__ import annotations

from dataclasses import dataclass
import json
from typing import Any

from src.export_utils import to_json_compatible
from src.llm_provider import StructuredGeneration


class BriefResponseError(RuntimeError):
    """A sanitized model-response failure."""

    def __init__(self, code: str, message: str) -> None:
        super().__init__(message)
        self.code = code


def _get(value: Any, name: str, default: Any = None) -> Any:
    if isinstance(value, dict):
        return value.get(name, default)
    return getattr(value, name, default)


def _walk(value: Any) -> list[Any]:
    items = [value]
    if isinstance(value, dict):
        for item in value.values():
            items.extend(_walk(item))
    elif isinstance(value, (list, tuple)):
        for item in value:
            items.extend(_walk(item))
    elif hasattr(value, "model_dump"):
        items.extend(_walk(value.model_dump()))
    return items


@dataclass(frozen=True)
class ParsedBriefResponse:
    payload: dict[str, Any]
    response_id: str | None
    model: str | None
    status: str
    usage: dict[str, int | None]


def response_usage(response: Any) -> dict[str, int | None]:
    usage = _get(response, "usage")
    input_details = _get(usage, "input_tokens_details", {})
    output_details = _get(usage, "output_tokens_details", {})
    return {
        "input_tokens": _get(usage, "input_tokens"),
        "cached_input_tokens": _get(input_details, "cached_tokens"),
        "output_tokens": _get(usage, "output_tokens"),
        "reasoning_tokens": _get(output_details, "reasoning_tokens"),
        "total_tokens": _get(usage, "total_tokens"),
    }


def parse_brief_response(response: Any) -> ParsedBriefResponse:
    """Parse only final ``output_text`` and reject refusal/incomplete output."""
    if response is None:
        raise BriefResponseError("missing_response", "The Responses API returned no response")

    for item in _walk(_get(response, "output", [])):
        if _get(item, "type") == "refusal" or _get(item, "refusal"):
            raise BriefResponseError("refusal", "The model refused the scientific brief request")

    status = str(_get(response, "status", "completed") or "completed")
    if status == "incomplete":
        reason = _get(_get(response, "incomplete_details", {}), "reason", "unknown")
        code = "output_token_limit" if reason == "max_output_tokens" else "incomplete_response"
        raise BriefResponseError(code, f"The model response was incomplete ({reason})")
    if status in {"failed", "cancelled", "queued", "in_progress"}:
        raise BriefResponseError("response_not_completed", f"The model response status was {status}")

    output_text = _get(response, "output_text")
    if not isinstance(output_text, str) or not output_text.strip():
        raise BriefResponseError("missing_output_text", "The model returned no final structured output text")
    try:
        payload = json.loads(output_text)
    except json.JSONDecodeError as exc:
        raise BriefResponseError("malformed_structured_output", "The model returned malformed JSON") from exc
    if not isinstance(payload, dict):
        raise BriefResponseError("invalid_structured_root", "The model output must be a JSON object")

    converted = to_json_compatible(payload)
    assert isinstance(converted, dict)
    return ParsedBriefResponse(
        payload=converted,
        response_id=_get(response, "id"),
        model=_get(response, "model"),
        status=status,
        usage=response_usage(response),
    )


def parse_structured_generation(response: StructuredGeneration) -> ParsedBriefResponse:
    """Parse final provider-neutral structured content, never private reasoning."""
    if response.status not in {"completed", "done", "success"}:
        raise BriefResponseError("incomplete_response", "The model response was incomplete")
    if not isinstance(response.content, str) or not response.content.strip():
        raise BriefResponseError("missing_output_text", "The model returned no final structured output text")
    try:
        payload = json.loads(response.content)
    except json.JSONDecodeError as exc:
        raise BriefResponseError("malformed_structured_output", "The model returned malformed JSON") from exc
    if not isinstance(payload, dict):
        raise BriefResponseError("invalid_structured_root", "The model output must be a JSON object")
    converted = to_json_compatible(payload)
    assert isinstance(converted, dict)
    return ParsedBriefResponse(
        payload=converted,
        response_id=response.response_id,
        model=response.model,
        status=response.status,
        usage=response.usage,
    )
