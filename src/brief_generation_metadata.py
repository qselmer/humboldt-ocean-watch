"""Sanitized metadata for stateless scientific brief generation."""

from __future__ import annotations

from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from typing import Any
from uuid import uuid4

from src.export_utils import to_json_compatible


@dataclass
class BriefGenerationMetadata:
    generation_id: str
    context_schema_version: str
    context_sha256: str
    prompt_version: str
    prompt_sha256: str
    model_requested: str
    model_returned: str | None
    language: str
    analysis_date: str
    response_id: str | None
    generated_at: str
    latency_seconds: float
    input_tokens: int | None
    cached_input_tokens: int | None
    output_tokens: int | None
    reasoning_tokens: int | None
    total_tokens: int | None
    retry_count: int
    repair_count: int
    store_setting: bool
    tools_enabled: bool
    validation_status: str
    validation_messages: list[str] = field(default_factory=list)
    provider: str = "openai"
    model: str | None = None
    base_url: str | None = None
    local_generation: bool = False
    external_network_used: bool = True
    think_enabled: bool = False
    num_ctx: int | None = None
    num_predict: int | None = None
    generation_duration_seconds: float | None = None
    load_duration: int | None = None
    total_duration: int | None = None
    prompt_eval_count: int | None = None
    prompt_eval_duration: int | None = None
    eval_count: int | None = None
    eval_duration: int | None = None
    structured_output_mode: str | None = None
    provider_schema_supplied: bool = False
    strict_local_schema_validation: bool = True
    schema_fallback_used: bool = False
    strict_schema_bytes: int | None = None
    provider_schema_bytes: int | None = None
    removed_schema_keyword_counts: dict[str, int] = field(default_factory=dict)
    converted_const_count: int = 0
    unresolved_reference_count: int = 0
    provider_schema_validation_result: str | None = None
    response_sha256: str | None = None
    generation_call_count: int = 1
    search_grounding_enabled: bool = False
    request_duration_seconds: float | None = None
    transport_attempt_count: int = 1
    transient_retry_count: int = 0
    transport_retry_delays_seconds: list[float] = field(default_factory=list)
    scientific_repair_count: int = 0
    final_provider_status: str | None = None
    automatic_function_calling_enabled: bool = False
    disclaimer_source: str = "application_constant"
    disclaimer_inserted: bool = False
    disclaimer_language: str | None = None
    raw_model_disclaimer_present: bool = False
    raw_model_disclaimer_matched: bool = False
    facts_used_source: str = "derived_from_claim_support"

    def to_dict(self) -> dict[str, Any]:
        converted = to_json_compatible(asdict(self))
        assert isinstance(converted, dict)
        return converted


def new_generation_metadata(
    *,
    context_schema_version: str,
    context_sha256: str,
    prompt_version: str,
    prompt_sha256: str,
    model_requested: str,
    model_returned: str | None,
    language: str,
    analysis_date: str,
    response_id: str | None,
    latency_seconds: float,
    usage: dict[str, int | None],
    retry_count: int,
    repair_count: int,
    validation_status: str,
    validation_messages: list[str],
    provider: str = "openai",
    base_url: str | None = None,
    local_generation: bool = False,
    external_network_used: bool = True,
    think_enabled: bool = False,
    num_ctx: int | None = None,
    num_predict: int | None = None,
    provider_metadata: dict[str, Any] | None = None,
    generation_call_count: int = 1,
    response_sha256: str | None = None,
    transport_attempt_count: int | None = None,
    transient_retry_count: int | None = None,
    transport_retry_delays_seconds: list[float] | None = None,
    final_provider_status: str | None = None,
    disclaimer_audit: dict[str, Any] | None = None,
    facts_used_source: str = "derived_from_claim_support",
) -> BriefGenerationMetadata:
    provider_metadata = provider_metadata or {}
    disclaimer_audit = disclaimer_audit or {}
    rounded_latency = round(float(latency_seconds), 3)
    return BriefGenerationMetadata(
        generation_id=f"GEN-{uuid4().hex}",
        context_schema_version=context_schema_version,
        context_sha256=context_sha256,
        prompt_version=prompt_version,
        prompt_sha256=prompt_sha256,
        model_requested=model_requested,
        model_returned=model_returned,
        language=language,
        analysis_date=analysis_date,
        response_id=response_id,
        generated_at=datetime.now(timezone.utc).isoformat(),
        latency_seconds=rounded_latency,
        input_tokens=usage.get("input_tokens"),
        cached_input_tokens=usage.get("cached_input_tokens"),
        output_tokens=usage.get("output_tokens"),
        reasoning_tokens=usage.get("reasoning_tokens"),
        total_tokens=usage.get("total_tokens"),
        retry_count=int(retry_count),
        repair_count=int(repair_count),
        store_setting=False,
        tools_enabled=False,
        validation_status=validation_status,
        validation_messages=list(validation_messages),
        provider=provider,
        model=model_returned or model_requested,
        base_url=base_url,
        local_generation=bool(local_generation),
        external_network_used=bool(external_network_used),
        think_enabled=bool(think_enabled),
        num_ctx=num_ctx,
        num_predict=num_predict,
        generation_duration_seconds=rounded_latency,
        load_duration=provider_metadata.get("load_duration"),
        total_duration=provider_metadata.get("total_duration"),
        prompt_eval_count=provider_metadata.get("prompt_eval_count"),
        prompt_eval_duration=provider_metadata.get("prompt_eval_duration"),
        eval_count=provider_metadata.get("eval_count"),
        eval_duration=provider_metadata.get("eval_duration"),
        structured_output_mode=provider_metadata.get("structured_output_mode", "json_schema"),
        provider_schema_supplied=bool(provider_metadata.get("provider_schema_supplied", True)),
        strict_local_schema_validation=bool(
            provider_metadata.get("strict_local_schema_validation", True)
        ),
        schema_fallback_used=bool(provider_metadata.get("schema_fallback_used", False)),
        strict_schema_bytes=provider_metadata.get("strict_schema_bytes"),
        provider_schema_bytes=provider_metadata.get("provider_schema_bytes"),
        removed_schema_keyword_counts=dict(provider_metadata.get("removed_keyword_counts", {})),
        converted_const_count=int(provider_metadata.get("converted_const_count", 0)),
        unresolved_reference_count=int(provider_metadata.get("unresolved_reference_count", 0)),
        provider_schema_validation_result=provider_metadata.get("provider_schema_validation_result"),
        response_sha256=response_sha256 or provider_metadata.get("response_sha256"),
        generation_call_count=int(generation_call_count),
        search_grounding_enabled=bool(
            provider_metadata.get("search_grounding_enabled", False)
        ),
        request_duration_seconds=provider_metadata.get("request_duration_seconds"),
        transport_attempt_count=int(
            transport_attempt_count
            if transport_attempt_count is not None
            else provider_metadata.get("transport_attempt_count", 1)
        ),
        transient_retry_count=int(
            transient_retry_count
            if transient_retry_count is not None
            else provider_metadata.get("transient_retry_count", retry_count)
        ),
        transport_retry_delays_seconds=[
            float(value)
            for value in (
                transport_retry_delays_seconds
                if transport_retry_delays_seconds is not None
                else provider_metadata.get("transport_retry_delays_seconds", [])
            )
        ],
        scientific_repair_count=int(repair_count),
        final_provider_status=(
            final_provider_status
            or provider_metadata.get("final_provider_status")
            or "completed"
        ),
        automatic_function_calling_enabled=bool(
            provider_metadata.get("automatic_function_calling_enabled", False)
        ),
        disclaimer_source=str(
            disclaimer_audit.get("disclaimer_source", "application_constant")
        ),
        disclaimer_inserted=bool(disclaimer_audit.get("disclaimer_inserted", False)),
        disclaimer_language=disclaimer_audit.get("disclaimer_language"),
        raw_model_disclaimer_present=bool(
            disclaimer_audit.get("raw_model_disclaimer_present", False)
        ),
        raw_model_disclaimer_matched=bool(
            disclaimer_audit.get("raw_model_disclaimer_matched", False)
        ),
        facts_used_source=str(facts_used_source),
    )
