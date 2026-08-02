"""Deterministic fixtures shared by Increment 5B tests."""

from __future__ import annotations

from copy import deepcopy
import json
from pathlib import Path
from types import SimpleNamespace
from typing import Any

from src.brief_disclaimer import get_mandatory_disclaimer


ROOT = Path(__file__).resolve().parents[1]


def validated_context() -> dict[str, Any]:
    return json.loads(
        (ROOT / "outputs/briefs/brief_context_latest.json").read_text(encoding="utf-8")
    )


def generation_settings() -> dict[str, Any]:
    import yaml

    return yaml.safe_load((ROOT / "config.yaml").read_text(encoding="utf-8"))["brief_generation"]


def _paragraph(
    paragraph_id: str,
    text: str,
    facts: list[str],
    *,
    flags: list[str] | None = None,
    claim: str = "observation",
) -> dict[str, Any]:
    return {
        "paragraph_id": paragraph_id,
        "text": text,
        "supporting_fact_ids": facts,
        "quality_flag_ids": flags or [],
        "claim_types": [claim],
    }


def valid_brief(language: str = "en", context: dict[str, Any] | None = None) -> dict[str, Any]:
    context = context or validated_context()
    spanish = language == "es"
    text = {
        "title": "Informe científico térmico diario" if spanish else "Daily thermal scientific brief",
        "exec": (
            "La anomalía regional actual es positiva y la estructura espacial presenta variabilidad."
            if spanish else
            "The current regional anomaly is positive and the spatial structure shows variability."
        ),
        "regional": (
            "La temperatura y la cobertura regional válida se resumen con los hechos validados."
            if spanish else
            "Regional temperature and valid coverage are summarized from validated facts."
        ),
        "recent": (
            "El cambio reciente observado se describe sin extrapolar una tendencia futura."
            if spanish else
            "The observed recent change is described without extrapolating a future trend."
        ),
        "spatial": (
            "La variabilidad espacial distingue la media regional de la estructura interna."
            if spanish else
            "Spatial variability distinguishes the regional mean from internal structure."
        ),
        "event": (
            "El evento univariado, el parche diario, el track y la familia se mantienen como conceptos distintos."
            if spanish else
            "The univariate event, daily patch, track, and family remain distinct concepts."
        ),
        "quality": (
            "La cobertura válida es adecuada; una advertencia indica contacto con el límite del dominio."
            if spanish else
            "Valid coverage is adequate; a warning notes contact with the domain boundary."
        ),
        "message": (
            "La anomalía regional es positiva en la fecha analizada."
            if spanish else
            "The regional anomaly is positive on the analysis date."
        ),
        "note": (
            "Contenido limitado al contexto validado."
            if spanish else
            "Content is limited to the validated context."
        ),
    }
    facts_by_section = {
        "executive_summary": ["regional.mean_anomaly_c", "spatial.spatial_standard_deviation_c"],
        "regional_state": ["regional.mean_sst_c", "regional.valid_coverage"],
        "recent_evolution": ["recent.7d.change_mean_anomaly_c"],
        "spatial_structure": ["spatial.spatial_standard_deviation_c"],
        "event_status": ["event.event_day", "patches.patch_count", "activity.active_track_count", "activity.active_family_count"],
        "data_quality": ["regional.valid_coverage"],
    }
    section_text = {
        "executive_summary": text["exec"],
        "regional_state": text["regional"],
        "recent_evolution": text["recent"],
        "spatial_structure": text["spatial"],
        "event_status": text["event"],
        "data_quality": text["quality"],
    }
    flag_ids = [
        str(flag["flag_id"])
        for flag in context.get("quality_flags", [])
        if flag.get("severity") in {"warning", "error"}
    ]
    headings_es = {
        "executive_summary": "Resumen ejecutivo",
        "regional_state": "Estado térmico regional",
        "recent_evolution": "Evolución reciente",
        "spatial_structure": "Estructura espacial",
        "event_status": "Estado de eventos",
        "data_quality": "Calidad de datos",
    }
    headings_en = {
        "executive_summary": "Executive summary",
        "regional_state": "Regional thermal state",
        "recent_evolution": "Recent evolution",
        "spatial_structure": "Spatial structure",
        "event_status": "Event status",
        "data_quality": "Data quality",
    }
    brief: dict[str, Any] = {
        "schema_version": "1.0.0",
        "language": language,
        "analysis_date": context["analysis"]["resolved_analysis_date"],
        "title": text["title"],
    }
    for section_id in facts_by_section:
        brief[section_id] = {
            "heading": (headings_es if spanish else headings_en)[section_id],
            "paragraphs": [
                _paragraph(
                    f"{section_id}-1",
                    section_text[section_id],
                    facts_by_section[section_id],
                    flags=flag_ids if section_id == "data_quality" else [],
                    claim="quality_caveat" if section_id == "data_quality" else "observation",
                )
            ],
            "status": "available",
            "reason": None,
        }
    limitations = []
    for item in context.get("limitations", []):
        limitations.append(
            {
                "limitation_id": item["limitation_id"],
                "text": item["statement_es" if spanish else "statement_en"],
                "supporting_fact_ids": [],
                "quality_flag_ids": [],
            }
        )
    brief["limitations"] = limitations
    brief["key_messages"] = [
        {
            "message_id": "key-1",
            "text": text["message"],
            "supporting_fact_ids": ["regional.mean_anomaly_c"],
            "quality_flag_ids": [],
            "claim_types": ["observation"],
        }
    ]
    brief["disclaimer"] = get_mandatory_disclaimer(language)
    brief["facts_used"] = sorted(
        {
            fact
            for section_id in facts_by_section
            for fact in facts_by_section[section_id]
        }
    )
    brief["generation_notes"] = [text["note"]]
    return brief


def fake_response(payload: dict[str, Any], *, response_id: str = "resp_test") -> Any:
    return SimpleNamespace(
        id=response_id,
        model="gpt-5.6",
        status="completed",
        output_text=json.dumps(payload, ensure_ascii=False),
        output=[],
        usage=SimpleNamespace(
            input_tokens=100,
            output_tokens=50,
            total_tokens=150,
            input_tokens_details=SimpleNamespace(cached_tokens=10),
            output_tokens_details=SimpleNamespace(reasoning_tokens=20),
        ),
    )


class FakeResponses:
    def __init__(self, responses: list[Any]) -> None:
        self.queue = list(responses)
        self.calls: list[dict[str, Any]] = []

    def create(self, **kwargs: Any) -> Any:
        self.calls.append(deepcopy(kwargs))
        if not self.queue:
            raise AssertionError("Unexpected OpenAI call")
        result = self.queue.pop(0)
        if isinstance(result, Exception):
            raise result
        return result


class FakeClient:
    def __init__(self, responses: list[Any]) -> None:
        self.responses = FakeResponses(responses)
