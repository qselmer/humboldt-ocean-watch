"""Deterministic Markdown rendering for validated structured briefs."""

from __future__ import annotations

import re
from typing import Any

from src.brief_disclaimer import get_mandatory_disclaimer
from src.brief_output_schema import NARRATIVE_SECTION_IDS, validate_output_schema


_SECTION_HEADINGS = {
    "es": {
        "executive_summary": "Resumen ejecutivo",
        "regional_state": "Estado térmico regional",
        "recent_evolution": "Evolución reciente",
        "spatial_structure": "Estructura espacial",
        "event_status": "Estado de los eventos térmicos",
        "data_quality": "Calidad de los datos",
        "key_messages": "Mensajes clave",
        "limitations": "Limitaciones",
        "disclaimer": "Descargo de responsabilidad",
        "facts": "Identificadores de hechos de respaldo",
    },
    "en": {
        "executive_summary": "Executive summary",
        "regional_state": "Regional thermal state",
        "recent_evolution": "Recent evolution",
        "spatial_structure": "Spatial structure",
        "event_status": "Thermal-event status",
        "data_quality": "Data quality",
        "key_messages": "Key messages",
        "limitations": "Limitations",
        "disclaimer": "Disclaimer",
        "facts": "Supporting fact IDs",
    },
}


def _climatology_reference(payload: dict[str, Any]) -> str:
    text = " ".join(
        str(item.get("text", ""))
        for item in payload.get("limitations", [])
        if item.get("limitation_id") == "reference_period"
    )
    match = re.search(r"\b\d{4}\s*[–—-]\s*\d{4}\b", text)
    return match.group(0) if match else "See limitations"


def render_scientific_brief_markdown(payload: dict[str, Any]) -> str:
    """Render a schema-valid brief without changing scientific content."""
    schema_issues = validate_output_schema(payload)
    if schema_issues:
        raise ValueError(f"Cannot render invalid structured brief: {schema_issues[0]['message']}")
    language = str(payload["language"])
    if payload.get("disclaimer") != get_mandatory_disclaimer(language):
        raise ValueError("Cannot render a brief with a non-authoritative disclaimer")
    headings = _SECTION_HEADINGS[language]
    product_status = "Experimental" if language == "en" else "Experimental"
    metadata_labels = (
        ("Analysis date", "Product status", "Climatological reference")
        if language == "en"
        else ("Fecha de análisis", "Estado del producto", "Referencia climatológica")
    )
    lines = [
        f"# {payload['title']}",
        "",
        f"- {metadata_labels[0]}: {payload['analysis_date']}",
        f"- {metadata_labels[1]}: {product_status}",
        f"- {metadata_labels[2]}: {_climatology_reference(payload)}",
        "",
    ]
    for section_id in NARRATIVE_SECTION_IDS:
        lines.extend([f"## {headings[section_id]}", ""])
        section = payload[section_id]
        if section["status"] == "unavailable":
            lines.extend([str(section["reason"]), ""])
        else:
            for paragraph in section["paragraphs"]:
                lines.extend([str(paragraph["text"]), ""])

    lines.extend([f"## {headings['key_messages']}", ""])
    for item in payload["key_messages"]:
        lines.append(f"- {item['text']}")
    lines.extend(["", f"## {headings['limitations']}", ""])
    for item in payload["limitations"]:
        lines.append(f"- {item['text']}")
    lines.extend(
        [
            "",
            f"## {headings['disclaimer']}",
            "",
            str(payload["disclaimer"]),
            "",
            f"## {headings['facts']}",
            "",
        ]
    )
    lines.extend(f"- `{fact_id}`" for fact_id in sorted(set(payload["facts_used"])))
    return "\n".join(lines).rstrip() + "\n"
