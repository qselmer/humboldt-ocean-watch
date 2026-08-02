"""Deterministic Spanish and English language safeguards."""

from __future__ import annotations

import re
import unicodedata
from collections.abc import Mapping
from typing import Any

from src.brief_disclaimer import MANDATORY_DISCLAIMERS, get_mandatory_disclaimer
from src.brief_output_schema import NARRATIVE_SECTION_IDS

_SPANISH_MARKERS = {
    "el", "la", "los", "las", "del", "una", "con", "para", "que", "fue",
    "es", "anomalía", "cobertura", "espacial", "temperatura", "parche", "advertencia",
}
_ENGLISH_MARKERS = {
    "the", "this", "with", "for", "that", "was", "is", "anomaly", "coverage",
    "spatial", "temperature", "patch", "warning", "regional", "current",
}
_PROMOTIONAL = {
    "es": ("revolucionario", "sin precedentes", "garantizado", "definitivo"),
    "en": ("revolutionary", "unprecedented", "guaranteed", "definitive proof"),
}


def _normalize(text: str) -> str:
    return unicodedata.normalize("NFKC", text).casefold()


def _append_text(
    pieces: list[str],
    issues: list[dict[str, str]],
    value: Any,
    *,
    path: str,
) -> None:
    if isinstance(value, str):
        pieces.append(value)
    else:
        issues.append(
            {
                "code": "invalid_text_field_type",
                "path": path,
                "message": "Expected a text value",
            }
        )


def brief_human_text(
    payload: Mapping[str, Any],
    *,
    issues: list[dict[str, str]] | None = None,
) -> str:
    """Collect human-readable text without trusting model-produced types."""
    collected_issues = issues if issues is not None else []
    pieces: list[str] = []
    if not isinstance(payload, Mapping):
        collected_issues.append(
            {
                "code": "invalid_root_type",
                "path": "root",
                "message": "The brief root must be an object",
            }
        )
        return ""

    _append_text(pieces, collected_issues, payload.get("title"), path="title")
    _append_text(
        pieces,
        collected_issues,
        payload.get("disclaimer"),
        path="disclaimer",
    )
    for section_id in NARRATIVE_SECTION_IDS:
        section = payload.get(section_id)
        if not isinstance(section, Mapping):
            collected_issues.append(
                {
                    "code": "invalid_section_type",
                    "path": section_id,
                    "message": "A narrative section must be an object",
                }
            )
            continue
        _append_text(
            pieces,
            collected_issues,
            section.get("heading"),
            path=f"{section_id}.heading",
        )
        paragraphs = section.get("paragraphs")
        if not isinstance(paragraphs, list):
            collected_issues.append(
                {
                    "code": "invalid_paragraphs_type",
                    "path": f"{section_id}.paragraphs",
                    "message": "Section paragraphs must be an array",
                }
            )
        else:
            for index, paragraph in enumerate(paragraphs):
                paragraph_path = f"{section_id}.paragraphs.{index}"
                if not isinstance(paragraph, Mapping):
                    collected_issues.append(
                        {
                            "code": "invalid_paragraph_type",
                            "path": paragraph_path,
                            "message": "A paragraph entry must be an object",
                        }
                    )
                    continue
                _append_text(
                    pieces,
                    collected_issues,
                    paragraph.get("text"),
                    path=f"{paragraph_path}.text",
                )
        reason = section.get("reason")
        if reason is not None:
            _append_text(
                pieces,
                collected_issues,
                reason,
                path=f"{section_id}.reason",
            )

    for collection_name, item_code in (
        ("limitations", "invalid_limitation_type"),
        ("key_messages", "invalid_key_message_type"),
    ):
        collection = payload.get(collection_name)
        if not isinstance(collection, list):
            collected_issues.append(
                {
                    "code": "invalid_array_type",
                    "path": collection_name,
                    "message": "Expected an array",
                }
            )
            continue
        for index, item in enumerate(collection):
            item_path = f"{collection_name}.{index}"
            if not isinstance(item, Mapping):
                collected_issues.append(
                    {
                        "code": item_code,
                        "path": item_path,
                        "message": "Expected an object",
                    }
                )
                continue
            _append_text(
                pieces,
                collected_issues,
                item.get("text"),
                path=f"{item_path}.text",
            )

    generation_notes = payload.get("generation_notes")
    if not isinstance(generation_notes, list):
        collected_issues.append(
            {
                "code": "invalid_array_type",
                "path": "generation_notes",
                "message": "Expected an array",
            }
        )
    else:
        for index, item in enumerate(generation_notes):
            _append_text(
                pieces,
                collected_issues,
                item,
                path=f"generation_notes.{index}",
            )
    return "\n".join(pieces)


def validate_brief_language(
    payload: Mapping[str, Any], expected_language: str,
) -> list[dict[str, str]]:
    if expected_language not in {"es", "en"}:
        raise ValueError("Expected language must be 'es' or 'en'")
    issues: list[dict[str, str]] = []
    if not isinstance(payload, Mapping):
        return [
            {
                "code": "invalid_root_type",
                "path": "root",
                "message": "The brief root must be an object",
            }
        ]
    if payload.get("language") != expected_language:
        issues.append(
            {
                "code": "wrong_language_code",
                "path": "language",
                "message": f"Expected language {expected_language!r}",
            }
        )
    if payload.get("disclaimer") != get_mandatory_disclaimer(expected_language):
        issues.append(
            {
                "code": "mandatory_disclaimer_mismatch",
                "path": "disclaimer",
                "message": "The mandatory experimental-product disclaimer must be exact",
            }
        )

    text = _normalize(brief_human_text(payload, issues=issues))
    words = set(re.findall(r"[^\W\d_]+", text, flags=re.UNICODE))
    spanish = len(words & _SPANISH_MARKERS)
    english = len(words & _ENGLISH_MARKERS)
    expected_score, other_score = (spanish, english) if expected_language == "es" else (english, spanish)
    if other_score >= 4 and other_score > expected_score:
        issues.append(
            {
                "code": "mixed_or_wrong_language",
                "path": "root",
                "message": "The narrative contains substantial text in the other supported language",
            }
        )
    for phrase in _PROMOTIONAL[expected_language]:
        if phrase in text:
            issues.append(
                {
                    "code": "non_neutral_language",
                    "path": "root",
                    "message": f"Promotional or exaggerated wording is not allowed: {phrase}",
                }
            )
    return issues
