from copy import deepcopy

import pytest

from src.brief_disclaimer import get_mandatory_disclaimer
from src.brief_language_validation import (
    brief_human_text,
    validate_brief_language,
)
from tests.scientific_brief_test_data import valid_brief


@pytest.mark.parametrize("language", ["es", "en"])
def test_professional_bilingual_briefs_pass(language: str) -> None:
    assert validate_brief_language(valid_brief(language), language) == []


def test_wrong_language_and_mixed_language_are_rejected() -> None:
    payload = deepcopy(valid_brief("en"))
    payload["language"] = "es"
    payload["disclaimer"] = get_mandatory_disclaimer("es")
    codes = {issue["code"] for issue in validate_brief_language(payload, "es")}
    assert "mixed_or_wrong_language" in codes
    assert "wrong_language_code" in {issue["code"] for issue in validate_brief_language(payload, "en")}


def test_mandatory_disclaimer_must_be_exact() -> None:
    payload = deepcopy(valid_brief("en"))
    payload["disclaimer"] = "Experimental only."
    assert "mandatory_disclaimer_mismatch" in {issue["code"] for issue in validate_brief_language(payload, "en")}
    assert "official classification" in get_mandatory_disclaimer("en")


@pytest.mark.parametrize(
    ("mutate", "expected_code", "expected_path"),
    [
        (
            lambda payload: payload.update({"executive_summary": "unexpected text"}),
            "invalid_section_type",
            "executive_summary",
        ),
        (
            lambda payload: payload["executive_summary"].update({"paragraphs": ["unexpected text"]}),
            "invalid_paragraph_type",
            "executive_summary.paragraphs.0",
        ),
        (
            lambda payload: payload["executive_summary"].update({"paragraphs": {"text": "unexpected"}}),
            "invalid_paragraphs_type",
            "executive_summary.paragraphs",
        ),
        (
            lambda payload: payload["executive_summary"]["paragraphs"][0].update({"text": ["unexpected"]}),
            "invalid_text_field_type",
            "executive_summary.paragraphs.0.text",
        ),
        (
            lambda payload: payload.update({"executive_summary": None}),
            "invalid_section_type",
            "executive_summary",
        ),
    ],
)
def test_malformed_human_text_shapes_return_issues_without_crashing(
    mutate, expected_code: str, expected_path: str,
) -> None:
    payload = deepcopy(valid_brief("en"))
    mutate(payload)
    issues = validate_brief_language(payload, "en")
    assert {item["code"] for item in issues} >= {expected_code}
    assert expected_path in {item["path"] for item in issues}
    assert isinstance(brief_human_text(payload), str)
