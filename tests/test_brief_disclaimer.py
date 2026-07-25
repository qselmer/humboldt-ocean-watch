from copy import deepcopy
import json

import pytest

from src.brief_disclaimer import (
    DISCLAIMER_SOURCE,
    MandatoryDisclaimerConfigurationError,
    get_mandatory_disclaimer,
    insert_mandatory_disclaimer,
)
from src.brief_language_validation import validate_brief_language
from tests.scientific_brief_test_data import valid_brief


def test_authoritative_disclaimers_are_deterministic_and_exact() -> None:
    assert get_mandatory_disclaimer("es") == (
        "Este producto es experimental y no constituye una clasificación oficial "
        "de la magnitud de El Niño Costero."
    )
    assert get_mandatory_disclaimer("en") == (
        "This is an experimental product and does not constitute an official "
        "classification of Coastal El Niño magnitude."
    )
    assert get_mandatory_disclaimer("es") == get_mandatory_disclaimer("es")
    assert get_mandatory_disclaimer("en") == get_mandatory_disclaimer("en")


@pytest.mark.parametrize(
    ("raw_value", "present", "matched"),
    [
        (None, False, False),
        ("This experimental product is not an official classification.", True, False),
        ("This is an experimental product and does not constitute an official classification of Coastal El Niño magnitude!", True, False),
        ("  This is an experimental product and does not constitute an official classification of Coastal El Niño magnitude.  ", True, False),
        (get_mandatory_disclaimer("en"), True, True),
    ],
)
def test_disclaimer_insertion_replaces_missing_or_inexact_model_value(
    raw_value: str | None, present: bool, matched: bool,
) -> None:
    payload = {"scientific_text": "Do not normalize this  text."}
    if present:
        payload["disclaimer"] = raw_value
    audit = insert_mandatory_disclaimer(payload, "en")
    assert payload["disclaimer"] == get_mandatory_disclaimer("en")
    assert payload["scientific_text"] == "Do not normalize this  text."
    assert audit.disclaimer_source == DISCLAIMER_SOURCE
    assert audit.disclaimer_inserted is True
    assert audit.raw_model_disclaimer_present is present
    assert audit.raw_model_disclaimer_matched is matched
    if isinstance(raw_value, str):
        assert raw_value not in json.dumps(audit.to_dict())


def test_validator_and_injector_use_the_same_authoritative_source() -> None:
    payload = valid_brief("es")
    payload["disclaimer"] = "paráfrasis"
    audit = insert_mandatory_disclaimer(payload, "es")
    assert audit.raw_model_disclaimer_matched is False
    assert "mandatory_disclaimer_mismatch" not in {
        issue["code"] for issue in validate_brief_language(payload, "es")
    }


def test_insertion_modifies_no_other_payload_field() -> None:
    payload = valid_brief("en")
    payload["disclaimer"] = "wrong"
    before = deepcopy(payload)
    insert_mandatory_disclaimer(payload, "en")
    before.pop("disclaimer")
    after = deepcopy(payload)
    after.pop("disclaimer")
    assert after == before


def test_strict_validator_still_rejects_every_inexact_disclaimer() -> None:
    payload = valid_brief("en")
    payload["disclaimer"] = get_mandatory_disclaimer("en") + " "
    assert "mandatory_disclaimer_mismatch" in {
        issue["code"] for issue in validate_brief_language(payload, "en")
    }


def test_unsupported_disclaimer_language_fails_locally() -> None:
    with pytest.raises(
        MandatoryDisclaimerConfigurationError,
        match="No mandatory disclaimer is configured",
    ):
        get_mandatory_disclaimer("fr")
