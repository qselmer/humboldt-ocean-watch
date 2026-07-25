from types import SimpleNamespace

import pytest

from src.brief_response_parser import BriefResponseError, parse_brief_response
from tests.scientific_brief_test_data import fake_response, valid_brief


def test_parse_valid_final_output_and_usage() -> None:
    parsed = parse_brief_response(fake_response(valid_brief("en")))
    assert parsed.payload["language"] == "en"
    assert parsed.usage["cached_input_tokens"] == 10
    assert parsed.usage["reasoning_tokens"] == 20


def test_refusal_is_not_parsed_as_brief() -> None:
    response = SimpleNamespace(status="completed", output=[{"type": "refusal", "refusal": "no"}], output_text="{}")
    with pytest.raises(BriefResponseError, match="refused") as error:
        parse_brief_response(response)
    assert error.value.code == "refusal"


@pytest.mark.parametrize(
    ("response", "code"),
    [
        (SimpleNamespace(status="incomplete", incomplete_details={"reason": "max_output_tokens"}, output=[]), "output_token_limit"),
        (SimpleNamespace(status="completed", output=[], output_text="{bad"), "malformed_structured_output"),
        (SimpleNamespace(status="completed", output=[], output_text="[]"), "invalid_structured_root"),
    ],
)
def test_incomplete_and_malformed_outputs(response, code: str) -> None:
    with pytest.raises(BriefResponseError) as error:
        parse_brief_response(response)
    assert error.value.code == code
