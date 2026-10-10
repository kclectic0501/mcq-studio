from datetime import UTC, datetime, timedelta
from email.utils import format_datetime
from types import SimpleNamespace

import pytest

from mcqgen.config import Settings
from mcqgen.llm import (
    ModelCallError,
    ModelConfigError,
    call_with_retry,
    get_model,
    parse_response,
    retry_after,
)
from mcqgen.schemas import Critique


def test_empty_hf_token_fails_before_request():
    settings = Settings(_env_file=None, hf_token="")
    with pytest.raises(ModelConfigError, match="HF_TOKEN"):
        get_model(settings)


def test_fenced_json_and_invalid_json():
    assert parse_response('```json\n{"passed":true,"issues":[]}\n```', Critique).passed
    with pytest.raises(ValueError):
        parse_response("This is not JSON", Critique)


@pytest.mark.parametrize(
    "content",
    [
        '<think>Reasoning with {braces}</think>\n```json\n{"passed":true,"issues":[]}\n```',
        'Here is the result: {"passed":true,"issues":[]}\nEnd of response.',
    ],
)
def test_wrapped_json_still_validates(content):
    assert parse_response(content, Critique).passed


@pytest.mark.parametrize(
    "content",
    [
        '<think>Unfinished reasoning {"passed":true}',
        '{"passed":true',
        '{"issues":[]}',
    ],
)
def test_invalid_or_incomplete_response_rejected(content):
    with pytest.raises(ValueError):
        parse_response(content, Critique)


def test_provider_key_loaded_from_env_file(tmp_path, monkeypatch: pytest.MonkeyPatch):
    env = tmp_path / ".env"
    env.write_text("LLM_PROVIDER=openai\nLLM_MODEL=test\nOPENAI_API_KEY=test-only\n")
    captured = {}

    def factory(name, **kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr("mcqgen.llm.init_chat_model", factory)
    get_model(Settings(_env_file=env))
    assert captured["api_key"] == "test-only"


class HTTPFailure(Exception):
    def __init__(self, status):
        self.status_code = status


def test_rate_limit_retries_then_succeeds():
    calls = []
    delays = []

    def request():
        calls.append(1)
        if len(calls) == 1:
            raise HTTPFailure(429)
        return "ok"

    assert call_with_retry(request, sleep=delays.append) == "ok"
    assert len(calls) == 2
    assert 1 <= delays[0] <= 1.5


@pytest.mark.parametrize("status", [401, 402, 403, 404])
def test_fatal_status_does_not_retry(status):
    delays = []

    def request():
        raise HTTPFailure(status)

    with pytest.raises(ModelConfigError, match=str(status)):
        call_with_retry(request, sleep=delays.append)
    assert delays == []


@pytest.mark.parametrize("error", [HTTPFailure(503), TimeoutError("timed out")])
def test_transient_exhaustion_is_bounded(error):
    calls = []
    delays = []

    def request():
        calls.append(1)
        raise error

    with pytest.raises(ModelCallError):
        call_with_retry(request, attempts=3, sleep=delays.append)
    assert len(calls) == 3
    assert len(delays) == 2


@pytest.mark.parametrize("error", [ValueError("invalid schema"), RuntimeError("bug")])
def test_schema_and_programming_errors_not_retried(error):
    def request():
        raise error

    with pytest.raises(type(error)):
        call_with_retry(request, sleep=lambda delay: pytest.fail("unexpected retry"))


def test_retry_after_is_respected():
    error = HTTPFailure(429)
    error.response = SimpleNamespace(headers={"retry-after": "5"})
    calls = []
    delays = []

    def request():
        calls.append(1)
        if len(calls) == 1:
            raise error
        return "ok"

    assert call_with_retry(request, sleep=delays.append) == "ok"
    assert 5 <= delays[0] <= 5.5


def test_retry_after_http_date_and_invalid_value():
    error = HTTPFailure(429)
    future = format_datetime(datetime.now(UTC) + timedelta(seconds=20), usegmt=True)
    error.response = SimpleNamespace(headers={"retry-after": future})
    assert 18 <= retry_after(error) <= 20
    for value in ["invalid", "NaN", "inf"]:
        error.response.headers["retry-after"] = value
        assert retry_after(error) is None


def test_long_retry_after_does_not_retry_early():
    error = HTTPFailure(429)
    error.response = SimpleNamespace(headers={"retry-after": "120"})

    def request():
        raise error

    with pytest.raises(ModelCallError, match="try again later"):
        call_with_retry(request, sleep=lambda delay: pytest.fail("unexpected retry"))
