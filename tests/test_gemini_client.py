"""Unit tests for Gemini API service wrapper."""

from __future__ import annotations

import pytest

from config import Settings
from services.gemini_client import GeminiQuizGenerator, GeminiServiceError, _extract_text_from_response


class _FakeResponse:
    def __init__(self, text: str) -> None:
        self.text = text


class _FakeModels:
    def generate_content(self, *, model: str, contents: str):
        return _FakeResponse('{"status":"ok"}')


class _FakeClient:
    def __init__(self, api_key: str) -> None:
        self.api_key = api_key
        self.models = _FakeModels()


def test_generate_quiz_text_requires_api_key(monkeypatch) -> None:
    monkeypatch.setattr("services.gemini_client.genai", type("G", (), {"Client": _FakeClient}))
    service = GeminiQuizGenerator(settings=Settings(llm_provider="gemini", gemini_api_key=None))
    with pytest.raises(GeminiServiceError, match="Missing GEMINI_API_KEY"):
        service.generate_quiz_text("hello")


def test_generate_quiz_text_success(monkeypatch) -> None:
    monkeypatch.setattr("services.gemini_client.genai", type("G", (), {"Client": _FakeClient}))
    service = GeminiQuizGenerator(
        settings=Settings(llm_provider="gemini", gemini_api_key="key", gemini_model="gemini-x")
    )
    result = service.generate_quiz_text("prompt")
    assert result == '{"status":"ok"}'


def test_generate_quiz_text_uses_fallback_model_on_404(monkeypatch) -> None:
    class _NotFoundError(Exception):
        status_code = 404

    class _FallbackModels:
        def __init__(self) -> None:
            self.calls = []

        def generate_content(self, *, model: str, contents: str):
            self.calls.append(model)
            if model == "bad-model":
                raise _NotFoundError("404 model not found")
            return _FakeResponse('{"status":"ok"}')

    class _FallbackClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.models = _FallbackModels()

    monkeypatch.setattr("services.gemini_client.genai", type("G", (), {"Client": _FallbackClient}))
    service = GeminiQuizGenerator(
        settings=Settings(
            llm_provider="gemini",
            gemini_api_key="key",
            gemini_model="bad-model",
            gemini_fallback_models="good-model",
        )
    )
    result = service.generate_quiz_text("prompt")
    assert result == '{"status":"ok"}'


def test_generate_quiz_text_includes_detailed_error_message(monkeypatch) -> None:
    class _BrokenModels:
        def generate_content(self, *, model: str, contents: str):
            raise RuntimeError("permission denied")

    class _BrokenClient:
        def __init__(self, api_key: str) -> None:
            self.api_key = api_key
            self.models = _BrokenModels()

    monkeypatch.setattr("services.gemini_client.genai", type("G", (), {"Client": _BrokenClient}))
    service = GeminiQuizGenerator(
        settings=Settings(llm_provider="gemini", gemini_api_key="key", gemini_model="gemini-x")
    )
    with pytest.raises(GeminiServiceError, match="permission denied"):
        service.generate_quiz_text("prompt")


def test_extract_text_from_response_fallback_candidates() -> None:
    class _Part:
        def __init__(self, text: str) -> None:
            self.text = text

    class _Content:
        def __init__(self, parts) -> None:
            self.parts = parts

    class _Candidate:
        def __init__(self, content) -> None:
            self.content = content

    response = type(
        "Resp",
        (),
        {"text": "", "candidates": [_Candidate(_Content([_Part("A"), _Part("B")]))]},
    )()
    assert _extract_text_from_response(response) == "A\nB"


def test_openai_provider_success(monkeypatch) -> None:
    class _Response:
        status_code = 200

        def json(self):
            return {
                "choices": [
                    {
                        "message": {
                            "content": '{"status":"ok"}',
                        }
                    }
                ]
            }

        text = ""

    def _fake_post(url, headers, json, timeout):
        assert "/chat/completions" in url
        assert headers["Authorization"].startswith("Bearer ")
        assert json["model"] == "gpt-4o-mini"
        return _Response()

    monkeypatch.setattr("services.gemini_client.httpx.post", _fake_post)
    service = GeminiQuizGenerator(
        settings=Settings(
            llm_provider="openai",
            openai_api_key="openai-key",
            openai_model="gpt-4o-mini",
        )
    )
    result = service.generate_quiz_text("prompt")
    assert result == '{"status":"ok"}'


def test_openai_provider_missing_key() -> None:
    service = GeminiQuizGenerator(
        settings=Settings(llm_provider="openai", openai_api_key=None),
    )
    with pytest.raises(GeminiServiceError, match="Missing OPENAI_API_KEY"):
        service.generate_quiz_text("prompt")
