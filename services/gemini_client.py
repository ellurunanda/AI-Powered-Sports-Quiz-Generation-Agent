"""Gemini API integration for quiz generation."""

from __future__ import annotations

from typing import Any

import httpx

from config import Settings, get_settings
from utils import get_logger

LOGGER = get_logger(__name__)

try:
    from google import genai
except Exception as exc:  # pragma: no cover
    genai = None
    _IMPORT_EXCEPTION = exc
else:
    _IMPORT_EXCEPTION = None


class GeminiServiceError(Exception):
    """Raised when Gemini generation fails."""


class GeminiQuizGenerator:
    """Adapter over Gemini text generation for quiz payloads."""

    def __init__(self, settings: Settings | None = None) -> None:
        self.settings = settings or get_settings()
        self._client: Any = None

    def _ensure_client(self) -> Any:
        is_ready, message = self.settings.validate_generation_readiness()
        if not is_ready:
            raise GeminiServiceError(message)
        if genai is None:
            msg = "google-genai package is unavailable in the current environment."
            raise GeminiServiceError(msg) from _IMPORT_EXCEPTION
        if self._client is None:
            try:
                self._client = genai.Client(api_key=self.settings.gemini_api_key)
            except Exception as exc:
                msg = "Failed to initialize Gemini client."
                LOGGER.error(msg)
                raise GeminiServiceError(msg) from exc
        return self._client

    def generate_quiz_text(self, prompt: str) -> str:
        """Generate quiz JSON text from grounded prompt."""

        if not prompt.strip():
            raise ValueError("prompt cannot be empty.")

        if self.settings.llm_provider == "openai":
            return self._generate_with_openai(prompt)
        return self._generate_with_gemini(prompt)

    def _generate_with_gemini(self, prompt: str) -> str:
        client = self._ensure_client()
        last_exception: Exception | None = None

        for model_name in self.settings.gemini_candidate_models:
            for attempt in range(1, 3):
                try:
                    response = client.models.generate_content(
                        model=model_name,
                        contents=prompt,
                    )
                    output_text = _extract_text_from_response(response)
                    if not output_text.strip():
                        raise GeminiServiceError("Gemini returned empty output.")
                    if model_name != self.settings.gemini_model:
                        LOGGER.warning(
                            "Primary model '%s' unavailable; fallback model '%s' used.",
                            self.settings.gemini_model,
                            model_name,
                        )
                    return output_text.strip()
                except Exception as exc:  # noqa: BLE001
                    last_exception = exc
                    if _is_model_not_found_error(exc):
                        LOGGER.warning("Gemini model '%s' not found, trying fallback.", model_name)
                        break
                    if _is_transient_error(exc) and attempt < 2:
                        LOGGER.warning("Transient Gemini error on model '%s'. Retrying once...", model_name)
                        continue
                    detail = str(exc).strip()
                    msg = f"Gemini generation request failed: {detail or 'unknown error'}"
                    LOGGER.error(msg)
                    raise GeminiServiceError(msg) from exc

        detail = str(last_exception).strip() if last_exception else "no error detail"
        msg = f"Gemini generation failed. None of the configured models were available. Last error: {detail}"
        LOGGER.error(msg)
        raise GeminiServiceError(msg) from last_exception

    def _generate_with_openai(self, prompt: str) -> str:
        if not self.settings.openai_api_key:
            raise GeminiServiceError("Missing OPENAI_API_KEY.")

        url = f"{self.settings.openai_base_url.rstrip('/')}/chat/completions"
        payload = {
            "model": self.settings.openai_model,
            "messages": [{"role": "user", "content": prompt}],
            "temperature": 0.2,
        }
        headers = {
            "Authorization": f"Bearer {self.settings.openai_api_key}",
            "Content-Type": "application/json",
        }

        try:
            response = httpx.post(
                url,
                headers=headers,
                json=payload,
                timeout=self.settings.request_timeout_seconds,
            )
        except Exception as exc:  # noqa: BLE001
            msg = "OpenAI request failed due to network/runtime error."
            LOGGER.error(msg)
            raise GeminiServiceError(msg) from exc

        if response.status_code >= 400:
            detail = response.text.strip()
            msg = f"OpenAI request failed ({response.status_code}): {detail}"
            LOGGER.error(msg)
            raise GeminiServiceError(msg)

        data = response.json()
        content = (
            data.get("choices", [{}])[0]
            .get("message", {})
            .get("content", "")
        )
        if not isinstance(content, str) or not content.strip():
            raise GeminiServiceError("OpenAI returned empty output.")
        return content.strip()


def _extract_text_from_response(response: Any) -> str:
    text = getattr(response, "text", None)
    if isinstance(text, str) and text.strip():
        return text

    candidates = getattr(response, "candidates", None)
    if not candidates:
        return ""

    parts: list[str] = []
    for candidate in candidates:
        content = getattr(candidate, "content", None)
        content_parts = getattr(content, "parts", None) if content else None
        if not content_parts:
            continue
        for part in content_parts:
            part_text = getattr(part, "text", None)
            if isinstance(part_text, str):
                parts.append(part_text)
    return "\n".join(parts).strip()


def _is_model_not_found_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code == 404:
        return True
    message = str(exc).lower()
    return "404" in message or "not found" in message


def _is_transient_error(exc: Exception) -> bool:
    status_code = getattr(exc, "status_code", None)
    if status_code in {429, 500, 502, 503, 504}:
        return True
    message = str(exc).lower()
    return any(token in message for token in ("429", "rate limit", "quota", "timeout", "temporarily unavailable"))
