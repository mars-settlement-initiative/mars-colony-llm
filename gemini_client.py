"""Small Gemini Developer API adapter using its documented REST interface.

No Vertex AI, billing setup, model switching or retries. The caller paces and
counts requests. httpx is already used by the existing OpenAI dependency.
"""

import json
import re

import httpx

from llm_config import API_TIMEOUT_SECONDS, MAX_OUTPUT_TOKENS


class GeminiError(Exception):
    """A safe, fixed error description, without key or provider response text."""

    def __init__(self, message, permanent=False):
        super().__init__(message)
        self.permanent = permanent


class GeminiClient:
    def __init__(self, api_key, http_client=None):
        self._api_key = api_key
        self._http = http_client or httpx.Client(timeout=API_TIMEOUT_SECONDS)

    def generate(self, model, prompt, observation, schema):
        if not re.fullmatch(r"gemini-[A-Za-z0-9._-]+", model):
            raise GeminiError("Invalid Gemini model name", permanent=True)
        payload = {
            "systemInstruction": {"parts": [{"text": prompt}]},
            "contents": [{"role": "user", "parts": [{"text": json.dumps(observation)}]}],
            "generationConfig": {
                "responseMimeType": "application/json",
                "responseJsonSchema": schema,
                "maxOutputTokens": MAX_OUTPUT_TOKENS,
                # This is a small routing decision, so extra reasoning only adds
                # latency and can consume the output budget before JSON is emitted.
                "thinkingConfig": {"thinkingLevel": "minimal"},
                "temperature": 0,
            },
        }
        try:
            response = self._http.post(
                f"https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent",
                headers={"x-goog-api-key": self._api_key},
                json=payload,
                timeout=API_TIMEOUT_SECONDS,
            )
        except httpx.TimeoutException as exc:
            raise GeminiError("Gemini request timed out") from exc
        except httpx.RequestError as exc:
            raise GeminiError("Gemini could not be reached") from exc
        permanent_errors = {
            400: "Gemini rejected the request; check the model and API key",
            401: "Gemini API key was rejected",
            403: "Gemini access denied; check the key and project permissions",
            404: "Gemini model unavailable; check GEMINI_MODEL in AI Studio",
            429: "Gemini quota exhausted; check Free Tier limits in AI Studio before restarting",
        }
        if not response.is_success:
            if response.status_code in permanent_errors:
                raise GeminiError(permanent_errors[response.status_code], permanent=True)
            raise GeminiError(f"Gemini service error (HTTP {response.status_code})")
        try:
            data = response.json()
        except ValueError as exc:
            raise GeminiError("Gemini returned an invalid response") from exc
        usage = data.get("usageMetadata") or {}
        candidates = data.get("candidates") or []
        # Return usage even for incomplete/blocked generations, so it is counted.
        candidate = candidates[0] if candidates else {}
        text = "".join(part.get("text", "") for part in candidate.get("content", {}).get("parts", [])
                       if not part.get("thought"))
        return {
            "text": text, "complete": candidate.get("finishReason") == "STOP",
            "finish_reason": candidate.get("finishReason") or
                             (data.get("promptFeedback") or {}).get("blockReason") or
                             "NO_CANDIDATE",
            "input_tokens": usage.get("promptTokenCount", 0),
            "output_tokens": (usage.get("candidatesTokenCount", 0) or 0) + (usage.get("thoughtsTokenCount", 0) or 0),
        }

    def close(self):
        self._http.close()
