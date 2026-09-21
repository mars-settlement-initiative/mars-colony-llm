"""Gemini tests use intercepted HTTP requests; no network or credentials needed."""

import contextlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from gemini_client import GeminiClient
from llm_config import default_model
from llm_decision import DecisionEngine
from metrics import simulate
from model import MarsColonyModel
from test_decisions import observation


def response_body(action="COLLECT_6_5", finish="STOP"):
    return {
        "candidates": [{"finishReason": finish, "content": {"parts": [
            {"text": json.dumps({"action": action, "reason": "A nearby resource is available."})}
        ]}}],
        "usageMetadata": {"promptTokenCount": 80, "candidatesTokenCount": 12, "thoughtsTokenCount": 3},
    }


def make_engine(handler, **kwargs):
    client = GeminiClient("fake-test-key", httpx.Client(transport=httpx.MockTransport(handler)))
    return DecisionEngine("gemini", client=client, min_request_seconds=0, **kwargs)


class GeminiTests(unittest.TestCase):
    def test_default_model_and_missing_key(self):
        with patch.dict(os.environ, {}, clear=True):
            with self.assertRaisesRegex(ValueError, "GEMINI_API_KEY"):
                DecisionEngine("gemini")
        self.assertEqual(default_model("gemini"), "gemini-3.1-flash-lite")
        self.assertEqual(default_model("openai"), "gpt-4o-mini")

    def test_google_key_alias_and_provider_selection(self):
        with patch.dict(os.environ, {"GOOGLE_API_KEY": "fake-test-key"}, clear=True):
            engine = DecisionEngine("gemini")
            self.assertEqual(engine.model_name, "gemini-3.1-flash-lite")
            engine.close()

    def test_native_request_schema_key_and_usage(self):
        calls = []

        def respond(request):
            calls.append(request)
            self.assertEqual(request.url.host, "generativelanguage.googleapis.com")
            self.assertEqual(request.url.path, "/v1beta/models/gemini-3.1-flash-lite:generateContent")
            self.assertEqual(request.url.query, b"")
            self.assertEqual(request.headers["x-goog-api-key"], "fake-test-key")
            body = json.loads(request.content)
            self.assertNotIn("fake-test-key", request.content.decode())
            config = body["generationConfig"]
            self.assertEqual(config["responseMimeType"], "application/json")
            self.assertEqual(config["responseJsonSchema"]["properties"]["action"]["enum"], observation()["allowed_actions"])
            self.assertEqual(config["maxOutputTokens"], 300)
            self.assertNotIn("tools", body)
            obs = json.loads(body["contents"][0]["parts"][0]["text"])
            self.assertNotIn("baseline_target", obs)
            return httpx.Response(200, json=response_body())

        engine = make_engine(respond)
        self.assertEqual(engine.decide(observation())["action"], "COLLECT_6_5")
        self.assertEqual(len(calls), 1)
        self.assertEqual(engine.input_tokens, 80)
        self.assertEqual(engine.output_tokens, 15)
        self.assertEqual(engine.summary()["model"], "gemini-3.1-flash-lite")
        self.assertEqual(engine.records[-1]["source"], "llm")
        engine.close()

    def test_429_stops_requests_and_reports_quota(self):
        calls = []

        def quota(request):
            calls.append(request)
            return httpx.Response(429, json={"error": {"message": "sensitive provider message"}})

        engine = make_engine(quota)
        with contextlib.redirect_stdout(io.StringIO()):
            engine.decide(observation())
            engine.decide(observation())
        self.assertEqual(len(calls), 1)
        self.assertEqual(engine.fallbacks, 2)
        self.assertIn("quota exhausted", engine.records[0]["error"])
        self.assertNotIn("sensitive", json.dumps(engine.records))
        engine.close()

    def test_rejects_blocked_truncated_and_illegal_outputs(self):
        for body in [{"promptFeedback": {"blockReason": "SAFETY"}},
                     response_body(finish="MAX_TOKENS"),
                     response_body("COLLECT_999_999")]:
            with self.subTest(body=body):
                engine = make_engine(lambda request: httpx.Response(200, json=body))
                with contextlib.redirect_stdout(io.StringIO()):
                    engine.decide(observation())
                self.assertEqual(engine.llm_decisions, 0)
                self.assertEqual(engine.fallbacks, 1)
                engine.close()

    def test_pacing_uses_wall_time_without_changing_simulation_tick(self):
        engine = make_engine(lambda request: httpx.Response(200, json=response_body()), decision_interval=1)
        engine.min_request_seconds = 15
        with patch("llm_decision.time.monotonic", return_value=100), patch("llm_decision.time.sleep") as sleep:
            engine.decide(observation())
            engine.decide(observation())
            sleep.assert_called_once_with(15)
        self.assertEqual(engine.api_calls, 2)
        self.assertEqual(engine.records[-1]["step"], 1)
        engine.close()

    def test_continuation_invalidates_on_arrival_depletion_energy_and_expiry(self):
        for change in [{"position": [6, 5]}, {"known_resources": []},
                       {"energy": 1}, {"step": 11}, {"current_target": None}]:
            with self.subTest(change=change):
                engine = make_engine(lambda request: httpx.Response(200, json=response_body()))
                engine.decide(observation())
                obs = observation()
                obs.update(step=2, current_target=[6, 5])
                self.assertIsNotNone(engine.continued_choice(obs))
                obs.update(change)
                self.assertIsNone(engine.continued_choice(obs))
                engine.close()

    def test_valid_journey_continues_without_another_request(self):
        engine = make_engine(lambda request: httpx.Response(200, json=response_body()), max_calls=1)
        engine.decide(observation())
        obs = observation()
        obs.update(step=2, current_target=[6, 5])
        engine.decide(obs)
        self.assertEqual(engine.api_calls, 1)
        self.assertEqual(engine.records[-1]["source"], "continued_llm")
        self.assertEqual(engine.continued_decisions, 1)
        engine.close()

    def test_simulation_with_gemini_transport(self):
        def choose(request):
            obs = json.loads(json.loads(request.content)["contents"][0]["parts"][0]["text"])
            choices = obs["allowed_actions"]
            action = next((a for a in choices if a.startswith("COLLECT_")), choices[0])
            return httpx.Response(200, json=response_body(action))

        engine = make_engine(choose)
        model = MarsColonyModel(seed=42, decision_engine=engine)
        with contextlib.redirect_stdout(io.StringIO()):
            report = simulate(model, 100, 42)
        self.assertEqual(report["steps_survived"], 100)
        self.assertGreater(engine.llm_decisions, 0)
        self.assertGreater(engine.continued_decisions, 0)
        self.assertLessEqual(engine.api_calls, engine.max_calls)
        engine.close()


if __name__ == "__main__":
    unittest.main()
