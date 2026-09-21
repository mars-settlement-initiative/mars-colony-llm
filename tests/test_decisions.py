"""Offline tests, including the real OpenAI SDK with a fake HTTP transport."""

import contextlib
import io
import json
import os
import sys
import unittest
from pathlib import Path
from unittest.mock import patch

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import httpx
from openai import OpenAI

from agents import CollectorAgent, ResourceAgent, RobotAgent
from llm_decision import DecisionEngine, baseline_decision, validate_decision
from metrics import simulate
from model import MarsColonyModel


def observation():
    return {
        "step": 1, "agent_id": 15, "position": [5, 5],
        "base_position": [10, 10], "energy": 60, "max_energy": 100,
        "cargo": 0, "capacity": 10, "current_target": None,
        "known_resources": [[6, 5]], "other_collector_intentions": [],
        "allowed_actions": ["WAIT", "RETURN_BASE", "COLLECT_6_5"],
        "return_threshold": 25, "baseline_target": [6, 5],
    }


def api_response(action="COLLECT_6_5", status="completed"):
    return {
        "id": "resp_test", "object": "response", "created_at": 0,
        "status": status, "model": "test-model", "error": None,
        "incomplete_details": None, "instructions": None,
        "output": [{"id": "msg_test", "type": "message", "role": "assistant",
                    "status": "completed", "content": [{"type": "output_text",
                    "text": json.dumps({"action": action, "reason": "Nearby resource."}),
                    "annotations": []}]}],
        "usage": {"input_tokens": 50, "output_tokens": 10, "total_tokens": 60},
        "parallel_tool_calls": False, "tools": [], "tool_choice": "auto",
    }


class DecisionTests(unittest.TestCase):
    def test_sdk_request_validation_and_budget(self):
        requests = []

        def respond(request):
            payload = json.loads(request.content)
            requests.append(payload)
            self.assertEqual(request.url.path, "/v1/responses")
            self.assertEqual(payload["text"]["format"]["schema"]["properties"]["action"]["enum"], observation()["allowed_actions"])
            self.assertNotIn("baseline_target", json.loads(payload["input"]))
            self.assertFalse(payload["store"])
            return httpx.Response(200, json=api_response())

        with OpenAI(api_key="test-not-a-real-key", max_retries=0,
                    http_client=httpx.Client(transport=httpx.MockTransport(respond))) as client:
            engine = DecisionEngine("openai", "test-model", 1, client=client)
            self.assertEqual(engine.decide(observation())["action"], "COLLECT_6_5")
            with contextlib.redirect_stdout(io.StringIO()):
                engine.decide(observation())
            self.assertEqual(len(requests), 1)
            self.assertEqual(engine.llm_decisions, 1)
            self.assertEqual(engine.fallbacks, 1)
            self.assertEqual(engine.input_tokens, 50)
            self.assertEqual(engine.summary()["result_label"], "llm_with_fallback")

    def test_illegal_action_and_provider_error_disable_calls(self):
        for response in [httpx.Response(200, json=api_response("COLLECT_99_99")),
                         httpx.Response(200, json=api_response(status="incomplete")),
                         httpx.Response(401, json={"error": {"message": "test", "type": "authentication_error"}})]:
            with self.subTest(status=response.status_code):
                with OpenAI(api_key="test-not-a-real-key", max_retries=0,
                            http_client=httpx.Client(transport=httpx.MockTransport(lambda request: response))) as client:
                    engine = DecisionEngine("openai", client=client)
                    with contextlib.redirect_stdout(io.StringIO()):
                        first = engine.decide(observation())
                        engine.decide(observation())
                    self.assertEqual(first, baseline_decision(observation()))
                    self.assertEqual(engine.api_calls, 1)
                    self.assertEqual(engine.fallbacks, 2)

    def test_local_validation_rejects_invalid_shapes(self):
        for bad in [None, [], {"action": "WAIT"},
                    {"action": ["WAIT"], "reason": "x"},
                    {"action": "WAIT", "reason": ""},
                    {"action": "COLLECT_99_99", "reason": "x"}]:
            with self.assertRaises(ValueError):
                validate_decision(bad, observation())

    def test_no_api_call_for_forced_action(self):
        engine = DecisionEngine("openai", client=object())
        obs = observation()
        obs.update(cargo=10, allowed_actions=["RETURN_BASE"])
        self.assertEqual(engine.decide(obs)["action"], "RETURN_BASE")
        self.assertEqual(engine.api_calls, 0)
        self.assertEqual(engine.records[-1]["source"], "forced_action")

    def test_key_is_required_only_for_live_mode(self):
        with patch.dict(os.environ, {}, clear=True):
            DecisionEngine("mock")
            with self.assertRaisesRegex(ValueError, "OPENAI_API_KEY"):
                DecisionEngine("openai")

    def test_observation_hides_undiscovered_deposits(self):
        model = MarsColonyModel(seed=42, decision_engine=DecisionEngine("mock"))
        collector = next(a for a in model.agents if isinstance(a, CollectorAgent))
        obs = collector.create_observation()
        self.assertEqual(obs["known_resources"], [])
        self.assertEqual(obs["allowed_actions"], ["WAIT"])
        self.assertNotIn("resource_amount", obs)
        resource = next(a for a in model.agents if isinstance(a, ResourceAgent))
        model.known_resources.add(resource.pos)
        self.assertEqual(collector.create_observation()["known_resources"], [list(resource.pos)])

    def test_one_action_moves_one_cell_and_uses_one_energy(self):
        model = MarsColonyModel(seed=42, decision_engine=DecisionEngine("mock"))
        collector = next(a for a in model.agents if isinstance(a, CollectorAgent))
        model.known_resources.add((15, 15))
        collector.step()
        self.assertEqual(collector.pos, (11, 10))
        self.assertEqual(collector.energy, 99)
        self.assertEqual(collector.distance_travelled, 1)
        self.assertEqual(collector.cargo, 0)

    def test_mock_preserves_original_trajectories(self):
        for seed in [10, 20, 30, 40, 50]:
            with self.subTest(seed=seed), contextlib.redirect_stdout(io.StringIO()):
                original = MarsColonyModel(seed=seed, decision_engine=DecisionEngine("mock"))
                mock = MarsColonyModel(seed=seed, decision_engine=DecisionEngine("mock"))
                for agent in original.agents:
                    if isinstance(agent, CollectorAgent):
                        agent.step = agent.baseline_step
                for _ in range(500):
                    if not original.running:
                        break
                    original.step()
                    mock.step()
                    self.assertEqual(original.base_resources, mock.base_resources)
                    self.assertEqual(original.known_resources, mock.known_resources)
                    def states(model):
                        return [(a.unique_id, a.pos, a.energy, a.active,
                                 getattr(a, "cargo", 0), getattr(a, "target_resource", None))
                                for a in model.agents if isinstance(a, RobotAgent)]
                    self.assertEqual(states(original), states(mock))
                self.assertEqual(original.running, mock.running)

    def test_resource_conservation_and_capacity(self):
        model = MarsColonyModel(seed=42, decision_engine=DecisionEngine("mock"))
        initial = model.base_resources + sum(a.amount for a in model.agents if isinstance(a, ResourceAgent))
        with contextlib.redirect_stdout(io.StringIO()):
            report = simulate(model, 500, 42)
        remaining = model.base_resources + sum(a.amount for a in model.agents if isinstance(a, ResourceAgent))
        remaining += sum(a.cargo for a in model.agents if isinstance(a, CollectorAgent))
        self.assertEqual(initial, remaining + report["steps_survived"] * model.consumption_rate)
        self.assertTrue(all(0 <= a.cargo <= a.capacity for a in model.agents if isinstance(a, CollectorAgent)))


if __name__ == "__main__":
    unittest.main()

