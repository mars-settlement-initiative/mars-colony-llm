"""Observation -> validated action. No model mutation or generated code here."""

import hashlib
import json
import math
import os
import time
from pathlib import Path

from llm_config import (
    API_TIMEOUT_SECONDS, LLM_MODE, MAX_API_CALLS, MAX_OUTPUT_TOKENS,
    GEMINI_MIN_REQUEST_SECONDS, GEMINI_DECISION_INTERVAL, default_model,
)


def collect_action(position):
    return f"COLLECT_{position[0]}_{position[1]}"


def baseline_decision(observation):
    """The original collector policy, also used as an explicit fallback."""
    if observation["energy"] < observation["return_threshold"] or observation["cargo"]:
        action = "RETURN_BASE"
    elif observation["baseline_target"] is not None:
        action = collect_action(observation["baseline_target"])
    else:
        action = "WAIT"
    return {"action": action, "reason": "Original threshold/nearest-resource policy."}


def validate_decision(decision, observation):
    """Validate again locally, even when the API promises structured output."""
    if not isinstance(decision, dict) or set(decision) != {"action", "reason"}:
        raise ValueError("Expected exactly action and reason.")
    if not isinstance(decision["action"], str) or decision["action"] not in observation["allowed_actions"]:
        raise ValueError("Action is not available in this observation.")
    if not isinstance(decision["reason"], str) or not 1 <= len(decision["reason"]) <= 500:
        raise ValueError("Expected a short explanation.")
    return decision


class DecisionEngine:
    """One engine per simulation; its budget is shared by all collectors.

    Mock mode exercises the same observation/action interface with coded rules.
    It is NOT an LLM and must never be reported as an LLM result.
    """

    def __init__(self, mode=LLM_MODE, model_name=None,
                 max_calls=MAX_API_CALLS, client=None, prompt_path=None,
                 min_request_seconds=None, decision_interval=None):
        if mode not in {"mock", "openai", "gemini"}:
            raise ValueError("Mode must be mock, gemini or openai.")
        if max_calls < 0:
            raise ValueError("API call budget must be nonnegative.")
        self.mode = mode
        self.model_name = model_name or default_model(mode)
        self.min_request_seconds = (GEMINI_MIN_REQUEST_SECONDS if mode == "gemini" else 0) if min_request_seconds is None else min_request_seconds
        self.decision_interval = (GEMINI_DECISION_INTERVAL if mode == "gemini" else 1) if decision_interval is None else decision_interval
        if not math.isfinite(self.min_request_seconds) or self.min_request_seconds < 0:
            raise ValueError("Request interval must be a finite nonnegative number.")
        if self.decision_interval < 1:
            raise ValueError("Decision interval must be at least 1.")
        self._next_request_at = 0
        self._last_llm_choice = {}
        self.continued_decisions = 0
        self.pacing_seconds = 0.0
        self.max_calls = max_calls
        self.client = client
        self.prompt = Path(prompt_path or Path(__file__).with_name("prompt.txt")).read_text(encoding="utf-8")
        self.prompt_hash = hashlib.sha256(self.prompt.encode()).hexdigest()[:12]
        self.api_calls = 0
        self.llm_decisions = 0
        self.fallbacks = 0
        self.input_tokens = 0
        self.output_tokens = 0
        self.records = []
        self.disabled_reason = None
        if mode == "gemini" and client is None:
            key = os.getenv("GEMINI_API_KEY") or os.getenv("GOOGLE_API_KEY")
            if not key:
                raise ValueError("Gemini needs GEMINI_API_KEY from a Free Tier Google AI Studio project. Use --mode mock for the offline demo.")
            from gemini_client import GeminiClient
            self.client = GeminiClient(key)
        if mode == "openai" and client is None:
            if not os.getenv("OPENAI_API_KEY"):
                raise ValueError("OpenAI mode needs OPENAI_API_KEY. Use --mode mock for the offline demo.")
            try:
                from openai import OpenAI
            except ImportError as exc:
                raise ValueError("Install mars_colony_3/requirements.txt for OpenAI mode.") from exc
            # Disable SDK retries so the visible call budget bounds attempts.
            self.client = OpenAI(timeout=API_TIMEOUT_SECONDS, max_retries=0)

    def request_decision(self, observation):
        schema = {
            "type": "object",
            "properties": {
                "action": {"type": "string", "enum": observation["allowed_actions"]},
                "reason": {"type": "string"},
            },
            "required": ["action", "reason"],
            "additionalProperties": False,
        }
        # baseline_target is used only by the demo/fallback; do not give the
        # LLM a suggested answer. All factual state is still in the observation.
        llm_observation = {k: v for k, v in observation.items() if k not in {"baseline_target", "return_threshold"}}
        llm_observation["decision_interval_ticks"] = self.decision_interval
        if self.mode == "gemini":
            response = self.client.generate(self.model_name, self.prompt, llm_observation, schema)
            self.input_tokens += response["input_tokens"] or 0
            self.output_tokens += response["output_tokens"] or 0
            if not response["complete"] or not response["text"]:
                raise ValueError("Gemini response incomplete or blocked.")
            return json.loads(response["text"])
        response = self.client.responses.create(
            model=self.model_name,
            instructions=self.prompt,
            input=json.dumps(llm_observation),
            text={"format": {"type": "json_schema", "name": "collector_decision", "strict": True, "schema": schema}},
            max_output_tokens=MAX_OUTPUT_TOKENS,
            store=False,
        )
        usage = getattr(response, "usage", None)
        self.input_tokens += getattr(usage, "input_tokens", 0) or 0
        self.output_tokens += getattr(usage, "output_tokens", 0) or 0
        if response.status != "completed" or not response.output_text:
            raise ValueError("Response incomplete or refused.")
        return json.loads(response.output_text)

    def continued_choice(self, observation):
        """Continue a previously validated journey for a bounded number of ticks.

        Reconsider on arrival, depletion, low energy, or cadence expiry. Waiting
        is never cached. This changes decision cadence, not movement mechanics.
        """
        previous = self._last_llm_choice.get(observation["agent_id"])
        if previous is None or self.decision_interval == 1:
            return None
        step, decision = previous
        if not 0 < observation["step"] - step < self.decision_interval:
            return None
        action = decision["action"]
        if observation["cargo"] or action not in observation["allowed_actions"]:
            return None
        if action == "RETURN_BASE" and observation["position"] != observation["base_position"]:
            return decision
        if action.startswith("COLLECT_"):
            _, x, y = action.split("_")
            target = [int(x), int(y)]
            distance_home = sum(abs(a - b) for a, b in zip(observation["position"], observation["base_position"]))
            if (target == observation["current_target"] and target in observation["known_resources"]
                    and target != observation["position"] and observation["energy"] > distance_home + 2):
                return decision
        return None

    def wait_for_request_slot(self):
        delay = max(0.0, self._next_request_at - time.monotonic())
        if delay:
            self.pacing_seconds += delay
            time.sleep(delay)

    def close(self):
        if self.client is not None and hasattr(self.client, "close"):
            self.client.close()

    def decide(self, observation):
        started = time.perf_counter()
        error = None
        fallback = baseline_decision(observation)
        continued = self.continued_choice(observation)
        if self.mode == "mock":
            decision, source = fallback, "mock_rules"
        elif len(observation["allowed_actions"]) == 1:
            decision = {"action": observation["allowed_actions"][0], "reason": "Only one available action; no API call needed."}
            source = "forced_action"
            self._last_llm_choice.pop(observation["agent_id"], None)
        elif continued is not None and not self.disabled_reason:
            decision, source = continued, "continued_llm"
            self.continued_decisions += 1
        elif self.disabled_reason or self.api_calls >= self.max_calls:
            decision, source = fallback, "fallback"
            error = self.disabled_reason or "API call budget exhausted"
        else:
            try:
                self.wait_for_request_slot()
                self.api_calls += 1
                decision = validate_decision(self.request_decision(observation), observation)
                source = "llm"
                self.llm_decisions += 1
                self._last_llm_choice[observation["agent_id"]] = (observation["step"], decision)
            except Exception as exc:
                # Never log raw provider errors: they may contain request data.
                from gemini_client import GeminiError
                error = str(exc) if isinstance(exc, GeminiError) else type(exc).__name__
                decision, source = fallback, "fallback"
                # Stop repeatedly calling a failing provider during this run.
                self.disabled_reason = f"Provider disabled after {error}; restart the run after fixing it"
            finally:
                self._next_request_at = time.monotonic() + self.min_request_seconds
        validate_decision(decision, observation)
        if source == "fallback":
            self.fallbacks += 1
            if self.fallbacks == 1:
                print(f"LLM fallback: {error}. Remaining fallback decisions are counted in the report.")
        record = {
            "step": observation["step"], "agent_id": observation["agent_id"],
            "source": source, **decision, "error": error,
            "seconds": round(time.perf_counter() - started, 4),
            "observation": observation,
        }
        self.records.append(record)
        return decision

    def summary(self):
        return {
            "mode": self.mode,
            "model": self.model_name if self.mode != "mock" else "none (offline coded rules)",
            "prompt_hash": self.prompt_hash, "max_api_calls": self.max_calls,
            "api_calls": self.api_calls, "llm_decisions": self.llm_decisions,
            "fallback_decisions": self.fallbacks,
            "continued_llm_decisions": self.continued_decisions,
            "decision_interval": self.decision_interval,
            "min_request_seconds": self.min_request_seconds,
            "pacing_seconds": round(self.pacing_seconds, 3),
            "input_tokens": self.input_tokens, "output_tokens": self.output_tokens,
            "result_label": "mock_rules" if self.mode == "mock" else ("llm_with_fallback" if self.fallbacks else ("llm" if self.llm_decisions else "no_llm_decisions")),
        }

    def save_log(self, path):
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as handle:
            for record in self.records:
                handle.write(json.dumps(record) + "\n")
