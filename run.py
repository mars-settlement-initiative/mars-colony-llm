"""Launch the Folder 3 dashboard or run a reproducible batch experiment."""

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from config import DEFAULT_SEED, MAX_STEPS
from llm_config import LLM_MODE, MAX_API_CALLS
from llm_decision import DecisionEngine
from metrics import simulate
from model import MarsColonyModel


def positive_int(value):
    number = int(value)
    if number < 1:
        raise argparse.ArgumentTypeError("Must be at least 1.")
    return number


def run_batch(max_steps, seed, engine, output):
    model = MarsColonyModel(seed=seed, decision_engine=engine)
    report = {**simulate(model, max_steps, seed), **engine.summary()}
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    (output / "report.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
    engine.save_log(output / "decisions.jsonl")
    (output / "prompt.txt").write_text(engine.prompt, encoding="utf-8")
    model.datacollector.get_model_vars_dataframe().to_csv(output / "timeseries.csv", index_label="step")
    print("\nMARS COLONY — MISSION 2: AI REASONING")
    for key, value in report.items():
        print(f"{key:24} {value}")
    print(f"Results: {output.resolve()}")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--batch", action="store_true")
    parser.add_argument("--mode", choices=["gemini", "mock", "openai"], default=LLM_MODE)
    parser.add_argument("--model", help="provider model (default: Gemini Flash-Lite or OpenAI GPT-4o mini)")
    parser.add_argument("--max-calls", type=int, default=MAX_API_CALLS)
    parser.add_argument("--steps", type=positive_int, default=MAX_STEPS)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--port", type=positive_int, default=8767)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "results" / datetime.now().strftime("%Y%m%d_%H%M%S_%f"))
    args = parser.parse_args()
    try:
        engine = DecisionEngine(args.mode, args.model, args.max_calls)
    except ValueError as exc:
        parser.error(str(exc))
    print("Offline demo: coded rules, no LLM calls." if args.mode == "mock" else f"{args.mode.title()}: {engine.model_name}; up to {args.max_calls} calls per run.")
    if args.mode == "gemini":
        print(f"Use a Free Tier project key. Requests are spaced by {engine.min_request_seconds:g}s; journeys reconsidered within {engine.decision_interval} ticks.")
    if args.batch:
        try:
            run_batch(args.steps, args.seed, engine, args.output)
        finally:
            engine.close()
    else:
        engine.close()
        env = os.environ.copy()
        env.update(MARS_LLM_MODE=args.mode,
                   MARS_MAX_API_CALLS=str(args.max_calls), MARS_RUN_SEED=str(args.seed),
                   MARS_RUN_STEPS=str(args.steps))
        env["GEMINI_MODEL" if args.mode == "gemini" else "OPENAI_MODEL"] = engine.model_name
        app = Path(__file__).with_name("app.py")
        raise SystemExit(subprocess.call([
            sys.executable, "-m", "solara", "run", str(app),
            "--host", "127.0.0.1", "--port", str(args.port), "--open",
        ], cwd=app.parent, env=env))


if __name__ == "__main__":
    main()
