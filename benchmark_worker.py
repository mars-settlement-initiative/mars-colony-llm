"""Run one folder in a fresh process so its imports cannot mix with another's."""

import argparse
import contextlib
import hashlib
import importlib
import io
import json
import sys
from pathlib import Path

from metrics import simulate

WORLD_SETTINGS = [
    "GRID_WIDTH", "GRID_HEIGHT", "NUMBER_EXPLORERS", "NUMBER_COLLECTORS",
    "NUMBER_RESOURCES", "RESOURCE_AMOUNT", "INITIAL_BASE_RESOURCES",
    "CONSUMPTION_RATE", "ROBOT_INITIAL_ENERGY", "COLLECTOR_CAPACITY",
]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("folder", type=Path)
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--steps", type=int, default=500)
    parser.add_argument("--mode", default="mock")
    parser.add_argument("--model")
    parser.add_argument("--max-calls", type=int, default=20)
    parser.add_argument("--output", type=Path)
    parser.add_argument("--describe", action="store_true")
    args = parser.parse_args()
    folder = args.folder.resolve()
    sys.path.insert(0, str(folder))
    config = importlib.import_module("config")
    settings = {key: getattr(config, key) for key in WORLD_SETTINGS}
    if args.describe:
        print(json.dumps(settings))
        return
    engine = None
    with contextlib.redirect_stdout(io.StringIO()):
        model_type = importlib.import_module("model").MarsColonyModel
        if folder == Path(__file__).parent.resolve():
            from llm_decision import DecisionEngine
            engine = DecisionEngine(args.mode, args.model, args.max_calls)
        kwargs = {"seed": args.seed}
        if engine is not None:
            kwargs["decision_engine"] = engine
        model = model_type(**kwargs)
        report = simulate(model, args.steps, args.seed)
    report.update(folder=folder.name, world_settings=settings)
    report["source_hashes"] = {
        name: hashlib.sha256((folder / name).read_bytes()).hexdigest()
        for name in ("agents.py", "model.py", "config.py")
    }
    if engine is not None:
        report.update(engine.summary())
        if args.output:
            engine.save_log(args.output / f"seed_{args.seed}_decisions.jsonl")
            (args.output / "prompt.txt").write_text(engine.prompt, encoding="utf-8")
        engine.close()
    else:
        report.update(result_label="coded_rules", api_calls=0, llm_decisions=0,
                      fallback_decisions=0, input_tokens=0, output_tokens=0)
    print(json.dumps(report))


if __name__ == "__main__":
    main()
