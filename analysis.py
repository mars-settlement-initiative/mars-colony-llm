"""Compare the actual code in Folders 1, 2 and 3 on the same benchmark seeds."""

import argparse
import csv
import json
import statistics
import subprocess
import sys
from datetime import datetime
from pathlib import Path

from config import BENCHMARK_SEEDS, MAX_STEPS
from llm_config import LLM_MODE, MAX_API_CALLS
from llm_decision import DecisionEngine
from run import positive_int

HERE = Path(__file__).resolve().parent
METRICS = ["steps_survived", "resources_collected", "distance_travelled",
           "failed_robots", "idle_collector_steps", "duplicate_target_steps"]


def worker(folder, *options):
    result = subprocess.run(
        [sys.executable, str(HERE / "benchmark_worker.py"), str(folder), *options],
        check=True, capture_output=True, text=True,
    )
    return json.loads(result.stdout)


def compare(folders, seeds, steps, mode, model_name, max_calls, output):
    # Catch mismatched experiment settings before spending any API calls.
    settings = [worker(folder, "--describe") for folder in folders]
    if any(setting != settings[0] for setting in settings[1:]):
        raise ValueError("World settings differ between folders. Restore matching settings before comparing.")
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    rows = []
    for folder in folders:
        for seed in seeds:
            print(f"Running {folder.name}, seed {seed} ...", flush=True)
            row = worker(folder, "--seed", str(seed), "--steps", str(steps),
                         "--mode", mode, "--model", model_name,
                         "--max-calls", str(max_calls), "--output", str(output))
            rows.append(row)
            # Save after every run, including partial progress if a later run fails.
            (output / "benchmark.json").write_text(json.dumps(rows, indent=2), encoding="utf-8")
    fields = ["folder", "result_label", "mode", "model", "seed", "max_steps", *METRICS,
              "api_calls", "llm_decisions", "fallback_decisions", "input_tokens", "output_tokens"]
    fields += ["continued_llm_decisions", "decision_interval", "min_request_seconds", "pacing_seconds"]
    with (output / "benchmark.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)
    print("\nMEAN +/- SAMPLE STANDARD DEVIATION")
    summaries = []
    for folder in folders:
        group = [row for row in rows if row["folder"] == folder.name]
        print(f"\n{folder.name}: {', '.join(sorted({r['result_label'] for r in group}))}")
        for key in METRICS:
            values = [row[key] for row in group]
            mean, sd = statistics.mean(values), statistics.stdev(values) if len(values) > 1 else 0
            print(f"  {key:24} {mean:.2f} +/- {sd:.2f}")
            summaries.append({"folder": folder.name, "metric": key, "mean": mean, "sample_sd": sd})
    # Paired differences are more useful than comparing unpaired overall means.
    paired = []
    for seed in seeds:
        baseline = next(r for r in rows if r["folder"] == folders[0].name and r["seed"] == seed)
        for folder in folders[1:]:
            row = next(r for r in rows if r["folder"] == folder.name and r["seed"] == seed)
            paired.append({"folder": folder.name, "seed": seed,
                           **{key: row[key] - baseline[key] for key in METRICS}})
    (output / "summary.json").write_text(json.dumps(summaries, indent=2), encoding="utf-8")
    (output / "paired_differences.json").write_text(json.dumps(paired, indent=2), encoding="utf-8")
    print(f"\nResults: {output.resolve()}")
    return rows


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["gemini", "mock", "openai"], default=LLM_MODE)
    parser.add_argument("--model", help="override the provider's default model")
    parser.add_argument("--max-calls", type=int, default=MAX_API_CALLS)
    parser.add_argument("--seeds", nargs="+", type=int, default=BENCHMARK_SEEDS)
    parser.add_argument("--steps", type=positive_int, default=MAX_STEPS)
    parser.add_argument("--output", type=Path, default=HERE / "results" / ("benchmark_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")))
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Use distinct benchmark seeds.")
    try:
        engine = DecisionEngine(args.mode, args.model, args.max_calls)  # Preflight; no API call.
        args.model = engine.model_name
        engine.close()
        if args.mode != "mock":
            print(f"Total call ceiling: {args.max_calls * len(args.seeds)} ({args.max_calls} per seed).")
            if args.mode == "gemini":
                print("Gemini requires a Free Tier project key. Quotas are shared across keys on the same project.")
        else:
            print("OFFLINE COMPARISON: Folder 3 uses coded mock rules, not an LLM.")
        compare([HERE.parent / "mars_colony", HERE.parent / "mars_colony_2", HERE],
                args.seeds, args.steps, args.mode, args.model, args.max_calls, args.output)
    except (ValueError, subprocess.CalledProcessError) as exc:
        parser.error(str(exc))


if __name__ == "__main__":
    main()
