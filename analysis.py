"""Benchmark this repository and export spreadsheet-ready matched-seed tables."""

import argparse
import csv
import json
import statistics
import sys
from datetime import datetime
from pathlib import Path

from config import BENCHMARK_SEEDS, MAX_STEPS
from llm_config import GEMINI_MIN_REQUEST_SECONDS, LLM_MODE, MAX_API_CALLS
from llm_decision import DecisionEngine
from metrics import simulate
from model import MarsColonyModel
from run import positive_int

HERE = Path(__file__).resolve().parent
METRICS = [
    "steps_survived",
    "resources_collected",
    "distance_travelled",
    "failed_robots",
    "idle_collector_steps",
    "duplicate_target_steps",
]
FIELDS = [
    "implementation",
    "seed",
    "max_steps",
    *METRICS,
    "base_resources",
    "result_label",
    "mode",
    "model",
    "prompt_hash",
    "max_api_calls",
    "api_calls",
    "llm_decisions",
    "continued_llm_decisions",
    "fallback_decisions",
    "decision_interval",
    "min_request_seconds",
    "input_tokens",
    "output_tokens",
]


def write_csv(path, fieldnames, rows):
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def print_table(rows):
    headings = ["seed", *METRICS]
    widths = {
        heading: max(len(heading), *(len(str(row[heading])) for row in rows))
        for heading in headings
    }
    print()
    print(" | ".join(heading.ljust(widths[heading]) for heading in headings))
    print("-+-".join("-" * widths[heading] for heading in headings))
    for row in rows:
        print(" | ".join(str(row[heading]).rjust(widths[heading]) for heading in headings))


def run_experiment(seed, max_steps, implementation, mode, model_name, max_calls,
                   request_seconds, output):
    def report_request(number, maximum):
        print(
            f"  seed {seed}: API request {number}/{maximum}",
            file=sys.stderr,
            flush=True,
        )

    engine = DecisionEngine(
        mode,
        model_name,
        max_calls,
        min_request_seconds=request_seconds,
        progress_callback=report_request,
    )
    try:
        model = MarsColonyModel(seed=seed, decision_engine=engine)
        row = {
            "implementation": implementation,
            **simulate(model, max_steps, seed),
            **engine.summary(),
        }
        engine.save_log(output / f"seed_{seed}_decisions.jsonl")
        engine.save_csv(output / f"seed_{seed}_decisions.csv")
        return row
    finally:
        engine.close()


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--mode", choices=["gemini", "mock", "openai"], default=LLM_MODE)
    parser.add_argument("--model", help="override the provider's default model")
    parser.add_argument("--max-calls", type=int, default=MAX_API_CALLS)
    parser.add_argument(
        "--request-seconds", type=float, default=GEMINI_MIN_REQUEST_SECONDS,
        help="minimum wall-clock seconds between Gemini requests",
    )
    parser.add_argument("--seeds", nargs="+", type=int, default=BENCHMARK_SEEDS)
    parser.add_argument("--steps", type=positive_int, default=MAX_STEPS)
    parser.add_argument("--label", default=HERE.name)
    parser.add_argument(
        "--output",
        type=Path,
        default=HERE / "results" / (
            "benchmark_" + datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        ),
    )
    args = parser.parse_args()
    if len(set(args.seeds)) != len(args.seeds):
        parser.error("Use distinct benchmark seeds")
    if args.request_seconds < 0:
        parser.error("--request-seconds must be nonnegative")

    try:
        preflight = DecisionEngine(
            args.mode, args.model, args.max_calls,
            min_request_seconds=args.request_seconds,
        )
        args.model = preflight.model_name
        request_spacing = preflight.min_request_seconds
        prompt = preflight.prompt
        preflight.close()
    except ValueError as exc:
        parser.error(str(exc))

    args.output.mkdir(parents=True, exist_ok=True)
    (args.output / "prompt.txt").write_text(prompt, encoding="utf-8")

    if args.mode != "mock":
        print(
            f"Total call ceiling: {args.max_calls * len(args.seeds)} "
            f"({args.max_calls} per seed)."
        )
        pacing_per_seed = max(0, args.max_calls - 1) * request_spacing
        if pacing_per_seed:
            print(
                f"Configured spacing can add {pacing_per_seed / 60:.1f} minutes "
                f"per seed at the call ceiling."
            )
    else:
        print("OFFLINE COMPARISON: mock mode uses coded rules, not an LLM.")

    rows = []
    for seed in args.seeds:
        print(f"Running {args.label}, seed {seed} ...", flush=True)
        rows.append(
            run_experiment(
                seed,
                args.steps,
                args.label,
                args.mode,
                args.model,
                args.max_calls,
                args.request_seconds,
                args.output,
            )
        )
        write_csv(args.output / "benchmark.csv", FIELDS, rows)
        (args.output / "benchmark.json").write_text(
            json.dumps(rows, indent=2), encoding="utf-8"
        )

    summary = []
    for metric in METRICS:
        values = [row[metric] for row in rows]
        summary.append(
            {
                "implementation": args.label,
                "metric": metric,
                "mean": statistics.mean(values),
                "sample_sd": statistics.stdev(values) if len(values) > 1 else 0,
            }
        )
    write_csv(
        args.output / "summary.csv",
        ["implementation", "metric", "mean", "sample_sd"],
        summary,
    )

    print_table(rows)
    print()
    print("Mean +/- sample standard deviation")
    for row in summary:
        print(f"{row['metric']:24} {row['mean']:.2f} +/- {row['sample_sd']:.2f}")
    print(f"\nSpreadsheet files: {args.output.resolve()}")


if __name__ == "__main__":
    main()
