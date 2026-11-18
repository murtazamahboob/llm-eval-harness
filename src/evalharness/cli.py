"""Command-line interface for the LLM evaluation harness."""

from __future__ import annotations

import argparse
import json
import sys

from . import report as report_mod
from .datasets import ALL_ITEMS, load_items
from .harness import (compare_runs, load_result, paired_bootstrap,
                      run_evaluation, save_result)
from .models import EchoModel, HttpModel, KeywordModel, OracleModel, RandomModel
from .prompts import TEMPLATES

MODELS = {
    "oracle": lambda: OracleModel(),
    "keyword": lambda: KeywordModel(),
    "random": lambda: RandomModel(seed=0),
    "echo": lambda: EchoModel(),
    "http": lambda: HttpModel(),
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="llmeval",
        description="Evaluate LLMs (or rule-based stand-ins) on bundled tasks.")
    sub = parser.add_subparsers(dest="command", required=True)

    run = sub.add_parser("run", help="Run models and print a leaderboard.")
    run.add_argument("--models", nargs="+", default=["oracle", "keyword", "random", "echo"],
                     choices=list(MODELS))
    run.add_argument("--templates", nargs="+", default=["zero-shot"],
                     choices=list(TEMPLATES))
    run.add_argument("--categories", nargs="+", default=None,
                     choices=["qa", "math", "classify", "extract"])
    run.add_argument("--out", default=None, help="Directory for JSON/Markdown/CSV reports.")
    run.add_argument("--save", default=None, help="Save the first run's JSON to this path.")

    board = sub.add_parser("leaderboard", help="Rank saved run JSON files.")
    board.add_argument("files", nargs="+")

    compare = sub.add_parser("compare", help="Regression-check a candidate run against a baseline.")
    compare.add_argument("baseline")
    compare.add_argument("candidate")
    compare.add_argument("--threshold", type=float, default=0.05)
    compare.add_argument("--bootstrap", action="store_true",
                         help="Add a seeded paired-bootstrap significance test.")

    sub.add_parser("prompts", help="List the available prompt templates.")
    sub.add_parser("models", help="List the available models.")
    return parser


def _run_command(args: argparse.Namespace) -> int:
    items = load_items(args.categories)
    results = []
    for model_name in args.models:
        for template_name in args.templates:
            result = run_evaluation(MODELS[model_name](), template_name, items)
            results.append(result)
            print(f"{model_name}/{template_name}: score "
                  f"{report_mod.mean_score(result):.3f} over {len(items)} items")
    print()
    print(report_mod.leaderboard_table(results))
    if args.save and results:
        save_result(results[0], args.save)
        print(f"\nSaved {args.save}")
    if args.out:
        import os
        os.makedirs(args.out, exist_ok=True)
        with open(os.path.join(args.out, "report.md"), "w", encoding="utf-8") as handle:
            handle.write(report_mod.markdown_report(results))
        with open(os.path.join(args.out, "leaderboard.csv"), "w", encoding="utf-8") as handle:
            handle.write(report_mod.results_to_csv(results))
        for result in results:
            save_result(result, os.path.join(args.out, f"{result.model}-{result.template}.json"))
        print(f"Reports written to {args.out}/")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.command == "run":
        return _run_command(args)
    if args.command == "leaderboard":
        results = [load_result(path) for path in args.files]
        print(report_mod.leaderboard_table(results))
        return 0
    if args.command == "compare":
        baseline = load_result(args.baseline)
        candidate = load_result(args.candidate)
        comparison = compare_runs(baseline, candidate, args.threshold)
        print(json.dumps(comparison, indent=2))
        if args.bootstrap:
            print(json.dumps(paired_bootstrap(baseline, candidate), indent=2))
        return 0 if comparison["regression_count"] == 0 else 1
    if args.command == "prompts":
        for name, template in TEMPLATES.items():
            print(f"{name}: {template.instruction}")
        return 0
    if args.command == "models":
        for name in MODELS:
            print(name)
        return 0
    parser.print_help()
    return 1


if __name__ == "__main__":
    sys.exit(main())
