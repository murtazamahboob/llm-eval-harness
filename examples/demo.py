"""Headline demo: every bundled model, zero-shot, plus prompt comparison.

Writes outputs/report.md, outputs/leaderboard.csv and one JSON per run
into outputs/, and prints the leaderboard. Deterministic scores — only
the measured latencies vary between runs.
"""

from __future__ import annotations

import json
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "src"))

from evalharness import report as report_mod
from evalharness.datasets import ALL_ITEMS
from evalharness.harness import compare_runs, paired_bootstrap, run_evaluation, save_result
from evalharness.models import EchoModel, KeywordModel, OracleModel, RandomModel

OUT = os.path.join(os.path.dirname(__file__), "..", "outputs")


def main() -> None:
    os.makedirs(OUT, exist_ok=True)
    models = [OracleModel(), KeywordModel(), RandomModel(seed=0), EchoModel()]
    results = [run_evaluation(model, "zero-shot") for model in models]
    keyword_templates = [run_evaluation(KeywordModel(), name)
                         for name in ("concise", "few-shot", "chain-of-thought", "verbose")]
    all_results = results + keyword_templates

    print(f"Items: {len(ALL_ITEMS)} (12 each of qa, math, classify, extract)\n")
    print(report_mod.leaderboard_table(all_results))
    print()
    for result in results:
        print(f"{result.model}: score {report_mod.mean_score(result):.3f}, "
              f"exact {result.mean_exact:.3f}, F1 {result.mean_f1:.3f}, "
              f"tokens in/out {result.total_input_tokens}/{result.total_output_tokens}, "
              f"cost ${result.total_cost_usd:.6f}")

    keyword_zero = results[1]
    keyword_verbose = keyword_templates[3]
    comparison = compare_runs(keyword_zero, keyword_verbose)
    print(f"\nPrompt effect — keyword zero-shot vs verbose: "
          f"mean delta {comparison['mean_score_delta']:+.3f}, "
          f"{comparison['regression_count']} regressions")
    print("Bootstrap keyword vs random:",
          json.dumps(paired_bootstrap(results[2], keyword_zero)))

    with open(os.path.join(OUT, "report.md"), "w", encoding="utf-8") as handle:
        handle.write(report_mod.markdown_report(all_results))
    with open(os.path.join(OUT, "leaderboard.csv"), "w", encoding="utf-8") as handle:
        handle.write(report_mod.results_to_csv(all_results))
    for result in all_results:
        save_result(result, os.path.join(OUT, f"{result.model}-{result.template}.json"))
    summary = {
        "items": len(ALL_ITEMS),
        "leaderboard": [
            {"model": r.model, "template": r.template,
             "score": round(report_mod.mean_score(r), 4)}
            for r in report_mod.leaderboard(all_results)],
    }
    with open(os.path.join(OUT, "summary.json"), "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2)
    print(f"\nReports written to {OUT}/")


if __name__ == "__main__":
    main()
