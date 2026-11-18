"""Leaderboards and reports: turn run results into readable output."""

from __future__ import annotations

import csv
import io

from .harness import RunResult


def leaderboard(results: list[RunResult]) -> list[RunResult]:
    """Rank runs: highest mean score first, ties broken by F1, then tokens."""
    def key(result: RunResult) -> tuple[float, float, int]:
        mean_score = sum(item.score() for item in result.items) / max(1, len(result.items))
        return (mean_score, result.mean_f1, -result.total_input_tokens - result.total_output_tokens)
    return sorted(results, key=key, reverse=True)


def mean_score(result: RunResult) -> float:
    if not result.items:
        return 0.0
    return sum(item.score() for item in result.items) / len(result.items)


def leaderboard_table(results: list[RunResult]) -> str:
    """Plain-text leaderboard table, readable in a terminal or Markdown."""
    ranked = leaderboard(results)
    lines = ["| Rank | Model / Template | Score | Exact | F1 | Tokens | Cost (USD) |",
             "|-----:|------------------|------:|------:|---:|-------:|-----------:|"]
    for rank, result in enumerate(ranked, start=1):
        tokens = result.total_input_tokens + result.total_output_tokens
        lines.append(
            f"| {rank} | {result.model} / {result.template} | {mean_score(result):.3f} "
            f"| {result.mean_exact:.3f} | {result.mean_f1:.3f} | {tokens} "
            f"| ${result.total_cost_usd:.6f} |")
    return "\n".join(lines)


def markdown_report(results: list[RunResult]) -> str:
    """A full Markdown report: leaderboard, per-category table, worst items."""
    ranked = leaderboard(results)
    lines = ["# LLM Evaluation Report", "",
             "## Leaderboard", "", leaderboard_table(results), "",
             "## Per-category scores", "",
             "| Model / Template | " + " | ".join(["qa", "math", "classify", "extract"]) + " |",
             "|---|---:|---:|---:|---:|"]
    for result in ranked:
        cells = " | ".join(f"{result.per_category.get(cat, 0.0):.3f}"
                           for cat in ["qa", "math", "classify", "extract"])
        lines.append(f"| {result.model} / {result.template} | {cells} |")
    lines += ["", "## Weakest items (best model)", ""]
    best = ranked[0]
    weak = sorted(best.items, key=lambda item: (item.score(), item.token_f1))[:5]
    for item in weak:
        lines.append(
            f"- `{item.item_id}` ({item.category}): expected {item.expected!r}, "
            f"got {item.prediction!r} — score {item.score():.2f}, F1 {item.token_f1:.2f}")
    return "\n".join(lines) + "\n"


def results_to_csv(results: list[RunResult]) -> str:
    """One row per model/template: the machine-readable leaderboard."""
    buffer = io.StringIO()
    writer = csv.writer(buffer)
    writer.writerow(["model", "template", "score", "exact_match", "token_f1",
                     "mean_latency_ms", "p95_latency_ms", "input_tokens",
                     "output_tokens", "cost_usd",
                     "qa", "math", "classify", "extract"])
    for result in leaderboard(results):
        writer.writerow([
            result.model, result.template, f"{mean_score(result):.4f}",
            f"{result.mean_exact:.4f}", f"{result.mean_f1:.4f}",
            f"{result.mean_latency_ms:.2f}", f"{result.p95_latency_ms:.2f}",
            result.total_input_tokens, result.total_output_tokens,
            f"{result.total_cost_usd:.6f}"] + [
                f"{result.per_category.get(cat, 0.0):.4f}"
                for cat in ["qa", "math", "classify", "extract"]])
    return buffer.getvalue()
