"""The evaluation harness: run models over tasks and score them.

Deterministic where it matters — scores, token counts and costs depend
only on the prompt and the answer. Latency is measured live (wall clock),
so it varies run to run; nothing in the headline results depends on it.
"""

from __future__ import annotations

import json
import random
import time
from dataclasses import asdict, dataclass

from . import metrics
from .datasets import ALL_ITEMS, Item
from .prompts import TEMPLATES, few_shot_examples_for, render


@dataclass
class ScoredItem:
    item_id: str
    category: str
    expected: str
    prediction: str
    exact_match: float
    token_f1: float
    numeric_match: float | None
    latency_ms: float
    input_tokens: int
    output_tokens: int

    def score(self) -> float:
        """The item's headline score: numeric match for maths, else exact."""
        if self.numeric_match is not None:
            return self.numeric_match
        return self.exact_match


@dataclass
class RunResult:
    model: str
    template: str
    items: list[ScoredItem]
    mean_exact: float
    mean_f1: float
    mean_latency_ms: float
    p95_latency_ms: float
    total_input_tokens: int
    total_output_tokens: int
    total_cost_usd: float
    per_category: dict[str, float]

    def to_dict(self) -> dict:
        data = asdict(self)
        return data

    @classmethod
    def from_dict(cls, data: dict) -> "RunResult":
        items = [ScoredItem(**item) for item in data["items"]]
        return cls(model=data["model"], template=data["template"], items=items,
                   mean_exact=data["mean_exact"], mean_f1=data["mean_f1"],
                   mean_latency_ms=data["mean_latency_ms"],
                   p95_latency_ms=data["p95_latency_ms"],
                   total_input_tokens=data["total_input_tokens"],
                   total_output_tokens=data["total_output_tokens"],
                   total_cost_usd=data["total_cost_usd"],
                   per_category=data["per_category"])


def score_item(prediction: str, item: Item) -> tuple[float, float, float | None]:
    """Return (exact_match, token_f1, numeric_match-or-None) for one answer."""
    exact = metrics.exact_match(prediction, item.expected)
    f1 = metrics.token_f1(prediction, item.expected)
    numeric = metrics.numeric_match(prediction, item.expected) if item.category == "math" else None
    return exact, f1, numeric


def run_evaluation(model, template_name: str = "zero-shot",
                   items: list[Item] | None = None,
                   input_price_per_million: float = 0.50,
                   output_price_per_million: float = 1.50) -> RunResult:
    """Run one model under one template over the given items."""
    items = list(items) if items is not None else list(ALL_ITEMS)
    template = TEMPLATES[template_name]
    scored: list[ScoredItem] = []
    for item in items:
        examples = few_shot_examples_for(item, items) if template.few_shot else None
        prompt = render(template, item, examples)
        started = time.perf_counter()
        prediction = model.generate(prompt, item)
        latency_ms = (time.perf_counter() - started) * 1000
        exact, f1, numeric = score_item(prediction, item)
        scored.append(ScoredItem(
            item_id=item.item_id, category=item.category,
            expected=item.expected, prediction=prediction,
            exact_match=exact, token_f1=f1, numeric_match=numeric,
            latency_ms=latency_ms,
            input_tokens=metrics.estimate_tokens(prompt),
            output_tokens=metrics.estimate_tokens(prediction)))
    total_in = sum(s.input_tokens for s in scored)
    total_out = sum(s.output_tokens for s in scored)
    per_category: dict[str, float] = {}
    for category in sorted({s.category for s in scored}):
        per_category[category] = metrics.mean([s.score() for s in scored if s.category == category])
    return RunResult(
        model=getattr(model, "name", type(model).__name__),
        template=template_name, items=scored,
        mean_exact=metrics.mean([s.exact_match for s in scored]),
        mean_f1=metrics.mean([s.token_f1 for s in scored]),
        mean_latency_ms=metrics.mean([s.latency_ms for s in scored]),
        p95_latency_ms=metrics.percentile([s.latency_ms for s in scored], 95),
        total_input_tokens=total_in, total_output_tokens=total_out,
        total_cost_usd=metrics.estimate_cost_usd(
            total_in, total_out, input_price_per_million, output_price_per_million),
        per_category=per_category)


def compare_runs(baseline: RunResult, candidate: RunResult,
                 threshold: float = 0.05) -> dict:
    """Regression report: where did the candidate get worse than baseline?

    Returns the overall and per-category deltas plus every item that got
    worse by more than ``threshold`` — the list a developer actually acts on.
    """
    regressions = [
        {"item_id": cand.item_id, "category": cand.category,
         "baseline_score": base.score(), "candidate_score": cand.score(),
         "delta": cand.score() - base.score(),
         "baseline_prediction": base.prediction,
         "candidate_prediction": cand.prediction}
        for base, cand in zip(baseline.items, candidate.items)
        if cand.score() - base.score() < -threshold
    ]
    category_delta = {
        category: candidate.per_category.get(category, 0.0) - baseline.per_category.get(category, 0.0)
        for category in sorted(set(baseline.per_category) | set(candidate.per_category))
    }
    return {
        "baseline": f"{baseline.model}/{baseline.template}",
        "candidate": f"{candidate.model}/{candidate.template}",
        "mean_score_delta": metrics.mean([c.score() for c in candidate.items])
        - metrics.mean([b.score() for b in baseline.items]),
        "category_delta": category_delta,
        "regressions": regressions,
        "regression_count": len(regressions),
    }


def paired_bootstrap(baseline: RunResult, candidate: RunResult,
                     resamples: int = 1000, seed: int = 0) -> dict:
    """Seeded paired bootstrap on per-item scores.

    Resamples items with replacement and re-computes the mean score
    difference each time. Reports the 95% confidence interval and the
    fraction of resamples where the candidate did not beat the baseline —
    a rough p-value for 'is this model actually better?'.
    """
    rng = random.Random(seed)
    diffs = [c.score() - b.score() for b, c in zip(baseline.items, candidate.items)]
    if not diffs:
        return {"mean_delta": 0.0, "ci_low": 0.0, "ci_high": 0.0, "p_value": 1.0}
    estimates = []
    for _ in range(resamples):
        sample = [rng.choice(diffs) for _ in diffs]
        estimates.append(metrics.mean(sample))
    estimates.sort()
    low = estimates[int(0.025 * len(estimates))]
    high = estimates[int(0.975 * len(estimates)) - 1]
    p_value = sum(1 for estimate in estimates if estimate <= 0) / len(estimates)
    return {"mean_delta": metrics.mean(diffs), "ci_low": low,
            "ci_high": high, "p_value": p_value}


def save_result(result: RunResult, path: str) -> None:
    with open(path, "w", encoding="utf-8") as handle:
        json.dump(result.to_dict(), handle, indent=2)


def load_result(path: str) -> RunResult:
    with open(path, encoding="utf-8") as handle:
        return RunResult.from_dict(json.load(handle))
