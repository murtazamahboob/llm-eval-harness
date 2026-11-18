"""Scoring metrics for the LLM evaluation harness.

Everything here is deterministic and stdlib-only. Scores are computed from
the model's raw text answer and the item's ground truth, so the same run
always produces the same numbers.
"""

from __future__ import annotations

import math
import re

_WORD_RE = re.compile(r"[a-z0-9]+")
_NUMBER_RE = re.compile(r"-?\d+(?:\.\d+)?")
_ARTICLES = {"a", "an", "the"}


def normalize(text: str) -> str:
    """Lowercase, strip punctuation and articles, collapse whitespace."""
    words = _WORD_RE.findall(text.lower())
    words = [w for w in words if w not in _ARTICLES]
    return " ".join(words)


def exact_match(prediction: str, expected: str) -> float:
    """1.0 when the normalised answers are identical, else 0.0."""
    return 1.0 if normalize(prediction) == normalize(expected) else 0.0


def token_f1(prediction: str, expected: str) -> float:
    """Token-level F1 between prediction and expected answer.

    A prediction that contains the right answer plus extra words still gets
    partial credit, which is what makes prompt-format effects visible:
    a verbose answer scores lower on exact match but keeps most of its F1.
    """
    pred_tokens = normalize(prediction).split()
    exp_tokens = normalize(expected).split()
    if not pred_tokens or not exp_tokens:
        return 0.0
    pred_counts: dict[str, int] = {}
    for token in pred_tokens:
        pred_counts[token] = pred_counts.get(token, 0) + 1
    overlap = 0
    for token in exp_tokens:
        if pred_counts.get(token, 0) > 0:
            overlap += 1
            pred_counts[token] -= 1
    if overlap == 0:
        return 0.0
    precision = overlap / len(pred_tokens)
    recall = overlap / len(exp_tokens)
    return 2 * precision * recall / (precision + recall)


def numeric_match(prediction: str, expected: str, rel_tol: float = 1e-6) -> float:
    """1.0 when the first number in the prediction equals the expected one.

    Numbers are compared with a relative tolerance so ``42`` and ``42.0``
    count as the same answer. Returns 0.0 when the prediction has no number.
    """
    pred_numbers = _NUMBER_RE.findall(prediction.replace(",", ""))
    exp_numbers = _NUMBER_RE.findall(expected.replace(",", ""))
    if not pred_numbers or not exp_numbers:
        return 0.0
    try:
        pred_value = float(pred_numbers[0])
        exp_value = float(exp_numbers[0])
    except ValueError:
        return 0.0
    if exp_value == 0:
        return 1.0 if pred_value == 0 else 0.0
    return 1.0 if abs(pred_value - exp_value) <= rel_tol * abs(exp_value) else 0.0


def estimate_tokens(text: str) -> int:
    """Deterministic token estimate: roughly one token per four characters."""
    if not text:
        return 0
    return max(1, math.ceil(len(text) / 4))


def estimate_cost_usd(input_tokens: int, output_tokens: int,
                      input_price_per_million: float = 0.50,
                      output_price_per_million: float = 1.50) -> float:
    """Estimated spend in USD for a token count at the given price card."""
    return (input_tokens * input_price_per_million
            + output_tokens * output_price_per_million) / 1_000_000


def mean(values: list[float]) -> float:
    return sum(values) / len(values) if values else 0.0


def percentile(values: list[float], pct: float) -> float:
    """Nearest-rank percentile (pct in 0..100)."""
    if not values:
        return 0.0
    ordered = sorted(values)
    rank = max(1, math.ceil(pct / 100 * len(ordered)))
    return ordered[min(rank, len(ordered)) - 1]
