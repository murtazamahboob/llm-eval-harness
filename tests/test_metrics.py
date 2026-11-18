"""Tests for the scoring metrics."""

from evalharness.metrics import (estimate_cost_usd, estimate_tokens,
                                 exact_match, mean, numeric_match,
                                 percentile, token_f1)


def test_exact_match_ignores_case_punctuation_and_articles():
    assert exact_match("Paris", "paris") == 1.0
    assert exact_match("The Paris.", "paris") == 1.0
    assert exact_match("London", "Paris") == 0.0


def test_token_f1_full_and_partial_credit():
    assert token_f1("Paris", "Paris") == 1.0
    assert token_f1("The city is Paris", "Paris") == 0.5  # 2*(1/3)*1 / (4/3)
    assert token_f1("London", "Paris") == 0.0
    assert token_f1("", "Paris") == 0.0


def test_numeric_match_tolerates_format():
    assert numeric_match("42", "42") == 1.0
    assert numeric_match("The answer is 42.0", "42") == 1.0
    assert numeric_match("There are 43 pens", "42") == 0.0
    assert numeric_match("no number here", "42") == 0.0
    assert numeric_match("1,000", "1000") == 1.0


def test_estimate_tokens_is_deterministic_and_monotone():
    assert estimate_tokens("") == 0
    assert estimate_tokens("hi") == 1
    assert estimate_tokens("a" * 8) == 2
    assert estimate_tokens("a" * 9) == 3


def test_estimate_cost_scales_with_tokens():
    assert estimate_cost_usd(0, 0) == 0.0
    cost = estimate_cost_usd(1_000_000, 1_000_000)
    assert cost == 0.50 + 1.50
    assert estimate_cost_usd(2_000_000, 0) == 2 * estimate_cost_usd(1_000_000, 0)


def test_mean_and_percentile():
    assert mean([]) == 0.0
    assert mean([1.0, 2.0, 3.0]) == 2.0
    assert percentile([1, 2, 3, 4], 50) == 2
    assert percentile([1, 2, 3, 4], 95) == 4
    assert percentile([], 95) == 0.0
