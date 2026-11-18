"""Tests for the harness: scoring, comparison and significance."""

import json
import os
import tempfile

from evalharness.datasets import load_items
from evalharness.harness import (compare_runs, load_result, paired_bootstrap,
                                 run_evaluation, save_result)
from evalharness.models import EchoModel, KeywordModel, OracleModel, RandomModel


def test_oracle_scores_100_percent():
    result = run_evaluation(OracleModel(), "zero-shot")
    assert result.mean_exact == 1.0
    assert result.mean_f1 == 1.0
    assert len(result.items) == 48
    assert result.per_category == {"qa": 1.0, "math": 1.0, "classify": 1.0, "extract": 1.0}


def test_leaderboard_ordering_is_oracle_keyword_random_echo():
    results = [run_evaluation(model, "zero-shot")
               for model in (EchoModel(), RandomModel(seed=0), KeywordModel(), OracleModel())]
    from evalharness.report import leaderboard, mean_score
    ranked = leaderboard(results)
    assert [r.model for r in ranked] == ["oracle", "keyword", "random", "echo"]
    assert mean_score(ranked[1]) > 0.85
    assert mean_score(ranked[2]) < 0.35


def test_verbose_prompt_regresses_keyword_exact_match():
    zero = run_evaluation(KeywordModel(), "zero-shot")
    verbose = run_evaluation(KeywordModel(), "verbose")
    comparison = compare_runs(zero, verbose)
    assert comparison["mean_score_delta"] < 0
    assert comparison["regression_count"] > 0
    assert all(r["delta"] < -0.05 for r in comparison["regressions"])


def test_compare_identical_runs_has_no_regressions():
    result = run_evaluation(KeywordModel(), "zero-shot")
    comparison = compare_runs(result, result)
    assert comparison["regression_count"] == 0
    assert comparison["mean_score_delta"] == 0.0


def test_bootstrap_keyword_beats_random_significantly():
    keyword = run_evaluation(KeywordModel(), "zero-shot")
    random_result = run_evaluation(RandomModel(seed=0), "zero-shot")
    stats = paired_bootstrap(random_result, keyword, resamples=200, seed=0)
    assert stats["mean_delta"] > 0.5
    assert stats["ci_low"] > 0
    assert stats["p_value"] == 0.0


def test_bootstrap_is_deterministic_for_same_seed():
    keyword = run_evaluation(KeywordModel(), "zero-shot")
    random_result = run_evaluation(RandomModel(seed=0), "zero-shot")
    first = paired_bootstrap(random_result, keyword, resamples=100, seed=7)
    second = paired_bootstrap(random_result, keyword, resamples=100, seed=7)
    assert first == second


def test_save_and_load_round_trip():
    result = run_evaluation(KeywordModel(), "zero-shot", load_items(["math"]))
    with tempfile.TemporaryDirectory() as tmp:
        path = os.path.join(tmp, "run.json")
        save_result(result, path)
        loaded = load_result(path)
        assert loaded.model == result.model
        assert loaded.mean_exact == result.mean_exact
        assert len(loaded.items) == 12
        assert json.load(open(path))["template"] == "zero-shot"


def test_cost_and_token_accounting():
    result = run_evaluation(KeywordModel(), "zero-shot")
    assert result.total_input_tokens > result.total_output_tokens > 0
    assert result.total_cost_usd > 0
    assert result.mean_latency_ms >= 0
    assert result.p95_latency_ms >= result.mean_latency_ms
