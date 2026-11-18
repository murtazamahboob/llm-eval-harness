"""Tests for the models — including the keyword model's real behaviour."""

from evalharness.datasets import ALL_ITEMS, load_items
from evalharness.models import EchoModel, KeywordModel, OracleModel, RandomModel
from evalharness.prompts import TEMPLATES, few_shot_examples_for, render

KEYWORD = KeywordModel()


def _prompt(item, template="zero-shot"):
    return render(TEMPLATES[template], item,
                  few_shot_examples_for(item, ALL_ITEMS) if TEMPLATES[template].few_shot else None)


def test_oracle_is_perfect_by_construction():
    for item in ALL_ITEMS:
        assert OracleModel().generate(_prompt(item), item) == item.expected


def test_keyword_model_scores_high_but_not_perfect():
    exact = 0
    for item in ALL_ITEMS:
        if KEYWORD.generate(_prompt(item), item).strip().lower() == item.expected.strip().lower():
            exact += 1
    # Strong rule baseline: most items, with genuine misses (qa-02, qa-08, qa-11).
    assert 40 <= exact <= 46


def test_keyword_qa_misses_are_the_expected_hard_items():
    qa = {item.item_id: item for item in load_items(["qa"])}
    assert KEYWORD.generate(_prompt(qa["qa-01"]), qa["qa-01"]) == "Mars"
    assert KEYWORD.generate(_prompt(qa["qa-02"]), qa["qa-02"]) == "Honshu"
    assert KEYWORD.generate(_prompt(qa["qa-05"]), qa["qa-05"]) == "1969"
    assert KEYWORD.generate(_prompt(qa["qa-11"]), qa["qa-11"]) == "Two"


def test_keyword_math_is_exact_on_all_word_problems():
    for item in load_items(["math"]):
        assert KEYWORD.generate(_prompt(item), item) == item.expected


def test_keyword_classify_and_extract():
    for item in load_items(["classify", "extract"]):
        assert KEYWORD.generate(_prompt(item), item) == item.expected


def test_keyword_prompt_format_changes_the_answer():
    item = load_items(["qa"])[0]  # qa-01, expected Mars
    assert KEYWORD.generate(_prompt(item, "concise"), item) == "Mars"
    verbose = KEYWORD.generate(_prompt(item, "verbose"), item)
    assert "Mars" in verbose and len(verbose) > len("Mars")
    cot = KEYWORD.generate(_prompt(item, "chain-of-thought"), item)
    assert "Mars" in cot and "step by step" in _prompt(item, "chain-of-thought").lower()


def test_random_model_is_deterministic_and_weak():
    first = [RandomModel(seed=0).generate(_prompt(i), i) for i in ALL_ITEMS]
    second = [RandomModel(seed=0).generate(_prompt(i), i) for i in ALL_ITEMS]
    assert first == second
    exact = sum(1 for text, item in zip(first, ALL_ITEMS)
                if text.strip().lower() == item.expected.strip().lower())
    assert exact <= 12  # a random baseline should not look competent


def test_echo_model_returns_the_prompt():
    item = ALL_ITEMS[0]
    prompt = _prompt(item)
    assert EchoModel().generate(prompt, item) == prompt[:200]
