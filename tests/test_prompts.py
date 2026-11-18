"""Tests for prompt templates."""

from evalharness.datasets import ALL_ITEMS, load_items
from evalharness.prompts import TEMPLATES, few_shot_examples_for, render


def test_all_templates_render_the_question_and_answer_marker():
    item = ALL_ITEMS[0]
    for name, template in TEMPLATES.items():
        prompt = render(template, item, few_shot_examples_for(item, ALL_ITEMS))
        assert item.question in prompt, name
        assert prompt.rstrip().endswith("Answer:"), name


def test_few_shot_includes_examples_before_the_item():
    item = load_items(["math"])[0]
    examples = few_shot_examples_for(item, ALL_ITEMS)
    assert len(examples) == 2
    assert all(e.category == "math" and e.item_id != item.item_id for e in examples)
    prompt = render(TEMPLATES["few-shot"], item, examples)
    assert prompt.count("Question:") == 3  # 2 examples + the item


def test_few_shot_examples_are_deterministic():
    item = ALL_ITEMS[0]
    assert few_shot_examples_for(item, ALL_ITEMS) == few_shot_examples_for(item, ALL_ITEMS)


def test_template_names_cover_the_five_styles():
    assert set(TEMPLATES) == {"zero-shot", "concise", "few-shot", "chain-of-thought", "verbose"}
