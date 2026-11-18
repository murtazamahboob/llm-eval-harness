"""Tests for the bundled datasets."""

from evalharness.datasets import ALL_ITEMS, CATEGORIES, load_items


def test_48_items_four_categories_of_12():
    assert len(ALL_ITEMS) == 48
    assert sorted(CATEGORIES) == ["classify", "extract", "math", "qa"]
    for category in CATEGORIES:
        assert len([i for i in ALL_ITEMS if i.category == category]) == 12


def test_item_ids_unique_and_expected_nonempty():
    ids = [item.item_id for item in ALL_ITEMS]
    assert len(ids) == len(set(ids))
    for item in ALL_ITEMS:
        assert item.expected.strip()
        assert item.question.strip()


def test_load_items_filters_by_category():
    assert len(load_items(["math"])) == 12
    assert len(load_items(["qa", "extract"])) == 24
    assert len(load_items()) == 48
    assert all(item.category == "math" for item in load_items(["math"]))
