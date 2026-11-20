"""Quality selection keeps successful plans and falls back when needed."""

from experiments.quality_batch import best_variant


def test_best_variant_ignores_invalid_and_chooses_lowest_penalty():
    rows = [{"variant": "failed", "quality_penalty": None},
            {"variant": "baseline", "quality_penalty": 6.0},
            {"variant": "improved", "quality_penalty": 4.0}]
    assert best_variant(rows)["variant"] == "improved"
    assert best_variant(rows[:1]) is None
