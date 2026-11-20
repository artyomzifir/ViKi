"""Before/after viewer only shows genuine measured improvements."""

from experiments.quality_viewer import improved_pairs


def test_improved_pairs_prioritizes_largest_gain():
    report = {"episodes": [
        {"candidate": 0, "best_variant": "a", "variants": [
            {"variant": "baseline", "quality_penalty": 6.0},
            {"variant": "a", "quality_penalty": 5.0}]},
        {"candidate": 1, "best_variant": "b", "variants": [
            {"variant": "baseline", "quality_penalty": 7.0},
            {"variant": "b", "quality_penalty": 4.0}]},
        {"candidate": 2, "best_variant": "baseline", "variants": [
            {"variant": "baseline", "quality_penalty": 3.0}]},
    ]}
    assert [episode["candidate"] for episode, _, _ in improved_pairs(report, 2)] == [1, 0]
