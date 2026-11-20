"""Layout-conditioned imitation contracts without importing torch or MuJoCo."""

import numpy as np
import pytest

from experiments.layout_bc import phase_features, split_layouts, summarize_independent


def test_split_layouts_stratifies_each_range_without_overlap():
    layouts = [{"layout_id": f"{width}mm_{index:03d}", "range_mm": width}
               for width in (20, 50) for index in range(20)]
    first = split_layouts(layouts)
    second = split_layouts(list(reversed(layouts)))
    assert first == second
    assert {name for group in first.values() for name in group} == {
        row["layout_id"] for row in layouts}
    assert [len(first[key]) for key in ("train", "validation", "test")] == [28, 4, 8]
    assert all(len([name for name in first["test"] if name.startswith(f"{width}mm_")]) == 4
               for width in (20, 50))


def test_phase_features_clamp_beyond_demonstration():
    result = phase_features(np.array([-1.0, 0.0, 2.5, 5.0, 6.0]), 5.0)
    assert result.shape == (5, 17)
    np.testing.assert_allclose(result[0], result[1])
    np.testing.assert_allclose(result[-1], result[-2])
    assert result[2, 0] == pytest.approx(0.5)


def test_independent_summary_keeps_all_and_teacher_accepted_rates_separate():
    rows = [
        {"teacher_accepted": True,
         "learned": {"released_stack_success": True},
         "fixed_template": {"released_stack_success": False}},
        {"teacher_accepted": False,
         "learned": {"released_stack_success": True},
         "fixed_template": {"released_stack_success": None}},
    ]
    assert summarize_independent(rows) == {
        "all_layouts": {"learned": 2, "fixed_template": 0},
        "teacher_accepted_layouts": {"learned": 1, "fixed_template": 0},
        "all_count": 2,
        "teacher_accepted_count": 1,
    }
