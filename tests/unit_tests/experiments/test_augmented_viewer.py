"""The GUI playlist selects successful scenes with spread-out layouts."""

from experiments.augmented_viewer import diverse_successes


def test_diverse_successes_skips_failed_scenes_and_spreads_layouts():
    rows = [
        {"candidate": 0, "released_stack_success": True,
         "red_offset_xy_m": [0.001, 0], "green_offset_xy_m": [0, 0]},
        {"candidate": 1, "released_stack_success": False,
         "red_offset_xy_m": [0.05, 0], "green_offset_xy_m": [0, 0]},
        {"candidate": 2, "released_stack_success": True,
         "red_offset_xy_m": [-0.02, 0], "green_offset_xy_m": [0, 0]},
        {"candidate": 3, "released_stack_success": True,
         "red_offset_xy_m": [0.02, 0], "green_offset_xy_m": [0, 0]},
    ]
    assert [row["candidate"] for row in diverse_successes(rows, 2)] == [2, 3]
    assert diverse_successes(rows, 0) == []
