"""Only held-out, physics-successful policy scenes enter the GUI playlist."""

from experiments.learned_viewer import selected_test_rows


def test_selected_test_rows_prioritize_improvement_over_fixed_template():
    report = {"layouts": [
        {"layout_id": "train", "split": "train", "red_offset_xy_m": [0.05, 0],
         "green_offset_xy_m": [0, 0], "learned": {"released_stack_success": True},
         "fixed_template": {"released_stack_success": False}},
        {"layout_id": "failed", "split": "test", "red_offset_xy_m": [0.05, 0],
         "green_offset_xy_m": [0, 0], "learned": {"released_stack_success": False},
         "fixed_template": {"released_stack_success": False}},
        {"layout_id": "improved", "split": "test", "red_offset_xy_m": [0.02, 0],
         "green_offset_xy_m": [0, 0], "learned": {"released_stack_success": True},
         "fixed_template": {"released_stack_success": False}},
        {"layout_id": "both", "split": "test", "red_offset_xy_m": [0.04, 0],
         "green_offset_xy_m": [0, 0], "learned": {"released_stack_success": True},
         "fixed_template": {"released_stack_success": True}},
    ]}
    assert [row["layout_id"] for row in selected_test_rows(report, 2)] == [
        "improved", "both"]
