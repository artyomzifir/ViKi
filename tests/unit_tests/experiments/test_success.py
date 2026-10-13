import numpy as np
import pytest

from experiments.success import released_stack_success


def _trace():
    count = 61
    return {
        "time_s": np.linspace(0.0, 1.0, count),
        "moved_xyz_m": np.tile([0.0, 0.0, 0.08], (count, 1)),
        "target_xyz_m": np.tile([0.0, 0.0, 0.04], (count, 1)),
        "moved_speed_m_s": np.zeros(count),
        "support_contact": np.ones(count, dtype=bool),
        "gripper_contact": np.zeros(count, dtype=bool),
        "gripper_clear": np.ones(count, dtype=bool),
    }


def test_released_stack_succeeds():
    assert released_stack_success(**_trace())


def test_released_stack_succeeds_when_samples_do_not_land_on_hold_boundary():
    trace = _trace()
    trace["time_s"] = np.linspace(0.0, 1.0, 501)[:-1]
    for field in ("moved_xyz_m", "target_xyz_m", "moved_speed_m_s",
                  "support_contact", "gripper_contact", "gripper_clear"):
        trace[field] = np.repeat(trace[field][:1], len(trace["time_s"]), axis=0)
    assert released_stack_success(**trace)


@pytest.mark.parametrize("field,value", [
    ("moved_xyz_m", [0.03, 0.0, 0.08]),
    ("moved_xyz_m", [0.0, 0.0, 0.12]),
    ("moved_speed_m_s", 0.04),
    ("support_contact", False),
    ("gripper_contact", True),
    ("gripper_clear", False),
])
def test_hovering_slipping_or_still_gripped_is_not_success(field, value):
    trace = _trace()
    trace[field][-1] = value
    assert not released_stack_success(**trace)


def test_short_or_invalid_time_is_not_success():
    trace = {key: value[:20] for key, value in _trace().items()}
    assert not released_stack_success(**trace)
    trace = _trace()
    trace["time_s"][-1] = trace["time_s"][-2]
    with pytest.raises(ValueError, match="strictly increasing"):
        released_stack_success(**trace)
