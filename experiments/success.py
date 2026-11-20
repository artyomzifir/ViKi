"""Strict terminal acceptance for a physically replayed cube transfer."""

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class StackCriteria:
    cube_side_m: float = 0.04
    xy_tolerance_m: float = 0.015
    height_tolerance_m: float = 0.008
    speed_tolerance_m_s: float = 0.02
    hold_s: float = 0.5


def released_stack_success(*, time_s, moved_xyz_m, target_xyz_m,
                           moved_speed_m_s, support_contact,
                           gripper_contact, gripper_clear,
                           criteria=StackCriteria()):
    """Accept only a supported stack after release and gripper retreat."""
    times = np.asarray(time_s, dtype=float)
    moved = np.asarray(moved_xyz_m, dtype=float)
    target = np.asarray(target_xyz_m, dtype=float)
    speed = np.asarray(moved_speed_m_s, dtype=float)
    support = np.asarray(support_contact, dtype=bool)
    gripping = np.asarray(gripper_contact, dtype=bool)
    clear = np.asarray(gripper_clear, dtype=bool)
    count = len(times)
    if count < 2 or moved.shape != (count, 3) or target.shape != (count, 3):
        raise ValueError("time and cube-position arrays have incompatible shapes")
    if any(values.shape != (count,) for values in (speed, support, gripping, clear)):
        raise ValueError("per-frame state arrays must match time")
    if not (np.isfinite(times).all() and np.isfinite(moved).all()
            and np.isfinite(target).all() and np.isfinite(speed).all()):
        return False
    if not np.all(np.diff(times) > 0):
        raise ValueError("time must be strictly increasing")
    start = max(0, int(np.searchsorted(
        times, times[-1] - criteria.hold_s, side="right") - 1))
    if times[-1] - times[start] < criteria.hold_s:
        return False
    delta = moved[start:] - target[start:]
    return bool(
        np.all(np.linalg.norm(delta[:, :2], axis=1) <= criteria.xy_tolerance_m)
        and np.all(np.abs(delta[:, 2] - criteria.cube_side_m)
                   <= criteria.height_tolerance_m)
        and np.all(speed[start:] <= criteria.speed_tolerance_m_s)
        and np.all(support[start:])
        and not np.any(gripping[start:])
        and np.all(clear[start:])
    )
