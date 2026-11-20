"""
viki.replay.screen
------------------
Numeric and joint-limit checks for an offline plan copy.

An accepted input receives ``dry-run``. Collision and task-level evaluation
belong to the retarget solver and separate simulator experiments.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Verdict:
    verdict: str  # reject | dry-run
    cause: str  # "" | joint_limit


def _joint_limits(robot_description: str):
    """Physical arm-angle limits in plan joint order, or unavailable."""
    try:
        from viki.retarget.robots import normalize_robot

        robot = normalize_robot(robot_description)
        if robot.position_limits is not None:
            limits = np.asarray(robot.position_limits, dtype=np.float64)
            return limits[:, 0], limits[:, 1]
        from viki.retarget.run import _load_robot_description

        model = _load_robot_description(robot.description).model
        lower = []
        upper = []
        for name in robot.joint_names:
            joint_id = int(model.getJointId(name))
            if int(model.nqs[joint_id]) != 1:
                lower.append(-np.inf)
                upper.append(np.inf)
                continue
            q_index = int(model.idx_qs[joint_id])
            lower.append(float(model.lowerPositionLimit[q_index]))
            upper.append(float(model.upperPositionLimit[q_index]))
        return np.asarray(lower), np.asarray(upper)
    except Exception as exc:  # pragma: no cover - depends on optional model cache
        logger.warning("joint-limit screen skipped (%s): %s", robot_description, exc)
        return None, None


def screen(
    q_plan: np.ndarray,
    robot_description: str,
) -> Verdict:
    q = np.asarray(q_plan, dtype=np.float64)
    if q.ndim != 2 or not q.size or not np.isfinite(q).all():
        raise ValueError("plan must be a finite, non-empty (T, nq) trajectory")

    q_min, q_max = _joint_limits(robot_description)
    if q_min is not None and q.shape[1] == q_min.shape[0]:
        if np.any(q < q_min - 1e-6) or np.any(q > q_max + 1e-6):
            return Verdict("reject", "joint_limit")

    return Verdict("dry-run", "")
