"""
viki.replay.screen
------------------
Feasibility screening of a replayed trajectory (paper §3.8).

Working: joint-limit violation from the robot model. STUB: singularity
proximity, collision, and controller tracking-fault classification — these need
the FK/Jacobian and a collision model, so for now they are not checked and the
verdict falls through to ``pass`` (or ``dry-run`` when the residual is unmeasured).
"""

from __future__ import annotations

import logging
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Verdict:
    verdict: str  # pass | reject | dry-run
    cause: str  # "" | joint_limit | singularity | collision | tracking_fault


def _joint_limits(robot_description: str):
    """(q_min, q_max) from the Pinocchio model, or (None, None) if unavailable."""
    try:
        from robot_descriptions.loaders.pinocchio import load_robot_description

        model = load_robot_description(robot_description).model
        return np.asarray(model.lowerPositionLimit), np.asarray(model.upperPositionLimit)
    except Exception as exc:  # pragma: no cover - depends on optional model cache
        logger.warning("joint-limit screen skipped (%s): %s", robot_description, exc)
        return None, None


def screen(
    q_attained: np.ndarray,
    controller_residual: np.ndarray,
    robot_description: str,
    *,
    residual_threshold: float = 0.05,
) -> Verdict:
    q = np.asarray(q_attained, dtype=np.float64)

    q_min, q_max = _joint_limits(robot_description)
    if q_min is not None and q.shape[1] == q_min.shape[0]:
        if np.any(q < q_min - 1e-6) or np.any(q > q_max + 1e-6):
            return Verdict("reject", "joint_limit")

    resid = np.asarray(controller_residual, dtype=np.float64)
    if not np.isfinite(resid).any():
        return Verdict("dry-run", "")
    if np.nanmax(resid) > residual_threshold:
        return Verdict("reject", "tracking_fault")

    # singularity / collision checks: not implemented (paper §3.8)
    return Verdict("pass", "")
