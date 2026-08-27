"""
viki.retarget
-------------
Pipeline stage 4: cln.npz -> plan.h5.

Object-relative / workspace-anchored end-effector targets -> PINK IK against a
Pinocchio robot description (:mod:`viki.retarget.robots`), moved into the robot
base frame from config (:mod:`viki.retarget.frames`). Cost-functional assembly
lives in :mod:`viki.retarget.cost` (partly stubbed, paper eq. 4).
"""

from viki.retarget.robots import RobotConfig, normalize_robot  # noqa: F401
from viki.retarget.run import RunConfig, retarget, retarget_from_poses  # noqa: F401

__all__ = [
    "RobotConfig",
    "normalize_robot",
    "RunConfig",
    "retarget",
    "retarget_from_poses",
]
