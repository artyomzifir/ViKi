"""
viki.retarget.frames
--------------------
Robot-base registration.

The manipulator is bolted at a fixed pose relative to the workspace ChArUco
anchor, so the world→robot transform ``T^W_R`` is a static calibration constant
read from config (``RETARGET_BASE_ROTATION`` 3x3 + ``RETARGET_BASE_TRANSLATION``
3-vector). A proper hand-eye procedure (observe a ChArUco target held in a known
end-effector pose; Daniilidis dual-quaternion — paper §3.3) would produce these
numbers; for now they are set by hand in the config.
"""

from __future__ import annotations

import numpy as np

__all__ = ["world_to_robot"]


def world_to_robot(cfg) -> np.ndarray:
    """4x4 homogeneous transform mapping a point in the workspace/world frame
    into the robot base frame. Identity when the config leaves it unset."""
    R = np.asarray(
        cfg.get("RETARGET_BASE_ROTATION", np.eye(3).tolist()), dtype=np.float64
    ).reshape(3, 3)
    t = np.asarray(
        cfg.get("RETARGET_BASE_TRANSLATION", [0.0, 0.0, 0.0]), dtype=np.float64
    ).reshape(3)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T
