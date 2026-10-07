"""
viki.prepare.represent
----------------------
Object-centric representation (paper §3.6).

The transfer-invariant form of a demonstration is the wrist pose expressed
relative to the manipulated object::

    T_obj_hand[t] = inv(T_world_obj[t]) @ T_world_hand[t]

Prepare runs before scene perception, so its call still passes ``None`` and
``cln.npz`` remains workspace-anchored. The post-IK ``viki object-relative``
stage calls this helper with the tracked object pose and writes a separate
sidecar without modifying the protected hand baseline.
"""

from __future__ import annotations

import logging

import numpy as np

logger = logging.getLogger(__name__)


def object_relative(
    wrist_world: np.ndarray,  # (T, 4, 4)
    object_world: np.ndarray | None,  # (T, 4, 4) or None
) -> np.ndarray | None:
    """
    Object-relative wrist trajectory, or ``None`` if no object track is available.
    """
    if object_world is None:
        logger.warning(
            "object-relative representation skipped: no object-pose tracker "
            "(paper §3.6); exporting workspace-anchored trajectory only"
        )
        return None

    w = np.asarray(wrist_world, dtype=np.float64)
    o = np.asarray(object_world, dtype=np.float64)
    if w.shape != o.shape or w.shape[1:] != (4, 4):
        raise ValueError(f"expected matching (T, 4, 4); got {w.shape} and {o.shape}")
    return np.linalg.inv(o) @ w
