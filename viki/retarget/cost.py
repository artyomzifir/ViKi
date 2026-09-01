"""
viki.retarget.cost
------------------
Single assembly point for the retargeting cost functional as PINK tasks
(paper eq. 4).

Terms, in order of effect:

  * **frame task** — the data term: end-effector pose tracks the human target.
  * **acceleration regulariser λ_a** — a ``PostureTask`` whose per-frame target
    is the geodesic constant-velocity extrapolation ``q_pred`` of the last two
    configurations, so its residual is exactly
    ``‖q_t − q_pred‖² = ‖q_t − 2q_{t−1} + q_{t−2}‖²`` (discrete acceleration).
    This is what replaces the old post-hoc Savitzky–Golay pass on the joint
    trajectory — the smoothness is now *in* the solve, C² by construction.

  * **Huber robustifier** ρ_δ on the data residual and the **collision /
    self-collision barriers** h_j — a later pass; they refine but do not change
    the structure. Stubs below.
"""

from __future__ import annotations

from typing import Any


def build_tasks(pin: Any, robot: Any, cfg: Any, *, orientation_cost: float):
    """PINK tasks for one differential-IK solve.

    Returns ``(frame_task, accel_task, task_list)``. The caller sets
    ``frame_task``'s target per frame and ``accel_task``'s target per frame
    from :func:`accel_reference`; ``task_list`` is what goes to ``solve_ik``.
    """
    from pink.tasks import FrameTask, PostureTask

    frame_task = FrameTask(
        cfg.robot.ee_frame,
        position_cost=cfg.ik_position_cost,
        orientation_cost=orientation_cost,
    )
    accel_task = PostureTask(cost=float(cfg.ik_accel_cost))  # λ_a
    return frame_task, accel_task, [frame_task, accel_task]


def accel_reference(pin: Any, model: Any, q_prev: Any, q_prev2: Any):
    """Geodesic constant-velocity extrapolation of the last two configs:
    ``q_pred = q_{t-1} ⊕ (q_{t-1} ⊖ q_{t-2})``. Pulling ``q_t`` toward this
    penalises the discrete acceleration on whatever joint manifold the model
    uses (revolute, free-flyer, …). With no history, returns ``q_prev``.
    """
    import numpy as np

    if q_prev is None:
        return None
    if q_prev2 is None:
        return np.asarray(q_prev, dtype=np.float64).copy()
    dv = pin.difference(model, q_prev2, q_prev)          # tangent q_{t-2} → q_{t-1}
    return np.asarray(pin.integrate(model, q_prev, dv), dtype=np.float64)


# ── later pass (paper eq. 4, §3.7) ──────────────────────────────────────────


def huber_residual(*_args, **_kwargs):
    raise NotImplementedError("Huber data-term robustifier ρ_δ not implemented yet")


def collision_barriers(*_args, **_kwargs):
    raise NotImplementedError(
        "collision / self-collision control-barrier functions h_j not implemented yet"
    )
