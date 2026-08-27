"""
viki.retarget.cost
------------------
Assembly of the retargeting cost functional as PINK tasks (paper eq. 4).

Working today: a frame task (data term) + a posture task (velocity/posture
regularisation) — this is what ``run.py`` builds and solves.

STUBS (paper eq. 4, §3.7): the Huber robustifier ρ_δ on the data residual, the
explicit second-order term λ_a‖q_t − 2q_{t−1} + q_{t−2}‖², and the collision /
self-collision barrier functions h_j. They are gated off in ``RunConfig``
(``use_huber=False``, ``lambda_accel=0.0``, ``collisions=False``) so current
behaviour is unchanged; wiring them in is where this module grows.
"""

from __future__ import annotations


def build_tasks(pin, robot, cfg):
    """
    Return the list of PINK tasks for one solve.

    Currently ``[FrameTask, PostureTask]``. Kept as a seam so the extra terms
    below can be appended without touching the IK loop.
    """
    raise NotImplementedError(
        "cost.build_tasks is a stub — run.py still assembles tasks inline; "
        "this becomes the single assembly point when the eq. 4 terms land"
    )


def huber_residual(*_args, **_kwargs):
    raise NotImplementedError("Huber data-term robustifier not implemented (paper eq. 4)")


def acceleration_penalty(*_args, **_kwargs):
    raise NotImplementedError(
        "explicit acceleration term λ_a‖q_t − 2q_{t−1} + q_{t−2}‖² not implemented "
        "(paper eq. 4, DexFlow C² argument §3.7)"
    )


def collision_barriers(*_args, **_kwargs):
    raise NotImplementedError(
        "collision / self-collision control-barrier functions not implemented "
        "(paper eq. 4, constraint h_j)"
    )
