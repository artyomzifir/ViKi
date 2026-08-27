"""
viki.replay.run
---------------
Pipeline stage 5: plan.h5 -> replay.h5 (paper §3.8).

Execute the synthesised joint trajectory, log what the robot actually attained,
screen for feasibility, and — on rejection — optionally re-solve retargeting
with modified weights (bounded).

STUB stage: with ``driver="dryrun"`` no hardware is touched and the verdict is
``dry-run``. ``driver="ur3"`` raises unless ``ur-rtde`` and the robot are present.
The re-solve loop is not implemented (``max_resolves`` is accepted but ignored).
"""

from __future__ import annotations

import logging

import numpy as np

from viki.contracts import Episode, REPLAY_KEYS
from viki.replay.driver import load_driver
from viki.replay.screen import screen
from viki.retarget.archive import load_archive, write_hdf5_archive

logger = logging.getLogger(__name__)


def replay_episode(
    ep: Episode,
    *,
    driver: str = "dryrun",
    max_resolves: int = 0,
) -> str:
    """Run replay for one episode and write ``replay.h5``. Returns the path."""
    if not ep.plan_h5.exists():
        raise FileNotFoundError(f"no plan.h5 for episode {ep.id}; run retarget first")

    with load_archive(ep.plan_h5) as plan:
        q_plan = np.asarray(plan["q_scene_smooth"], dtype=np.float64)
        dt = float(plan["dt"]) if "dt" in plan else 1.0 / 30.0
        robot = str(plan["robot"]) if "robot" in plan else ""

    gripper = _load_gripper(ep, len(q_plan))

    drv = load_driver(driver)
    try:
        log = drv.execute(q_plan, gripper, dt)
    finally:
        drv.close()

    if max_resolves:
        logger.warning("replay re-solve loop is not implemented (paper §3.8, Fig. 3.1)")

    v = screen(log.q_attained, log.controller_residual, robot)

    archive = {
        "q_attained": log.q_attained,
        "gripper_attained": log.gripper_attained,
        "controller_residual": log.controller_residual,
        "verdict": v.verdict,
        "rejection_cause": v.cause,
        "resolve_attempts": 0,
        "robot": robot,
        "dt": dt,
    }
    assert set(archive) == set(REPLAY_KEYS)
    write_hdf5_archive(ep.replay_h5, archive)
    logger.info("replay %s: verdict=%s cause=%s", ep.id, v.verdict, v.cause or "-")
    return str(ep.replay_h5)


def _load_gripper(ep: Episode, n: int) -> np.ndarray:
    if ep.cln_npz.exists():
        with np.load(ep.cln_npz) as d:
            if "gripper" in d:
                g = np.asarray(d["gripper"], dtype=bool)
                if len(g) == n:
                    return g
    return np.zeros(n, dtype=bool)
