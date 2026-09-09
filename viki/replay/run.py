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
import json

import numpy as np

from viki.contracts import Episode, REPLAY_KEYS
from viki.episode import mark_stage
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
        q_plan = np.asarray(plan["q"], dtype=np.float64)
        q_approach = np.asarray(
            plan["q_approach"] if "q_approach" in plan else np.empty((0, q_plan.shape[1])),
            dtype=np.float64,
        )
        dt = float(plan["dt"])
        robot = str(plan["robot"])
        command = np.asarray(plan["gripper_command"], dtype=np.float64)
        gripper_model = str(plan["gripper_model"])
        command_names = (
            json.loads(str(plan["gripper_command_names_json"]))
            if "gripper_command_names_json" in plan else []
        )
    if command.shape != (len(q_plan), 1):
        raise ValueError(
            f"the current replay driver accepts one-dimensional gripper "
            f"commands, got {gripper_model!r} {command.shape}"
        )
    # V3 and older plans encoded ``closed`` (1 closed). V4 encodes normalised
    # ``opening`` (0 closed, 1 open). Drivers only see the current convention.
    legacy_closed = command_names == ["closed"] or (
        not command_names and gripper_model == "binary"
    )
    gripper = 1.0 - command[:, 0] if legacy_closed else command[:, 0]
    if not np.isfinite(gripper).all() or np.any(gripper < 0.0) or np.any(gripper > 1.0):
        raise ValueError("gripper command must be a normalised opening in [0, 1]")
    if q_approach.ndim != 2 or q_approach.shape[1:] != q_plan.shape[1:]:
        raise ValueError("q_approach must have the same joint dimension as q")
    approach_gripper = np.full(
        len(q_approach),
        gripper[0] if len(gripper) else 1.0,
        dtype=np.float64,
    )
    q_execute = np.concatenate((q_approach, q_plan), axis=0)
    gripper_execute = np.concatenate((approach_gripper, gripper), axis=0)

    drv = load_driver(driver)
    try:
        execution_log = drv.execute(q_execute, gripper_execute, dt)
    finally:
        drv.close()

    if len(execution_log.q_attained) != len(q_execute):
        raise RuntimeError("replay driver returned a trajectory with the wrong length")
    approach_frames = len(q_approach)
    q_attained = np.asarray(execution_log.q_attained)[approach_frames:]
    gripper_attained = np.asarray(execution_log.gripper_attained)[approach_frames:]
    controller_residual = np.asarray(execution_log.controller_residual)[approach_frames:]

    if max_resolves:
        logger.warning("replay re-solve loop is not implemented (paper §3.8, Fig. 3.1)")

    # Screen the complete executed motion, including the transition from home.
    v = screen(execution_log.q_attained, execution_log.controller_residual, robot)

    archive = {
        # Keep replay.h5 aligned one-to-one with the demonstrated frames used by
        # export. The approach was executed and screened above, but has no human
        # observation/annotation counterpart.
        "q_attained": q_attained,
        "gripper_attained": gripper_attained,
        "controller_residual": controller_residual,
        "verdict": v.verdict,
        "rejection_cause": v.cause,
        "resolve_attempts": 0,
        "robot": robot,
        "dt": dt,
    }
    assert set(archive) == set(REPLAY_KEYS)
    write_hdf5_archive(ep.replay_h5, archive)
    mark_stage(ep, "replay", verdict=v.verdict, rejection_cause=v.cause, driver=driver)
    logger.info("replay %s: verdict=%s cause=%s", ep.id, v.verdict, v.cause or "-")
    return str(ep.replay_h5)
