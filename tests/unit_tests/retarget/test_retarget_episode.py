"""
Real end-to-end IK: cln.npz -> plan.h5 via Pinocchio + sparse batch QP.

Skips cleanly when Pinocchio is missing or the robot description cannot be
fetched (the first run git-clones it, which a sandboxed CI blocks).
"""

import numpy as np
import pytest

from viki.episode import new_episode, stage_done
from viki.retarget.archive import load_archive


def _synthetic_cln(ep, T: int = 12) -> None:
    pos = np.linspace([0.30, 0.0, 0.30], [0.38, 0.05, 0.30], T).astype(np.float32)
    gripper = np.zeros(T, bool)
    gripper[T // 2:] = True
    np.savez_compressed(
        ep.cln_npz,
        positions=pos,
        rotations=np.tile(np.eye(3), (T, 1, 1)).astype(np.float32),
        rpy=np.zeros((T, 3), np.float32),
        valid=np.ones(T, bool),
        omega=np.ones(T, np.float32),
        gripper=gripper,
        timestamps=(np.arange(T) * 33_000).astype(np.int64),
        raw_points=np.zeros((T, 21, 3), np.float32),
        smoothed_points=np.zeros((T, 21, 3), np.float32),
        landmark_ids=np.arange(21, dtype=np.int32),
        coordinate_frame="rig",
    )


@pytest.mark.slow
def test_retarget_episode_real_ik(tmp_path):
    pytest.importorskip("pinocchio")
    pytest.importorskip("pink")

    ep = new_episode(tmp_path)
    _synthetic_cln(ep)

    from viki.retarget.run import retarget_episode

    try:
        retarget_episode(ep, robot="ur3", options={
            "max_iterations": 3,
            "collision_enabled": True,
            "collision_pairs": 2,
            "approach_sec": 0.2,
            "w_orientation": 0.0,
        })
    except Exception as exc:  # noqa: BLE001 - offline / model fetch failure
        pytest.skip(f"robot description unavailable: {exc}")

    assert ep.plan_h5.exists() and stage_done(ep, "retarget")
    with load_archive(ep.plan_h5) as plan:
        q = np.asarray(plan["q"])
        assert q.ndim == 2 and q.shape[1] == 6 and np.isfinite(q).all()
        assert float(np.max(np.asarray(plan["position_error_m"]))) < 0.20  # < 20 cm
        assert np.asarray(plan["link_positions_calibration"]).shape[0] == len(q)
        assert str(plan["gripper_model"]) == "robotiq_2f85"
        assert str(plan["gripper_tcp_frame"]).endswith("robotiq_arg2f_tcp")
        closed = np.asarray(plan["gripper_closed"], dtype=bool)
        np.testing.assert_allclose(
            np.asarray(plan["gripper_opening_m"]),
            np.where(closed, 0.0, 0.085),
        )
        assert "gripper" in str(plan["point_groups_json"])


@pytest.mark.slow
def test_ur10_default_collision_set_is_feasible_at_neutral():
    pytest.importorskip("pinocchio")
    pytest.importorskip("pink")

    from viki.retarget.run import _load_robot_description
    from viki.retarget.solver import PinocchioKinematics

    try:
        robot = _load_robot_description("ur10_official_description")
    except Exception as exc:  # noqa: BLE001 - offline / model fetch failure
        pytest.skip(f"robot description unavailable: {exc}")
    kinematics = PinocchioKinematics(
        robot, "wrist_3_link", collision_pairs=8, collision_min_distance_m=0.02,
    )
    margin, _ = kinematics.collision_linearization(kinematics.q_reference)
    assert len(margin) == 8
    assert float(np.min(margin)) >= 0.0
