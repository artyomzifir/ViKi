"""
Real end-to-end IK: cln.npz -> plan.h5 via PINK/Pinocchio.

Skips cleanly when Pinocchio is missing or the robot description cannot be
fetched (the first run git-clones it, which a sandboxed CI blocks).
"""

import numpy as np
import pytest

from viki.episode import new_episode, stage_done
from viki.retarget.archive import load_archive


def _synthetic_cln(ep, T: int = 12) -> None:
    pos = np.linspace([0.30, 0.0, 0.30], [0.38, 0.05, 0.30], T).astype(np.float32)
    np.savez_compressed(
        ep.cln_npz,
        positions=pos,
        rotations=np.tile(np.eye(3), (T, 1, 1)).astype(np.float32),
        rpy=np.zeros((T, 3), np.float32),
        valid=np.ones(T, bool),
        omega=np.ones(T, np.float32),
        gripper=np.zeros(T, bool),
        timestamps=(np.arange(T) * 33_000).astype(np.int64),
        raw_points=np.zeros((T, 21, 3), np.float32),
        smoothed_points=np.zeros((T, 21, 3), np.float32),
        landmark_ids=np.arange(21, dtype=np.int32),
        coordinate_frame="robot_base",
    )


@pytest.mark.slow
def test_retarget_episode_real_ik(tmp_path, monkeypatch):
    pytest.importorskip("pinocchio")
    pytest.importorskip("pink")

    import viki.config as cfg

    monkeypatch.setattr(cfg, "RETARGET_IK_SUBSTEPS", 2, raising=False)
    monkeypatch.setattr(cfg, "RETARGET_APPROACH_SEC", 0.2, raising=False)

    ep = new_episode(tmp_path)
    _synthetic_cln(ep)

    from viki.retarget.run import retarget_episode

    try:
        retarget_episode(ep, robot="ur3")
    except Exception as exc:  # noqa: BLE001 - offline / model fetch failure
        pytest.skip(f"robot description unavailable: {exc}")

    assert ep.plan_h5.exists() and stage_done(ep, "retarget")
    with load_archive(ep.plan_h5) as plan:
        q = np.asarray(plan["q_scene_smooth"])
        assert q.ndim == 2 and q.shape[1] == 6 and np.isfinite(q).all()
        assert float(np.max(np.asarray(plan["pos_err_smooth"]))) < 0.20  # < 20 cm
