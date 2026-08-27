"""Weighted cross-camera fusion (viki.prepare.fuse)."""

import numpy as np

from viki.prepare.fuse import fuse_trajectories

_IDS = np.arange(3, dtype=np.int32)  # 3 landmarks


def _traj(value: float, n: int = 4) -> np.ndarray:
    return np.full((n, 3, 3), value, dtype=np.float32)


def _ts(n: int = 4) -> np.ndarray:
    return (np.arange(n) * 1000).astype(np.int64)


def test_equal_weights_match_plain_mean():
    trajs = {"a": _traj(0.0), "b": _traj(2.0)}
    tss = {"a": _ts(), "b": _ts()}
    fused_unw, _ = fuse_trajectories(trajs, tss, _IDS)
    fused_w, _ = fuse_trajectories(
        trajs, tss, _IDS, weights={"a": np.ones((4, 3)), "b": np.ones((4, 3))}
    )
    np.testing.assert_allclose(fused_unw, 1.0)
    np.testing.assert_allclose(fused_w, fused_unw)


def test_zero_weight_drops_that_cameras_vote():
    trajs = {"a": _traj(0.0), "b": _traj(2.0)}
    tss = {"a": _ts(), "b": _ts()}
    fused, _ = fuse_trajectories(
        trajs, tss, _IDS, weights={"a": np.zeros((4, 3)), "b": np.ones((4, 3))}
    )
    np.testing.assert_allclose(fused, 2.0)


def test_weighted_average_between_cameras():
    trajs = {"a": _traj(0.0), "b": _traj(10.0)}
    tss = {"a": _ts(), "b": _ts()}
    fused, _ = fuse_trajectories(
        trajs, tss, _IDS,
        weights={"a": np.full((4, 3), 3.0), "b": np.full((4, 3), 1.0)},
    )
    np.testing.assert_allclose(fused, 2.5)  # (3*0 + 1*10) / 4
