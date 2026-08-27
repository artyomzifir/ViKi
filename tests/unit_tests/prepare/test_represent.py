"""Object-relative representation stub (viki.prepare.represent)."""

import numpy as np

from viki.prepare.represent import object_relative


def test_none_object_track_returns_none():
    wrist = np.tile(np.eye(4), (5, 1, 1))
    assert object_relative(wrist, None) is None


def test_object_relative_is_inv_obj_times_hand():
    T = 3
    wrist = np.tile(np.eye(4), (T, 1, 1))
    wrist[:, :3, 3] = [[1.0, 2.0, 3.0]] * T
    obj = np.tile(np.eye(4), (T, 1, 1))
    obj[:, :3, 3] = [[1.0, 0.0, 0.0]] * T

    rel = object_relative(wrist, obj)
    assert rel.shape == (T, 4, 4)
    np.testing.assert_allclose(rel[:, :3, 3], [[0.0, 2.0, 3.0]] * T, atol=1e-9)
