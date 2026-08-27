"""lift_to_3d emits per-landmark fusion weights (paper eq. 2)."""

import numpy as np

from viki.contracts import HAND_LM_COUNT, HandDetection, LM, PreparedFrame
from viki.perception.geometry import lift_to_3d

_K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)


class _Id:
    def project_color_to_depth(self, u, v, z):
        return (u, v)


def _det(conf=0.9):
    pts = {LM(i): np.array([320.0, 240.0], dtype=np.float32) for i in range(HAND_LM_COUNT)}
    return HandDetection(
        points=pts,
        lm_z_rel=np.zeros(HAND_LM_COUNT, dtype=np.float32),
        confidence=conf,
        device_id="cam0",
        timestamp_us=0,
    )


def _frame(depth_val=0.5):
    d = np.full((480, 640), depth_val, dtype=np.float32)
    return PreparedFrame(rgb=np.zeros((480, 640, 3), np.uint8), depth_m=d,
                         depth_K=_K, device_id="cam0", timestamp_us=0)


def test_weights_present_and_positive():
    lms = lift_to_3d(_det(), _frame(), _Id())
    assert lms.weights is not None
    assert set(lms.weights) == set(lms.points)
    assert all(w > 0 for w in lms.weights.values())


def test_closer_surface_gets_more_weight():
    near = lift_to_3d(_det(), _frame(0.4), _Id()).weights[LM.WRIST]
    far = lift_to_3d(_det(), _frame(1.2), _Id()).weights[LM.WRIST]
    assert near > far  # d^-2 term


def test_visibility_scales_weight_linearly():
    hi = lift_to_3d(_det(conf=0.9), _frame(), _Id()).weights[LM.WRIST]
    lo = lift_to_3d(_det(conf=0.3), _frame(), _Id()).weights[LM.WRIST]
    assert hi / lo == np.float32(3.0) or abs(hi / lo - 3.0) < 1e-4
