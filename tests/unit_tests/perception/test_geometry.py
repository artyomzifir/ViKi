"""lift_to_3d: 2-D hand detection + measured depth -> 3-D camera-frame landmarks."""

import numpy as np

from viki.contracts import HAND_LM_COUNT, HandDetection, LM, PreparedFrame
from viki.perception.geometry import lift_to_3d


class _IdentityProjector:
    """Colour pixel == depth pixel."""

    def project_color_to_depth(self, u, v, z):
        return (u, v)


_K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)


def _frame(depth_m):
    return PreparedFrame(
        rgb=np.zeros((576, 640, 3), dtype=np.uint8),
        depth_m=depth_m,
        depth_K=_K,
        device_id="cam0",
        timestamp_us=0,
    )


def _detection(points=None):
    pts = points or {
        LM(i): np.array([340.0, 240.0], dtype=np.float32) for i in range(HAND_LM_COUNT)
    }
    return HandDetection(
        points=pts,
        lm_z_rel=np.zeros(HAND_LM_COUNT, dtype=np.float32),
        confidence=0.95,
        device_id="cam0",
        timestamp_us=0,
    )


def _n_valid(result):
    return sum(1 for p in result.points.values() if not np.isnan(p).any())


def test_all_landmarks_lift_over_a_depth_blob():
    depth = np.full((576, 640), np.nan, dtype=np.float32)
    depth[200:280, 300:380] = 0.5
    result = lift_to_3d(_detection(), _frame(depth), _IdentityProjector())
    assert _n_valid(result) == HAND_LM_COUNT
    assert abs(result.points[LM.WRIST][2] - 0.5) < 0.05


def test_no_depth_returns_none():
    depth = np.full((576, 640), np.nan, dtype=np.float32)
    result = lift_to_3d(_detection(), _frame(depth), _IdentityProjector())
    # The hand region is a depth hole -> the frame is dropped.
    assert result is None


def test_nan_pixel_landmarks_are_individually_dropped():
    pts = {
        LM(i): np.array([340.0, 240.0], dtype=np.float32) for i in range(HAND_LM_COUNT)
    }
    pts[LM.PINKY_TIP] = np.array([np.nan, np.nan], dtype=np.float32)
    pts[LM.RING_TIP] = np.array([np.nan, np.nan], dtype=np.float32)
    result = lift_to_3d(
        _detection(pts), _frame(np.full((576, 640), 0.5, dtype=np.float32)),
        _IdentityProjector(),
    )
    assert _n_valid(result) == HAND_LM_COUNT - 2
