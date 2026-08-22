"""
Tests for geometry utilities.
Verifies lift_to_3d with MediaPipe z_rel + depth median.
"""

import numpy as np
from viki.skeleton.geometry import lift_to_3d
from viki.skeleton.models import HandDetection, PreparedFrame, LM


class _MockBackend:
    """Mock backend returning identity projection (color = depth)."""

    def __init__(self) -> None:
        di = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
        self._fx, self._fy = di[0, 0], di[1, 1]
        self._cx, self._cy = di[0, 2], di[1, 2]

    def project_color_to_depth(self, u: float, v: float, z: float) -> tuple[float, float]:
        return (u, v)

    def deproject_2d_to_3d(self, u: float, v: float, z: float) -> tuple[float, float, float]:
        X = (u - self._cx) * z / self._fx
        Y = (v - self._cy) * z / self._fy
        return (float(X), float(Y), z)


def test_lift_to_3d_all_valid():
    """All 21 landmarks projected at a uniform depth blob → all valid."""
    depth_m = np.full((576, 640), np.nan, dtype=np.float32)
    depth_m[200:280, 300:380] = 0.5

    depth_K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
    frame = PreparedFrame(
        rgb=np.zeros((720, 1280, 3), dtype=np.uint8),
        depth_m=depth_m,
        K=np.eye(3, dtype=np.float32),
        depth_K=depth_K,
        device_id="cam0",
        timestamp_us=0,
    )

    points = {LM(i): np.array([340.0, 240.0], dtype=np.float32) for i in range(LM.N)}
    z_rel = np.linspace(-0.3, 0.5, LM.N, dtype=np.float32)
    det = HandDetection(
        points=points, lm_z_rel=z_rel, confidence=0.95,
        device_id="cam0", timestamp_us=0,
    )

    result = lift_to_3d(det, frame, _MockBackend())
    valid = sum(1 for p in result.points.values() if not np.isnan(p).any())
    assert valid == LM.N, f"Expected {LM.N} valid, got {valid}"

    # Wrist Z should be close to the median depth
    wrist_z = result.points[LM.WRIST][2]
    assert abs(wrist_z - 0.5) < 0.05, f"Wrist Z {wrist_z:.3f} far from 0.5"


def test_lift_to_3d_no_depth():
    """All NaN depth → all landmarks NaN."""
    depth_m = np.full((576, 640), np.nan, dtype=np.float32)
    depth_K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
    frame = PreparedFrame(
        rgb=np.zeros((720, 1280, 3), dtype=np.uint8),
        depth_m=depth_m,
        K=np.eye(3, dtype=np.float32),
        depth_K=depth_K,
        device_id="cam0",
        timestamp_us=0,
    )

    points = {LM(i): np.array([340.0, 240.0], dtype=np.float32) for i in range(LM.N)}
    z_rel = np.zeros(LM.N, dtype=np.float32)
    det = HandDetection(
        points=points, lm_z_rel=z_rel, confidence=0.95,
        device_id="cam0", timestamp_us=0,
    )

    result = lift_to_3d(det, frame, _MockBackend())
    valid = sum(1 for p in result.points.values() if not np.isnan(p).any())
    assert valid == 0, f"Expected 0 valid with no depth, got {valid}"


def test_lift_to_3d_nan_landmarks():
    """Some MediaPipe landmarks NaN → those individually NaN, others valid."""
    depth_m = np.full((576, 640), 0.5, dtype=np.float32)
    depth_K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
    frame = PreparedFrame(
        rgb=np.zeros((720, 1280, 3), dtype=np.uint8),
        depth_m=depth_m,
        K=np.eye(3, dtype=np.float32),
        depth_K=depth_K,
        device_id="cam0",
        timestamp_us=0,
    )

    points: dict[LM, np.ndarray] = {}
    for i in range(LM.N):
        points[LM(i)] = np.array([340.0, 240.0], dtype=np.float32)
    points[LM(21)] = np.array([np.nan, np.nan])  # NaN landmark
    points[LM(22)] = np.array([np.nan, np.nan])

    z_rel = np.linspace(-0.3, 0.5, LM.N, dtype=np.float32)
    det = HandDetection(
        points=points, lm_z_rel=z_rel, confidence=0.95,
        device_id="cam0", timestamp_us=0,
    )

    result = lift_to_3d(det, frame, _MockBackend())
    valid = sum(1 for p in result.points.values() if not np.isnan(p).any())
    assert valid == LM.N - 2, f"Expected {LM.N-2} valid, got {valid}"


if __name__ == "__main__":
    test_lift_to_3d_all_valid()
    test_lift_to_3d_no_depth()
    test_lift_to_3d_nan_landmarks()
    print("All unit tests passed!")
