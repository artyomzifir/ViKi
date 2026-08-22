import numpy as np
from viki.skeleton.geometry import lift_to_3d
from viki.skeleton.models import HandDetection, PreparedFrame, LM


class _MockBackend:
    def __init__(self) -> None:
        di = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
        self._fx, self._fy = di[0, 0], di[1, 1]
        self._cx, self._cy = di[0, 2], di[1, 2]

    def project_color_to_depth(self, u, v, z):
        return (u, v)

    def deproject_2d_to_3d(self, u, v, z):
        X = (u - self._cx) * z / self._fx
        Y = (v - self._cy) * z / self._fy
        return (float(X), float(Y), z)


def test_lift_to_3d():
    backend = _MockBackend()
    K = np.array([[600, 0, 320], [0, 600, 240], [0, 0, 1]], dtype=np.float32)
    depth_m = np.full((576, 640), 0.5, dtype=np.float32)

    frame = PreparedFrame(
        rgb=np.zeros((720, 1280, 3), dtype=np.uint8),
        depth_m=depth_m,
        K=np.eye(3, dtype=np.float32),
        depth_K=K,
        device_id="cam0",
        timestamp_us=0,
    )

    points = {LM(i): np.array([320.0, 240.0]) for i in range(LM.N)}
    z_rel = np.linspace(-0.3, 0.5, LM.N, dtype=np.float32)
    det = HandDetection(
        points=points,
        lm_z_rel=z_rel,
        confidence=1.0,
        device_id="cam0",
        timestamp_us=0,
    )

    res = lift_to_3d(det, frame, backend)
    wrist_z = res.points[LM(0)][2]
    print(f"Wrist Z: {wrist_z:.3f}")
    assert abs(wrist_z - 0.5) < 0.1, f"Wrist Z {wrist_z:.3f} expected ~0.5"

    valid = sum(1 for p in res.points.values() if not np.isnan(p).any())
    print(f"Valid landmarks: {valid}/{LM.N}")
    assert valid == LM.N, f"Expected {LM.N} valid, got {valid}"


if __name__ == "__main__":
    try:
        test_lift_to_3d()
        print("All tests passed!")
    except Exception as e:
        print(f"Test failed: {e}")
        import traceback
        traceback.print_exc()
