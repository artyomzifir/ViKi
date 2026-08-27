"""Contract tests for viki.perception.backends."""

import numpy as np
import pytest

from viki.contracts import HAND_LM_COUNT, HandDetection, LM, PreparedFrame
from viki.perception.backends import BACKENDS, HandPoseBackend, load_backend


class _FakeBackend(HandPoseBackend):
    """Minimal in-spec backend used to exercise the contract."""

    name = "fake"

    def detect(self, frame: PreparedFrame, hand):
        pts = {LM(i): np.array([i * 1.0, i * 2.0], dtype=np.float32) for i in range(HAND_LM_COUNT)}
        return HandDetection(
            points=pts,
            lm_z_rel=np.zeros(HAND_LM_COUNT, dtype=np.float32),
            confidence=0.9,
            device_id=frame.device_id,
            timestamp_us=frame.timestamp_us,
        )


def _frame():
    return PreparedFrame(
        rgb=np.zeros((8, 8, 3), dtype=np.uint8),
        depth_m=np.zeros((8, 8), dtype=np.float32),
        depth_K=np.eye(3, dtype=np.float32),
        device_id="cam0",
        timestamp_us=1,
    )


def test_registry_has_the_four_backends():
    assert set(BACKENDS) == {"mediapipe", "rtmpose", "hamer", "yolo"}


def test_unknown_backend_raises():
    with pytest.raises(ValueError):
        load_backend("nope")


@pytest.mark.parametrize("name", ["rtmpose", "hamer", "yolo"])
def test_stub_backends_raise_not_implemented(name):
    with pytest.raises(NotImplementedError):
        load_backend(name)


def test_fake_backend_satisfies_contract():
    det = _FakeBackend().detect(_frame(), "right")
    assert isinstance(det, HandDetection)
    assert set(det.points) == {LM(i) for i in range(HAND_LM_COUNT)}
    assert det.lm_z_rel.shape == (HAND_LM_COUNT,)
    assert 0.0 <= det.confidence <= 1.0
