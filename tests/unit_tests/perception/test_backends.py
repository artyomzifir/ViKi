"""Contract tests for viki.perception.backends."""

import numpy as np
import pytest
from types import SimpleNamespace

from viki.contracts import HAND_LM_COUNT, HandDetection, LM, PreparedFrame
from viki.perception.backends import HandPoseBackend, load_backend
from viki.perception.backends import registry


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


def test_registry_is_a_flat_21pt_model_list():
    ids = {m["id"] for m in registry.MODELS}
    assert {"mediapipe", "rtmpose-m-hand5"} <= ids
    for m in registry.MODELS:
        assert m["impl"] in {"mediapipe", "rtmpose", "mmpose-heatmap"}
        assert m["license"] in {"Apache-2.0", "MIT"}
    rows = registry.list_models()
    assert all({"id", "label", "present", "downloadable"} <= set(r) for r in rows)


def test_unknown_model_raises():
    with pytest.raises(ValueError):
        load_backend("nope")


def test_mmpose_heatmap_model_needs_a_local_onnx():
    # no ONNX in models/ -> the backend must refuse with a helpful message
    with pytest.raises(RuntimeError):
        load_backend("hrnetv2-w18-hand")


def test_backend_classes_are_in_spec():
    from viki.perception.backends.rtmpose import RTMPoseHandBackend
    from viki.perception.backends.mmpose_heatmap import MMPoseHeatmapBackend

    assert issubclass(RTMPoseHandBackend, HandPoseBackend)
    assert issubclass(MMPoseHeatmapBackend, HandPoseBackend)


def test_fake_backend_satisfies_contract():
    det = _FakeBackend().detect(_frame(), "right")
    assert isinstance(det, HandDetection)
    assert set(det.points) == {LM(i) for i in range(HAND_LM_COUNT)}
    assert det.lm_z_rel.shape == (HAND_LM_COUNT,)
    assert 0.0 <= det.confidence <= 1.0


def test_mediapipe_zero_threshold_has_a_native_safe_floor():
    from viki.perception.backends.mediapipe import _graph_confidence

    assert 0.0 < _graph_confidence(0.0) < 1e-5
    assert _graph_confidence(0.5) == 0.5


def test_mediapipe_ignores_the_handedness_label():
    """A mislabelled hand must still be used.

    MediaPipe infers left/right from one 2-D view, so two cameras looking at
    the same physical hand from opposite sides disagree. The episode declares
    which hand was recorded, so the label carries no information we need and
    must never discard a detection.
    """
    from viki.perception.backends.mediapipe import MediaPipeHandBackend

    landmarks = [
        SimpleNamespace(x=0.5, y=0.5, z=0.0) for _ in range(HAND_LM_COUNT)
    ]
    backend = MediaPipeHandBackend.__new__(MediaPipeHandBackend)

    for label, score in (("Left", 0.51), ("Left", 0.97), ("Right", 0.99)):
        raw = SimpleNamespace(
            hand_landmarks=[landmarks],
            handedness=[[SimpleNamespace(category_name=label, score=score)]],
        )
        detection = backend._extract(raw, _frame())
        assert detection is not None, f"{label}@{score} was discarded"
        assert len(detection.points) == HAND_LM_COUNT
        # The label's certainty is not landmark quality; geometry owns rejection.
        assert detection.confidence == 1.0

    empty = SimpleNamespace(hand_landmarks=[], handedness=[])
    assert backend._extract(empty, _frame()) is None
