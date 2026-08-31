"""
viki.perception.backends.rtmpose
--------------------------------
Hand landmarks from RTMPose-Hand (MMPose), run through ``rtmlib`` on ONNX
Runtime. ``rtmlib.Hand`` bundles an RTMDet hand detector + the RTMPose-Hand
keypoint model + SimCC decoding, and downloads its own ONNX weights to
``~/.cache/rtmlib`` on first use.

RTMPose-Hand returns 21 keypoints already in the MediaPipe hand topology, so the
index map to :class:`~viki.contracts.LM` is 1:1. There is no per-landmark z, so
``lm_z_rel`` is zeros — the depth lift in :mod:`viki.perception.geometry` uses
measured depth and does not need it. RTMPose does not classify left/right; ViKi
tracks a single hand and the caller picks which, so we take the top-scoring hand
in the frame and trust the requested ``hand``.
"""

from __future__ import annotations

import logging

import numpy as np

from viki.contracts import HAND_LM_COUNT, Hand, HandDetection, LM, PreparedFrame
from viki.perception.backends.base import HandPoseBackend
from viki.perception.backends.registry import RTM_DET_URL, get as _get_model

logger = logging.getLogger(__name__)


def _pick_device() -> str:
    try:
        import onnxruntime as ort

        return "cuda" if "CUDAExecutionProvider" in ort.get_available_providers() else "cpu"
    except Exception:  # noqa: BLE001
        return "cpu"


class RTMPoseHandBackend(HandPoseBackend):
    """RTMPose-Hand via rtmlib. One instance per camera stream."""

    name = "rtmpose"

    def __init__(
        self,
        *,
        mode: str = "video",  # accepted for parity with MediaPipe; unused
        model_entry: dict | None = None,
        min_confidence: float = 0.5,
        device: str | None = None,
        **_ignored,
    ) -> None:
        try:
            from rtmlib import Hand as _RtmHand
        except ImportError as exc:  # pragma: no cover - dep not in the base image
            raise RuntimeError(
                "RTMPose backend needs `rtmlib` + `onnxruntime` "
                "(add to pyproject.toml and rebuild the image)"
            ) from exc

        entry = model_entry or _get_model("rtmpose-m-hand5")
        pose_url = entry["pose_url"]
        self._min_conf = float(min_confidence)
        self._tier = entry["id"]
        self._device = device or _pick_device()
        self._warned_multi = False
        logger.info(
            "RTMPose-Hand: %s device=%s (onnxruntime)", self._tier, self._device
        )
        self._hand = _RtmHand(
            mode="lightweight",
            det=RTM_DET_URL, det_input_size=(320, 320),
            pose=pose_url, pose_input_size=(256, 256),
            backend="onnxruntime", device=self._device,
        )

    def detect(self, frame: PreparedFrame, hand: Hand) -> HandDetection | None:
        # rtmlib is cv2-based and expects BGR; PreparedFrame.rgb is RGB.
        bgr = np.ascontiguousarray(frame.rgb[:, :, ::-1])
        keypoints, scores = self._hand(bgr)  # (N,21,2), (N,21)
        if keypoints is None or len(keypoints) == 0:
            return None

        mean = scores.mean(axis=1)
        best = int(np.argmax(mean))
        if len(keypoints) > 1 and not self._warned_multi:
            logger.warning(
                "RTMPose sees %d hands on %s; taking the top-scoring one as %r "
                "(RTMPose has no left/right label)",
                len(keypoints), frame.device_id, hand,
            )
            self._warned_multi = True
        if float(mean[best]) < self._min_conf:
            return None

        kp = keypoints[best]
        points = {
            LM(i): np.asarray(kp[i], dtype=np.float32) for i in range(HAND_LM_COUNT)
        }
        return HandDetection(
            points=points,
            lm_z_rel=np.zeros(HAND_LM_COUNT, dtype=np.float32),
            confidence=float(mean[best]),
            device_id=frame.device_id,
            timestamp_us=frame.timestamp_us,
        )

    def close(self) -> None:
        self._hand = None
