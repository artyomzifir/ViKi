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
    """`"cuda"` only if a CUDAExecutionProvider session actually initialises —
    the provider can be *listed* (onnxruntime-gpu installed) yet fail at runtime
    (missing CUDA/cuDNN libs, driver too old for the GPU arch). Verify, don't
    assume; fall back to CPU with a warning."""
    try:
        import numpy as _np
        import onnxruntime as ort
    except Exception:  # noqa: BLE001
        return "cpu"
    # nvidia pip wheels drop libcublas/libcudnn/… where the loader can't see
    # them; preload_dlls() adds them explicitly (the image's ldconfig entry is
    # the primary fix, this is the backstop). Idempotent, no-op on old ORT.
    try:
        ort.preload_dlls()
    except Exception:  # noqa: BLE001
        pass
    if "CUDAExecutionProvider" not in ort.get_available_providers():
        return "cpu"
    try:
        from onnx import TensorProto, helper  # rtmlib pulls onnx in

        g = helper.make_graph(
            [helper.make_node("Identity", ["x"], ["y"])], "probe",
            [helper.make_tensor_value_info("x", TensorProto.FLOAT, [1])],
            [helper.make_tensor_value_info("y", TensorProto.FLOAT, [1])],
        )
        model = helper.make_model(g, opset_imports=[helper.make_opsetid("", 13)])
        so = ort.SessionOptions()
        so.log_severity_level = 3
        sess = ort.InferenceSession(
            model.SerializeToString(), so, providers=["CUDAExecutionProvider"]
        )
        if "CUDAExecutionProvider" not in sess.get_providers():
            raise RuntimeError("CUDA EP not active in the probe session")
        sess.run(None, {"x": _np.zeros(1, _np.float32)})
        return "cuda"
    except ImportError:
        # can't build a probe graph without onnx — trust the provider list
        logger.info("RTMPose: onnx unavailable to verify CUDA EP; trusting provider list")
        return "cuda"
    except Exception as exc:  # noqa: BLE001
        logger.warning("RTMPose: CUDA EP present but unusable (%s) — running on CPU", exc)
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
        # last picked hand's centroid (pixels) — used to stay locked on one hand
        self._prev_center = None
        self._misses = 0
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
            self._misses += 1
            if self._misses > 15:
                self._prev_center = None  # hand left the frame — stop tracking it
            return None

        mean = scores.mean(axis=1)
        best = int(np.argmax(mean))
        if len(keypoints) > 1:
            if not self._warned_multi:
                logger.warning(
                    "RTMPose sees %d hands on %s; locking onto one by proximity "
                    "(RTMPose has no left/right label or tracker)",
                    len(keypoints), frame.device_id,
                )
                self._warned_multi = True
            # Stay on the hand nearest last frame's pick, as long as it's not a
            # clearly worse detection than the top-scoring one.
            if self._prev_center is not None:
                centers = keypoints.mean(axis=1)  # (N, 2)
                d = np.linalg.norm(centers - self._prev_center, axis=1)
                near = int(np.argmin(d))
                if mean[near] >= 0.7 * mean[best]:
                    best = near

        if float(mean[best]) < self._min_conf:
            self._misses += 1
            if self._misses > 15:
                self._prev_center = None
            return None

        kp = keypoints[best]
        self._prev_center = kp.mean(axis=0)
        self._misses = 0
        points = {
            LM(i): np.asarray(kp[i], dtype=np.float32) for i in range(HAND_LM_COUNT)
        }
        return HandDetection(
            points=points,
            lm_z_rel=np.zeros(HAND_LM_COUNT, dtype=np.float32),
            confidence=float(mean[best]),
            device_id=frame.device_id,
            timestamp_us=frame.timestamp_us,
            lm_score=np.asarray(scores[best], dtype=np.float32),  # SimCC per-keypoint
        )

    def close(self) -> None:
        self._hand = None
