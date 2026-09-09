"""
viki.perception.backends.mediapipe
----------------------------------
Hand landmarks from MediaPipe Tasks ``HandLandmarker`` (21 keypoints).

Self-contained: the small amount of MediaPipe Tasks plumbing that used to live
in ``detectors/mediapipe_base.py`` is inlined here. Only IMAGE and VIDEO running
modes are supported — ViKi runs offline over recorded frames, there is no live
stream.
"""

from __future__ import annotations

import logging
import urllib.request
from pathlib import Path

import numpy as np

from viki.contracts import HAND_LM_COUNT, Hand, HandDetection, LM, PreparedFrame
from viki.perception.backends.base import HandPoseBackend

logger = logging.getLogger(__name__)

_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)
_MIN_GRAPH_CONFIDENCE = 1e-6


def _graph_confidence(value: float) -> float:
    """Map the public zero/no-gate value to MediaPipe's valid open interval."""
    return max(float(value), _MIN_GRAPH_CONFIDENCE)


def _ensure_model(models_dir: str) -> str:
    """Download the .task file once and cache it under ``models_dir``."""
    path = Path(models_dir) / "hand_landmarker.task"
    if not path.exists():
        path.parent.mkdir(parents=True, exist_ok=True)
        logger.info("downloading MediaPipe hand_landmarker model → %s", path)
        urllib.request.urlretrieve(_MODEL_URL, path)
    return str(path)


class MediaPipeHandBackend(HandPoseBackend):
    """MediaPipe HandLandmarker wrapper. One instance per camera stream."""

    name = "mediapipe"

    def __init__(
        self,
        *,
        mode: str = "video",
        models_dir: str = "models",
        model_path: str | None = None,
        model_entry: dict | None = None,  # registry row; MediaPipe has one model
        min_confidence: float = 0.5,
        tracking_confidence: float | None = None,
        **_ignored,
    ) -> None:
        if mode not in ("image", "video"):
            raise ValueError("MediaPipe backend supports 'image' or 'video' only")
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision

        self._mode = mode
        self._last_ts_ms = -1
        running_mode = (
            vision.RunningMode.VIDEO if mode == "video" else vision.RunningMode.IMAGE
        )
        # MediaPipe's native HandAssociationCalculator aborts the entire
        # process when min_tracking_confidence is exactly zero. Keep a tiny
        # positive implementation floor while exposing 0 as "no score gate".
        detection_confidence = _graph_confidence(min_confidence)
        tracking_confidence = _graph_confidence(
            min_confidence if tracking_confidence is None else tracking_confidence
        )
        opts = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=model_path or _ensure_model(models_dir)
            ),
            running_mode=running_mode,
            num_hands=1,
            min_hand_detection_confidence=detection_confidence,
            min_hand_presence_confidence=detection_confidence,
            min_tracking_confidence=tracking_confidence,
        )
        self._task = vision.HandLandmarker.create_from_options(opts)

    def detect(self, frame: PreparedFrame, hand: Hand) -> HandDetection | None:
        import mediapipe as mp

        image = mp.Image(
            image_format=mp.ImageFormat.SRGB,
            data=np.ascontiguousarray(frame.rgb),
        )
        if self._mode == "image":
            raw = self._task.detect(image)
        else:
            ts_ms = max(frame.timestamp_us // 1000, self._last_ts_ms + 1)
            self._last_ts_ms = ts_ms
            raw = self._task.detect_for_video(image, ts_ms)

        if raw is None or not raw.hand_landmarks:
            return None
        return self._extract(raw, frame)

    def close(self) -> None:
        task = getattr(self, "_task", None)
        if task is not None:
            task.close()
            self._task = None

    def _extract(self, raw, frame: PreparedFrame) -> HandDetection | None:
        # Handedness is deliberately not consulted. MediaPipe infers left/right
        # from the hand's appearance in one 2-D image, so the same physical hand
        # is labelled differently depending on which side a camera sees it from:
        # measured on cup_grab, kinect_0 reported "Right" in 98.4% of frames
        # while kinect_1 reported "Left" in 11.8% of them with a median score of
        # 0.967 — confidently, and for the same right hand. Rejecting on that
        # label threw away the whole detection (all 21 landmarks) for a frame
        # whose geometry was fine.
        #
        # ViKi records one hand per episode and the operator declares which one
        # (`meta["hand"]`, the Record tab's Hand selector), so the anatomical
        # side is a known recording parameter, not something to re-derive per
        # camera per frame. The graph runs with num_hands=1, so whatever came
        # back is the hand we asked for.
        if not raw.hand_landmarks:
            return None
        # The handedness score measures certainty about a label we no longer
        # use; it is not landmark quality and must not masquerade as one.
        # Triangulation geometry owns rejection from here on.
        match_score = 1.0

        h, w = frame.rgb.shape[:2]
        lms = raw.hand_landmarks[0]
        points: dict[LM, np.ndarray] = {}
        z = np.zeros(HAND_LM_COUNT, dtype=np.float32)
        for i in range(HAND_LM_COUNT):
            lm = lms[i]
            points[LM(i)] = np.array([lm.x * w, lm.y * h], dtype=np.float32)
            z[i] = lm.z

        return HandDetection(
            points=points,
            lm_z_rel=z,
            confidence=float(match_score),
            device_id=frame.device_id,
            timestamp_us=frame.timestamp_us,
        )
