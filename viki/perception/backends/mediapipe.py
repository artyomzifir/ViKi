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
_LABEL = {"right": "Right", "left": "Left"}


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
        opts = vision.HandLandmarkerOptions(
            base_options=python.BaseOptions(
                model_asset_path=model_path or _ensure_model(models_dir)
            ),
            running_mode=running_mode,
            num_hands=1,
            min_hand_detection_confidence=min_confidence,
            min_hand_presence_confidence=min_confidence,
            min_tracking_confidence=min_confidence,
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
        return self._extract(raw, frame, hand)

    def close(self) -> None:
        task = getattr(self, "_task", None)
        if task is not None:
            task.close()
            self._task = None

    @staticmethod
    def _extract(raw, frame: PreparedFrame, hand: Hand) -> HandDetection | None:
        target = _LABEL[hand]
        match_idx = match_score = None
        for i, handedness in enumerate(raw.handedness):
            if handedness[0].category_name == target:
                match_idx = i
                match_score = float(handedness[0].score)
                break
        if match_idx is None:
            return None

        h, w = frame.rgb.shape[:2]
        lms = raw.hand_landmarks[match_idx]
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
