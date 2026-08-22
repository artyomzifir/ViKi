"""
viki.skeleton.detectors.hand_pose
---------------------------------
Partial detector that emits 21 hand keypoints from MediaPipe HandLandmarker.
Writes into global slots 0..20 (WRIST..PINKY_TIP).

MediaPipe Hand landmark indices reference:
https://developers.google.com/edge/mediapipe/solutions/vision/hand_landmarker
"""

from __future__ import annotations

from typing import Literal, Optional

import numpy as np

from viki.skeleton.detectors.base import (
    PartialDetection2D,
    PartialLandmarkDetector,
)
from viki.skeleton.detectors.mediapipe_base import (
    MODELS_DIR_DEFAULT,
    MediaPipeTaskRunner,
    ensure_model,
)
from viki.skeleton.models import PreparedFrame

_HAND_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "hand_landmarker/hand_landmarker/float16/1/hand_landmarker.task"
)

_LABEL_RIGHT = "Right"
_LABEL_LEFT = "Left"


class MediaPipeHand(PartialLandmarkDetector):
    """
    Partial detector for one hand using MediaPipe Hand Landmarker.

    Emits 21 hand landmarks (slots 0..20). This detector has higher priority
    than the arm detector (10), so it will overwrite the wrist slot if both are
    present.

    Attributes
    ----------
    name : str
        Detector identifier ("hand").
    indices : tuple[int, ...]
        Global slots written: (0..20).
    priority : int
        Priority (10).
    """

    name = "hand"
    indices = tuple(range(21))
    priority = 10

    def __init__(
        self,
        hand: Literal["right", "left"] = "right",
        mode: Literal["image", "video", "live"] = "image",
        hand_model: Optional[str] = None,
        models_dir: str = MODELS_DIR_DEFAULT,
        min_hand_confidence: float = 0.5,
    ) -> None:
        """
        Parameters
        ----------
        hand : Literal["right", "left"], default="right"
            Which hand to track.
        mode : Literal["image", "video", "live"], default="image"
            MediaPipe running mode.
        hand_model : Optional[str], default=None
            Explicit path to a hand_landmarker.task file. If None, the model is
            auto‑downloaded into `models_dir`.
        models_dir : str, default=MODELS_DIR_DEFAULT
            Local cache directory for downloaded models.
        min_hand_confidence : float, default=0.5
            Threshold for detection, presence, and tracking confidence.
        """
        super().__init__()

        self._hand = hand
        self._target_label = _LABEL_RIGHT if hand == "right" else _LABEL_LEFT

        model_path = hand_model or ensure_model(
            "hand_landmarker.task",
            _HAND_URL,
            models_dir,
        )

        # Detector-specific config stays in the closure; runner only sees
        # shared infrastructure pieces.
        def _factory(base_options, running_mode, result_callback):
            from mediapipe.tasks.python import vision

            opts = vision.HandLandmarkerOptions(
                base_options=base_options,
                running_mode=running_mode,
                num_hands=1,
                min_hand_detection_confidence=min_hand_confidence,
                min_hand_presence_confidence=min_hand_confidence,
                min_tracking_confidence=min_hand_confidence,
                **({"result_callback": result_callback} if result_callback else {}),
            )
            return vision.HandLandmarker.create_from_options(opts)

        self._runner = MediaPipeTaskRunner(_factory, model_path, mode)

    def detect(self, frame: PreparedFrame) -> Optional[PartialDetection2D]:
        """
        Run the detector on a prepared frame.

        Parameters
        ----------
        frame : PreparedFrame
            Prepared camera frame (RGB + depth + intrinsics).

        Returns
        -------
        Optional[PartialDetection2D]
            Detection over slots 0..20 if a hand of the requested handedness
            is detected; otherwise None.
        """
        raw = self._runner.submit(frame.rgb, frame.timestamp_us)
        if raw is None or not raw.hand_landmarks:
            return None
        return self._extract(raw, frame)

    def close(self) -> None:
        """Release the underlying MediaPipe task resources."""
        self._runner.close()

    def _extract(self, raw, frame: PreparedFrame) -> Optional[PartialDetection2D]:
        """
        Extract pixel coordinates, z, and confidence from the raw MediaPipe result.

        Parameters
        ----------
        raw : mediapipe.tasks.python.vision.HandLandmarkerResult
            Raw result from the runner.
        frame : PreparedFrame
            The source frame (used for image size and metadata).

        Returns
        -------
        Optional[PartialDetection2D]
            Detection object with shape (21, 2) px, (21,) z, (21,) confidence,
            or None if the requested handedness is not present.
        """
        # Iterate handedness to find the hand whose label matches our target.
        match_idx: Optional[int] = None
        match_score: float = 0.0
        for i, handedness_list in enumerate(raw.handedness):
            label = handedness_list[0].category_name
            if label == self._target_label:
                match_idx = i
                match_score = float(handedness_list[0].score)
                break
        if match_idx is None:
            return None

        h, w = frame.rgb.shape[:2]
        lms = raw.hand_landmarks[match_idx]  # 21 NormalizedLandmark

        n = len(self.indices)
        px = np.zeros((n, 2), dtype=np.float32)
        z = np.zeros(n, dtype=np.float32)
        conf = np.full(n, match_score, dtype=np.float32)
        
        for k, idx in enumerate(self.indices):
            lm = lms[idx]
            px[k] = (lm.x * w, lm.y * h)
            z[k] = lm.z

        return PartialDetection2D(
            indices=self.indices,
            px=px,
            lm_z_rel=z,
            per_index_confidence=conf,
            device_id=frame.device_id,
            timestamp_us=frame.timestamp_us,
        )
