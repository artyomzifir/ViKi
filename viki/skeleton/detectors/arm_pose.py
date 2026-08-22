"""
viki.skeleton.detectors.arm_pose
--------------------------------
Partial detector that emits (wrist, elbow, shoulder) for one arm using
MediaPipe Pose Landmarker. Writes into global slots (0, 21, 22).

MediaPipe Pose landmark indices reference:
https://developers.google.com/edge/mediapipe/solutions/vision/pose_landmarker
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

_POSE_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_full/float16/1/pose_landmarker_full.task"
)

_RIGHT = (16, 14, 12)  # WRIST, ELBOW, SHOULDER
_LEFT = (15, 13, 11)


class MediaPipeArm(PartialLandmarkDetector):
    """
    Partial detector for one arm using MediaPipe Pose Landmarker.

    Emits landmarks for WRIST (slot 0), ELBOW (21), and SHOULDER (22).
    The wrist slot is also used by the hand detector; this detector has
    lower priority (0) so it will be overwritten by the hand detector if
    both are present, but it provides a fallback when no hand is detected.

    Attributes
    ----------
    name : str
        Detector identifier ("arm_pose").
    indices : tuple[int, ...]
        Global slots written: (0, 21, 22).
    priority : int
        Priority (0) – lower means higher priority.
    """

    name = "arm_pose"
    indices = (0, 21, 22)
    priority = 0

    def __init__(
        self,
        hand: Literal["right", "left"] = "right",
        mode: Literal["image", "video", "live"] = "image",
        pose_model: Optional[str] = None,
        models_dir: str = MODELS_DIR_DEFAULT,
        min_pose_confidence: float = 0.3,
    ) -> None:
        """
        parameters
        ----------
        hand                : "right" or "left" arm to track.
        mode                : MediaPipe running mode ("image" / "video" / "live").
        pose_model          : explicit path to a pose_landmarker.task; auto-
                              downloaded into `models_dir` when None.
        models_dir          : local cache directory for downloaded models.
        min_pose_confidence : threshold reused for detection, presence,
                              and tracking confidence.
        """
        super().__init__()

        self._hand = hand
        self._pose_indices = _RIGHT if hand == "right" else _LEFT

        model_path = pose_model or ensure_model(
            "pose_landmarker.task",
            _POSE_URL,
            models_dir,
        )

        # Closure captures detector-specific config (thresholds). Runner
        # supplies the shared infrastructure pieces as call args.
        def _factory(base_options, running_mode, result_callback):
            from mediapipe.tasks.python import vision

            opts = vision.PoseLandmarkerOptions(
                base_options=base_options,
                running_mode=running_mode,
                min_pose_detection_confidence=min_pose_confidence,
                min_pose_presence_confidence=min_pose_confidence,
                min_tracking_confidence=min_pose_confidence,
                **({"result_callback": result_callback} if result_callback else {}),
            )
            return vision.PoseLandmarker.create_from_options(opts)

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
            Detection over slots (0, 21, 22) if pose is detected and the requested
            hand arm is visible; otherwise None.
        """
        raw = self._runner.submit(frame.rgb, frame.timestamp_us)
        if raw is None or not raw.pose_landmarks:
            return None
        return self._extract(raw, frame)

    def close(self) -> None:
        """Release the underlying MediaPipe task resources."""
        self._runner.close()

    def _extract(self, raw, frame: PreparedFrame) -> PartialDetection2D:
        """
        Extract pixel coordinates, z, and confidence from the raw MediaPipe result.

        Parameters
        ----------
        raw : mediapipe.tasks.python.vision.PoseLandmarkerResult
            Raw result from the runner.
        frame : PreparedFrame
            The source frame (used for image size and metadata).

        Returns
        -------
        PartialDetection2D
            Detection object with shape (3, 2) px, (3,) z, (3,) confidence.
        """
        h, w = frame.rgb.shape[:2]
        lms = raw.pose_landmarks[0]  # take only one person

        px = np.zeros((3, 2), dtype=np.float32)
        z = np.zeros(3, dtype=np.float32)
        conf = np.zeros(3, dtype=np.float32)
        for k, idx in enumerate(self._pose_indices):
            lm = lms[idx]
            px[k] = (lm.x * w, lm.y * h)
            z[k] = lm.z
            # MediaPipe pose landmarks expose `presence` (in-frame probability).
            conf[k] = float(getattr(lm, "presence", 1.0))

        return PartialDetection2D(
            indices=self.indices,
            px=px,
            lm_z_rel=z,
            per_index_confidence=conf,
            device_id=frame.device_id,
            timestamp_us=frame.timestamp_us,
        )
