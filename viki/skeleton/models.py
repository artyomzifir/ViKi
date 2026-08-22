"""
viki.skeleton.models
--------------------
Dataclasses for data flowing between skeleton pipeline stages.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import IntEnum
from typing import Optional

import numpy as np


#
# from Stage 1 to 2: synced frames to input of hand detector
# Used other class to encapsulate from capture sync logic
#
@dataclass
class PreparedFrame:
    """
    A single camera frame ready for model inference.

    Produced by camera_prep from a raw Frame:
      - color is converted BGR to RGB (no undistort)
      - depth is float32 metres with 0 replaced by nan
      - depth_K is the real depth-camera intrinsic matrix (for 3D lifting)
      - base_depth_m is the optional static background depth (metres) used to
        subtract the scene so the tracked hand stands out.
    """

    rgb: np.ndarray  # (H, W, 3) uint8, RGB
    depth_m: np.ndarray  # (H, W)    float32, metres
    depth_K: Optional[np.ndarray]  # (3, 3)    depth intrinsic matrix
    device_id: str
    timestamp_us: int
    base_depth_m: Optional[np.ndarray] = None  # (H, W) float32, background depth (m)


#
# from Stage 2 to 3: hand detector output to geometry input
#
@dataclass
class HandDetection:
    """
    Raw MediaPipe output for one camera, pixel-space.

    Parameters
    ----------

    px[i] : (u, v) pixel coordinates of landmark i (float, subpixel).
    lm_z_rel[i] : MediaPipe's relative z for landmark i.
                  NOT metric depth. Relative to wrist (landmark 0). Approximated by mediapipe as ratio of landmarks
                  Use only as fallback when depth_m is nan at that pixel.
    confidence : overall hand detection score from MediaPipe [0..1].

    None is returned by the detector when no hand is found; this dataclass
    is only instantiated on a successful detection.
    """

    points: dict[LM, np.ndarray]
    lm_z_rel: np.ndarray  # float32, MediaPipe relative z
    confidence: float
    device_id: str
    timestamp_us: int


@dataclass
class Landmarks3D:
    points: dict[LM, np.ndarray]
    device_id: str
    timestamp_us: int

    def as_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "points": {index.value: vec.tolist() for index, vec in self.points.items()},
            "timestamp_us": self.timestamp_us,
        }


@dataclass
class EndEffectorPose:
    """
    World-frame pose of the end-effector (wrist).

    Fields
    ------
    position     : (3,) float32 — WRIST world XYZ in metres.
    R_world_palm : (3, 3) float32 — rotation from palm frame to world.
                   Palm frame:
                       x_palm = normalise(MIDDLE_MCP - WRIST)
                       z_palm = normalise((MIDDLE_MCP - WRIST) × (THUMB_CMC - WRIST))
                       y_palm = z_palm × x_palm
    rpy_deg      : (3,) float32 — roll/pitch/yaw in degrees, extrinsic XYZ
                   i.e. R = Rz(yaw) · Ry(pitch) · Rx(roll).
    valid        : True when every required landmark was present and the
                   palm frame could be resolved.
    timestamp_us : same as the containing SkeletonFrame.
    """

    position: np.ndarray
    R_world_palm: np.ndarray
    rpy_deg: np.ndarray
    valid: bool
    timestamp_us: int

    def as_dict(self) -> dict:
        return {
            "position": self.position.tolist(),
            "R_world_palm": self.R_world_palm.tolist(),
            "rpy_deg": self.rpy_deg.tolist(),
            "valid": self.valid,
            "timestamp_us": self.timestamp_us,
        }


@dataclass
class SkeletonFrame:
    """
    Per‑camera 3‑D hand skeleton in world coordinates.

    The live skeleton pipeline no longer fuses cameras at capture time; it
    emits one ``SkeletonFrame`` per camera that detected a hand. Fusion of the
    multiple per‑camera trajectories is deferred to the smooth/optimisation
    stage. ``device_id`` identifies the source camera so the frontend can draw
    each hand in a distinct colour.

    Attributes
    ----------
    device_id : str
        Identifier of the camera that produced this frame.
    points : dict[LM, np.ndarray]
        Mapping from landmark enum to world‑frame (X, Y, Z) in metres.
    timestamp_us : int
        Sync timestamp of the frame.
    end_effector : Optional[EndEffectorPose]
        World‑frame wrist pose, if computable.
    """
    device_id: str
    points: dict[LM, np.ndarray]
    timestamp_us: int
    end_effector: Optional[EndEffectorPose] = None

    def as_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "points": {index.value: vec.tolist() for index, vec in self.points.items()},
            "timestamp_us": self.timestamp_us,
            "end_effector": (
                self.end_effector.as_dict() if self.end_effector is not None else None
            ),
        }


@dataclass
class DepthDebug:
    """
    Per‑camera, per‑frame depth diagnostics captured for recording/debugging.

    Records what the depth camera was doing this frame: the fraction of valid
    depth pixels, the median/mean depth across the frame, and — for the camera
    that was actually lifted — the depth measured at the wrist landmark. The
    wrist depth is the single value that drives hand‑position estimation, so its
    trajectory is the most direct signal when depth starts mis‑behaving.
    """

    device_id: str
    depth_valid_fraction: float  # 0..1 share of in‑range depth pixels
    depth_median_m: float  # median valid depth (m); NaN if none
    depth_mean_m: float  # mean valid depth (m); NaN if none
    hand_detected: bool
    wrist_depth_m: float  # depth at wrist (m); NaN if no hand / no depth


@dataclass
class PipelineResult:
    """
    Result of a full pipeline run (no capture‑time fusion).

    Attributes
    ----------
    frames : list[SkeletonFrame]
        One ``SkeletonFrame`` per camera that detected a hand (world‑frame,
        each tagged with its ``device_id``). The frontend draws each in its own
        colour; the smooth stage fuses these trajectories later.
    detections : dict[str, HandDetection | None]
        Per‑camera 2D detections (None if no hand found).
    debug_depth_marks : dict[str, dict[LM, np.ndarray]] | None
        Per‑camera, per‑landmark 3D points obtained purely from the depth
        camera (deprojected at each landmark's measured depth, transformed to
        world frame). Only populated when depth debugging is enabled. Passed
        through *as‑is* (never fused) so the frontend can render every camera's
        raw depth estimate.
    depth_debug : dict[str, DepthDebug] | None
        Per‑camera depth diagnostics for this frame group (every camera present
        in the group, regardless of whether it was lifted). Used by the recorder
        to capture what the depth cameras were doing during a recording.
    """
    frames: list[SkeletonFrame]
    detections: dict[str, HandDetection | None]  # Per-camera 2D landmarks
    debug_depth_marks: Optional[dict[str, dict[LM, np.ndarray]]] = None
    depth_debug: Optional[dict[str, "DepthDebug"]] = None


# MediaPipe Hands landmark indices
class LM(IntEnum):
    """
    Landmark indices for MediaPipe Hands (21 hand landmarks) plus two arm landmarks.

    Hand landmarks: 0 (WRIST) to 20 (PINKY_TIP).
    Arm landmarks (not detected): 21 (ELBOW), 22 (SHOULDER) – kept for schema compatibility.
    """
    WRIST = 0
    THUMB_CMC = 1
    THUMB_MCP = 2
    THUMB_IP = 3
    THUMB_TIP = 4
    INDEX_MCP = 5
    INDEX_PIP = 6
    INDEX_DIP = 7
    INDEX_TIP = 8
    MIDDLE_MCP = 9
    MIDDLE_PIP = 10
    MIDDLE_DIP = 11
    MIDDLE_TIP = 12
    RING_MCP = 13
    RING_PIP = 14
    RING_DIP = 15
    RING_TIP = 16
    PINKY_MCP = 17
    PINKY_PIP = 18
    PINKY_DIP = 19
    PINKY_TIP = 20

    # Arm landmarks — reserved for schema compatibility; not produced by the
    # hand detector (MediaPipeArm is disabled).
    ELBOW = 21
    SHOULDER = 22

    N = 23
