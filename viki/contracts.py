"""
viki.contracts
--------------
The single shared module between pipeline stages: every DTO, enum, Protocol,
and file-artifact schema constant lives here.

Rule of the refactor: a stage imports from ``viki.contracts`` and from the
public ``__init__`` of a neighbouring stage — never from another stage's
internal modules. This module depends only on numpy/cv2; nothing from
``viki.*`` may be imported here.

Artifact chain (see ``Episode``)::

    episodes/<id>/raw/      raw synced RGB-D  (cameras.record)
        -> rec.npz          per-camera hand landmark trajectories  (perception.extract)
        -> cln.npz          fused + smoothed wrist trajectory + EE pose  (prepare.run)
        -> plan.h5          synthesised robot joint trajectory  (retarget.run)
        -> replay.h5        proprioception attained on hardware  (replay.run)  [stub]
    datasets/<name>/        LeRobot dataset  (export.run)  [stub]
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from pathlib import Path
from typing import Literal, Optional, Protocol, Tuple, runtime_checkable

import numpy as np

# ─────────────────────────────── aliases ────────────────────────────────

Hand = Literal["left", "right"]
Outcome = Literal["good", "bad", "unrated"]
Verdict = Literal["pass", "reject", "dry-run", "unverified"]


# ──────────────────────────── hand landmarks ────────────────────────────


class LM(IntEnum):
    """
    MediaPipe Hands landmark indices, 0..20.

    ViKi tracks a single hand only; the arm landmarks that older revisions
    reserved (ELBOW/SHOULDER) are gone. Use ``HAND_LM_COUNT`` for the count —
    it is deliberately *not* an enum member so ``list(LM)`` stays clean.
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


HAND_LM_COUNT = 21


# ──────────────────────────── camera / capture ──────────────────────────


@dataclass
class CameraIntrinsics:
    """SDK-reported pinhole intrinsics for one colour or depth stream."""

    fx: float
    fy: float
    cx: float
    cy: float
    width: int
    height: int
    # Distortion coefficients (k1,k2,p1,p2[,k3]) — optional.
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5))

    @property
    def matrix(self) -> np.ndarray:
        """3x3 pinhole camera matrix K."""
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float64,
        )


@dataclass
class Frame:
    """
    A single frame from one camera.

    Fields
    ------
    color             : HxWx3 uint8 BGR (OpenCV convention)
    depth             : HxW   uint16 millimetres
    timestamp_us      : device monotonic clock, microseconds
    device_id         : serial or ``kinect_<n>``
    aligned_depth     : HxW uint16 SDK-aligned depth (optional)
    host_timestamp_us : host clock stamped by the worker on arrival; used for
                        cross-camera sync
    color_intrinsics / depth_intrinsics : CameraIntrinsics | None
    """

    color: np.ndarray
    depth: np.ndarray
    timestamp_us: int
    device_id: str
    aligned_depth: Optional[np.ndarray] = None
    host_timestamp_us: int = 0
    color_intrinsics: Optional[CameraIntrinsics] = None
    depth_intrinsics: Optional[CameraIntrinsics] = None

    def has_depth(self) -> bool:
        return self.depth is not None and self.depth.size > 0


@dataclass
class SyncedFrameGroup:
    """
    One frame per camera, aligned to a common host-clock tick.

    ``offsets_us[device_id] = frame.host_timestamp_us - sync_timestamp_us``
    (negative = frame arrived before the tick).
    """

    frames: dict
    sync_timestamp_us: int
    offsets_us: dict

    @property
    def device_ids(self) -> list:
        return list(self.frames.keys())

    def has_depth(self) -> bool:
        return all(f.has_depth() for f in self.frames.values())


@runtime_checkable
class DepthProjector(Protocol):
    """
    Minimal slice of a camera backend that ``perception`` needs for lifting:
    map a colour pixel to the depth image plane at range ``z``.
    """

    def project_color_to_depth(
        self, u: float, v: float, z: float
    ) -> tuple[float, float] | None: ...


# ──────────────────────────── calibration ───────────────────────────────


@dataclass
class BoardParameters:
    """Physical parameters of a plain chessboard."""

    board_size: Tuple[int, int]  # internal corners (cols, rows)
    square_size: float  # metres


@dataclass
class ArucoBoardParameters(BoardParameters):
    """Physical parameters of a ChArUco board."""

    marker_size: float
    aruco_dict: int  # cv2.aruco predefined dictionary id


@dataclass
class CalibrationSample:
    """One accepted board observation for one camera."""

    frame: "Frame"
    corners: np.ndarray
    resolution: Tuple[int, int]  # (width, height)
    board_params: BoardParameters


@dataclass
class ArucoCalibrationSample(CalibrationSample):
    """ChArUco sample: also carries detected chessboard-corner ids."""

    c_ids: np.ndarray


@dataclass
class CalibrationIntrinsics:
    """Solved intrinsics for one camera (calibration result / stored form)."""

    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5))

    @property
    def camera_matrix(self) -> np.ndarray:
        return np.array(
            [[self.fx, 0.0, self.cx], [0.0, self.fy, self.cy], [0.0, 0.0, 1.0]],
            dtype=np.float32,
        )


@dataclass
class CalibrationExtrinsics:
    """
    Pose of a ChArUco board relative to one camera (Rodrigues ``rvec`` +
    ``tvec``). ``transform_matrix`` is the 4x4 camera→world (board) transform
    consumed by ``perception.lift``.
    """

    rvec: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tvec: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def rotation_matrix(self) -> np.ndarray:
        import cv2

        R, _ = cv2.Rodrigues(self.rvec)
        return np.array(R)

    @property
    def transform_matrix(self) -> np.ndarray:
        """4x4 homogeneous transform from the camera frame to the board frame."""
        R = self.rotation_matrix
        R_inv = R.T
        t = self.tvec.flatten()
        T = np.eye(4)
        T[:3, :3] = R_inv
        T[:3, 3] = -R_inv @ t
        return np.array(T)


# ──────────────────────────── perception DTOs ───────────────────────────


@dataclass
class PreparedFrame:
    """
    A camera frame ready for a pose backend.

    rgb          : (H, W, 3) uint8, RGB
    depth_m      : (H, W) float32, metres (0 → NaN)
    depth_K      : (3, 3) depth intrinsic matrix, or None
    base_depth_m : optional static-background depth (metres) for scene subtraction
    """

    rgb: np.ndarray
    depth_m: np.ndarray
    depth_K: Optional[np.ndarray]
    device_id: str
    timestamp_us: int
    base_depth_m: Optional[np.ndarray] = None


@dataclass
class HandDetection:
    """
    Pixel-space output of a pose backend for one camera, one hand.

    points   : {LM: (u, v)} subpixel pixel coordinates (NaN where missing)
    lm_z_rel : (21,) float32 backend-relative z (NOT metric); wrist-relative
    confidence : overall detection score in [0, 1]
    """

    points: dict
    lm_z_rel: np.ndarray
    confidence: float
    device_id: str
    timestamp_us: int


@dataclass
class Landmarks3D:
    """Per-landmark 3-D positions in one camera's frame (metres)."""

    points: dict  # {LM: (3,) float32}
    device_id: str
    timestamp_us: int
    # Per-landmark fusion weight (paper §3.5, eq. 2). Stub: detector visibility
    # only — the range and incidence factors are not computed yet.
    weights: Optional[dict] = None

    def as_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "points": {int(k): v.tolist() for k, v in self.points.items()},
            "timestamp_us": self.timestamp_us,
        }


@dataclass
class EndEffectorPose:
    """
    World-frame wrist pose.

    position     : (3,) float32 — WRIST world XYZ (m)
    R_world_palm : (3, 3) float32 — palm frame → world. Built from the MCP
                   knuckle spread (INDEX→PINKY), not the thumb.
    rpy_deg      : (3,) float32 — roll/pitch/yaw, extrinsic XYZ
    valid        : all required landmarks present and the palm frame resolved
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
    """Per-camera 3-D hand skeleton in world coordinates."""

    device_id: str
    points: dict  # {LM: (3,) world XYZ, metres}
    timestamp_us: int
    end_effector: Optional[EndEffectorPose] = None

    def as_dict(self) -> dict:
        return {
            "device_id": self.device_id,
            "points": {int(k): v.tolist() for k, v in self.points.items()},
            "timestamp_us": self.timestamp_us,
            "end_effector": (
                self.end_effector.as_dict() if self.end_effector is not None else None
            ),
        }


@dataclass
class DepthDebug:
    """Per-camera, per-frame depth diagnostics kept alongside a recording."""

    device_id: str
    depth_valid_fraction: float
    depth_median_m: float
    depth_mean_m: float
    hand_detected: bool
    wrist_depth_m: float


@dataclass
class PipelineResult:
    """Result of one ``perception.extract`` step over a synced frame group."""

    frames: list  # list[SkeletonFrame], one per camera that saw a hand
    detections: dict  # {device_id: HandDetection | None}
    debug_depth_marks: Optional[dict] = None
    depth_debug: Optional[dict] = None


# ──────────────────────────── gripper ───────────────────────────────────


@dataclass(frozen=True)
class GripperState:
    """
    Estimated gripper state for one frame.

    closed     : True = closed / grasping
    width      : 0..1 normalised opening (for continuous grippers; binary sets 0/1)
    confidence : 0..1
    """

    closed: bool
    width: float
    confidence: float


# ──────────────────────────── labelling ─────────────────────────────────


@dataclass
class Segment:
    """A frame range within an episode tagged with a phase label."""

    start: int
    end: int
    label: str  # approach / grasp / transport / release / ...


@dataclass
class EpisodeLabels:
    """
    Human annotation for one episode, persisted under ``meta.json["labels"]``.
    ``task`` must be non-empty before the episode can be exported.
    """

    task: str = ""
    hand: Hand = "right"
    segments: list = field(default_factory=list)  # list[Segment]
    outcome: Outcome = "unrated"
    notes: str = ""


# ──────────────────────────── episode paths ─────────────────────────────


@dataclass(frozen=True)
class Episode:
    """
    Filesystem view of one demonstration. Every stage reads this directory and
    writes exactly one more artifact into it.
    """

    root: Path

    @property
    def id(self) -> str:
        return self.root.name

    @property
    def meta_path(self) -> Path:
        return self.root / "meta.json"

    @property
    def status_path(self) -> Path:
        return self.root / "status.json"

    @property
    def raw_dir(self) -> Path:
        return self.root / "raw"

    @property
    def rec_npz(self) -> Path:
        return self.root / "rec.npz"

    @property
    def cln_npz(self) -> Path:
        return self.root / "cln.npz"

    @property
    def plan_h5(self) -> Path:
        return self.root / "plan.h5"

    @property
    def replay_h5(self) -> Path:
        return self.root / "replay.h5"


# ──────────────────────── artifact schema keys ─────────────────────────
# The exact array keys each stage writes. Writers and readers assert against
# these so a schema drift fails loudly instead of silently.

REC_KEYS: tuple[str, ...] = (
    "device_ids",
    "timestamps",
    "points",  # (N, 21, 3)
    "landmark_ids",  # (21,)
    "confidence",  # (N, 21)  — stub: detector visibility only
)

CLN_KEYS: tuple[str, ...] = (
    "timestamps",  # (T,)
    "positions",  # (T, 3)
    "rotations",  # (T, 3, 3)
    "valid",  # (T,)
    "omega",  # (T,)  — per-frame confidence weight ω_t  (stub)
    "gripper",  # (T,)  bool
    "coordinate_frame",
    "raw_points",
    "smoothed_points",
    "landmark_ids",
)
CLN_OPTIONAL_KEYS: tuple[str, ...] = (
    "T_world_obj",  # (T, 4, 4)  — object pose track  (stub: absent)
    "T_obj_hand",  # (T, 4, 4)  — object-relative form  (stub: absent)
)

REPLAY_KEYS: tuple[str, ...] = (
    "q_attained",  # (T, nq)
    "gripper_attained",  # (T,)
    "controller_residual",  # (T,)  NaN under dry-run
    "verdict",
    "rejection_cause",
    "resolve_attempts",
    "robot",
    "dt",
)
