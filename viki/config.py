"""
viki.config
-----------
Centralised tunables for the ViKi capture server.
Loaded from data/user_configuration.json.

The module reads the JSON configuration file once at import and updates
the global namespace with all keys. This allows constants to be imported
directly from the module (e.g., from viki.config import DEFAULT_FPS).
"""

import json
import os
import shutil
from typing import Any

DEFAULT_CONFIG_PATH = "data/default_configuration.json"
USER_CONFIG_PATH = "data/user_configuration.json"

# Duck variables which LSP can catch and use
EXTRINSICS_FILENAME: str
WORLD_ANCHOR_FILENAME: str  # live world-anchor (T_world_display); folded into a preset on save-as
VALIDATION_FILENAME: str    # live pre-record cloud-agreement report
ACTIVE_CALIBRATION: str  # name of the active preset under data/calibrations/, or ""
DEFAULT_FPS: int
DEFAULT_COLOR_WIDTH: int
DEFAULT_COLOR_HEIGHT: int
DEFAULT_DEPTH_MODE: str
JPEG_QUALITY: int
STREAM_IDLE_SLEEP: float
PLACEHOLDER_SIZE: list[int]
DEFAULT_SYNCHRONIZED_IMAGES_ONLY: bool
FRAME_BUFFER_SIZE: int
RECORD_DEPTH: bool
CLOUD_STRIDE: int  # keep every Nth depth pixel per axis when building the cloud
CLOUD_VOXEL_M: float  # voxel-downsample leaf size (metres); 0 disables
CLOUD_WORKSPACE_BBOX: list[float]  # world AABB [xmin,xmax,ymin,ymax,zmin,zmax]; empty = no crop
CLOUD_MAX_POINTS_PER_FRAME: int
CLOUD_BG_SUBTRACT: bool  # drop cloud points matching the calibrated empty-scene depth
CLOUD_BG_TOLERANCE_MM: float  # |depth - background| below this = static scene, dropped
PERCEPTION_TRACK_LM: list[int]  # hand-landmark indices to keep (others left NaN)
PERCEPTION_INTERP_MAX_GAP: int  # >0: leave interior gaps longer than N frames unfilled
PERCEPTION_CONF_ALPHA: float  # α in ω_t = (mean_i max_k w_i)^α  (paper §3.5 eq. 5)
PERCEPTION_HAND_FIT: bool  # run trajectory-level capsule hand fit at the end of prepare
PERCEPTION_SAVE_OBSERVATIONS: bool  # extract also writes raw/observations.npz (2-D obs for multi-view triangulation)
PERCEPTION_HAND_POSE_SOURCE: str  # landmarks | hand_fit; consumers select without overwriting cln pose
PERCEPTION_HAND_FIT_ROI_MARGIN_M: float  # adaptive capsule-union ROI padding (m)
PERCEPTION_HAND_FIT_FOREARM_CUT_M: float  # proximal offset of wrist cut plane (m)
PERCEPTION_HAND_FIT_VOXEL_M: float  # deterministic voxel sample size (m)
PERCEPTION_HAND_FIT_HUBER_M: float  # Huber δ (m) on the point→capsule data residual
PERCEPTION_HAND_FIT_W_DATA: float  # depth-cloud weight; multiplies the per-frame mean squared point→surface distance
PERCEPTION_HAND_FIT_W_VEL_TRANSLATION: float
PERCEPTION_HAND_FIT_W_VEL_ROTATION: float
PERCEPTION_HAND_FIT_W_VEL_JOINTS: float
PERCEPTION_HAND_FIT_W_ACC_TRANSLATION: float
PERCEPTION_HAND_FIT_W_ACC_ROTATION: float
PERCEPTION_HAND_FIT_W_ACC_JOINTS: float
PERCEPTION_HAND_FIT_W_PRIOR: float  # λ_prior — joint-limit barrier weight
PERCEPTION_HAND_FIT_W_POSTURE: float  # weak pull toward calibrated rest pose
PERCEPTION_HAND_FIT_W_LANDMARK: float  # confidence-weighted initial landmark anchor
PERCEPTION_HAND_FIT_LANDMARK_DECAY: float  # multiplier per outer ICP iteration
PERCEPTION_HAND_FIT_INSIDE_SCALE: float  # one-sided attenuation for points inside capsules
PERCEPTION_HAND_FIT_MIN_POINTS: int  # below this, frame has an empty data block
PERCEPTION_HAND_FIT_MAX_POINTS: int  # max deterministic voxel representatives per frame
PERCEPTION_HAND_FIT_MAX_NFEV: int
PERCEPTION_HAND_FIT_OUTER_ITERATIONS: int
PERCEPTION_HAND_FIT_WINDOW: int  # frames per sliding window; 0 = whole-episode batch
PERCEPTION_HAND_FIT_WINDOW_OVERLAP: int  # overlapping frames blended between windows
PERCEPTION_HAND_FIT_WORKERS: int  # window-solver threads; 0 = auto (min(4, cpu/2))
PERCEPTION_HAND_FIT_WARM_START_MAD_K: float  # wrist warm-start spike gate (robust MAD units)
PERCEPTION_HAND_FIT_DEADLINE_S: float  # wall-clock guard per fit_trajectory call; 0 = off
KINECT_SYNC: dict  # {"master": "kinect_0", "subordinates": [...], "subordinate_delay_us": 160}; {} = software sync only
SKELETON_RECS_DIR: str
SKELETON_SMOOTHED_DIR: str
SKELETON_COORDINATE_FRAME: str
SKELETON_SAVE_JSON_DEBUG: bool
HAND_TO_DETECT: str
DISCARD_OUTLIERS: bool
DISCARD_OUTLIERS_MAX_PORTION: float
POSITION_FROM_WRIST: bool
DEPTH_DEBUG: bool
CALIB_MODE: str
CALIB_BOARD_TYPE: str
CALIB_CHESS_BOARD_SIZE: list[int]
CALIB_CHESS_SQUARE_SIZE: float
CALIB_ARUCO_BOARD_SIZE: list[int]
CALIB_ARUCO_SQUARE_SIZE: float
CALIB_ARUCO_MARKER_SIZE: float
CALIB_ARUCO_DICT: int
CALIB_POSE_MIN_ANGLE_DEG: float       # reject a capture set whose board pose is within this angle …
CALIB_POSE_MIN_TRANSLATION_M: float   # … AND this translation of an already-collected set
CALIB_TILT_MIN_DEG: float             # a set counts as "tilted" when board-normal vs ref optical axis exceeds this
CALIB_MIN_SETS: int                   # Solve gate: minimum capture sets
CALIB_MIN_COVISIBLE_SETS: int         # Solve gate: sets seen by every active camera at once
CALIB_MIN_TILTED_SETS: int            # Solve gate: sets above CALIB_TILT_MIN_DEG
CALIB_MIN_FRAME_COVERAGE: float       # Solve gate: min fraction of a 4×4 image grid touched by corners, per camera
CALIB_VALIDATE_GREEN_NN_MM: float     # §6 verdict thresholds — green: pairwise NN median …
CALIB_VALIDATE_GREEN_ICP_TRANS_MM: float  # … ICP correction translation …
CALIB_VALIDATE_GREEN_ICP_ROT_DEG: float   # … ICP correction rotation
CALIB_VALIDATE_AMBER_NN_MM: float     # amber band (above ⇒ red): NN median …
CALIB_VALIDATE_AMBER_ICP_TRANS_MM: float
CALIB_VALIDATE_AMBER_ICP_ROT_DEG: float
RECORDING_DURATION: int
RECORDING_FPS: int
RETARGET_DEFAULT_ROBOT: str
RETARGET_LANDMARK_SG_WINDOW: int
RETARGET_LANDMARK_SG_POLYORDER: int
RETARGET_IK_POSITION_COST: float
RETARGET_IK_ORIENTATION_COST: float
RETARGET_IK_POSTURE_COST: float
RETARGET_IK_ACCEL_COST: float  # λ_a: in-solver acceleration regulariser weight (replaces joint SG)
RETARGET_IK_CONF_FLOOR: float  # lower clamp on ω_t when it scales the IK data term (0 disables the ω_t weighting)
RETARGET_TARGET_MODE: str
RETARGET_IK_SUBSTEPS: int
RETARGET_IK_SOLVER: str
RETARGET_APPROACH_SEC: float
RETARGET_JOINT_SG_WINDOW: int
RETARGET_JOINT_SG_POLYORDER: int
RETARGET_RECENTER_TO_NEUTRAL: bool
RETARGET_TRAJECTORY_SCALE: float
ROBOT_BASE_OFFSET: list[float]
TARGET_OFFSET: list[float]
RETARGET_BASE_ROTATION: list[list[float]]
RETARGET_BASE_TRANSLATION: list[float]
MODELS_DIR: str


def _load_config():
    """
    Load configuration from the user JSON file, or copy from default if missing.

    Returns
    -------
    dict
        The configuration dictionary. If neither file exists, returns an empty dict.
    """
    if not os.path.exists(USER_CONFIG_PATH):
        if os.path.exists(DEFAULT_CONFIG_PATH):
            shutil.copy(DEFAULT_CONFIG_PATH, USER_CONFIG_PATH)
        else:
            # Fallback if even default is missing (shouldn't happen with our setup)
            return {}

    with open(USER_CONFIG_PATH, "r") as f:
        return json.load(f)


_config = _load_config()

# Legacy access: `from viki.config import CONSTANT`. Kept until every stage takes
# an explicit `Config` argument; new code should use `viki.config.load()`.
globals().update(_config)

if "ROBOT_BASE_OFFSET" not in _config:
    globals()["ROBOT_BASE_OFFSET"] = [0.0, 0.0, 0.0]
if "TARGET_OFFSET" not in _config:
    globals()["TARGET_OFFSET"] = [0.0, 0.0, 0.0]


# ─────────────────────────── explicit Config object ────────────────────────────

from dataclasses import dataclass
from types import MappingProxyType
from typing import Any, Mapping


@dataclass(frozen=True)
class Config:
    """
    Immutable view over the merged configuration. Stages take this as an
    argument instead of importing module globals.

    Hot keys are plain attributes (via ``__getattr__`` over the frozen mapping);
    ``cfg.get("KEY", default)`` covers optional/rare keys. UPPER_SNAKE names are
    kept to match the JSON so the migration is mechanical.
    """

    _raw: Mapping[str, Any]

    def __getattr__(self, name: str) -> Any:
        try:
            return object.__getattribute__(self, "_raw")[name]
        except KeyError as exc:
            raise AttributeError(name) from exc

    def get(self, name: str, default: Any = None) -> Any:
        return self._raw.get(name, default)

    def as_dict(self) -> dict:
        return dict(self._raw)


_DEFAULTS: dict[str, Any] = {
    "ROBOT_BASE_OFFSET": [0.0, 0.0, 0.0],
    "TARGET_OFFSET": [0.0, 0.0, 0.0],
    "RETARGET_BASE_ROTATION": [[1.0, 0.0, 0.0], [0.0, 1.0, 0.0], [0.0, 0.0, 1.0]],
    "RETARGET_BASE_TRANSLATION": [0.0, 0.0, 0.0],
    "POSE_BACKEND": "rtmpose-m-hand5",
    "GRIPPER": "binary",
    "EXPORT_FPS": 15,
    "EPISODES_DIR": "data/episodes",
    "DATASETS_DIR": "data/datasets",
    "SKELETON_SAVE_JSON_DEBUG": False,
    "ACTIVE_CALIBRATION": "",
    "CLOUD_STRIDE": 1,
    "CLOUD_VOXEL_M": 0.005,
    "CLOUD_WORKSPACE_BBOX": [-0.8, 0.8, -0.8, 0.8, -0.8, 1.2],
    "CLOUD_MAX_POINTS_PER_FRAME": 40000,
    "CLOUD_BG_SUBTRACT": True,
    "CLOUD_BG_TOLERANCE_MM": 50.0,
    "PERCEPTION_TRACK_LM": list(range(21)),
    "PERCEPTION_INTERP_MAX_GAP": 0,
    "PERCEPTION_CONF_ALPHA": 1.0,
    "PERCEPTION_HAND_FIT": True,
    "PERCEPTION_SAVE_OBSERVATIONS": True,
    "PERCEPTION_HAND_POSE_SOURCE": "hand_fit",
    "PERCEPTION_HAND_FIT_ROI_MARGIN_M": 0.030,
    "PERCEPTION_HAND_FIT_FOREARM_CUT_M": 0.010,
    "PERCEPTION_HAND_FIT_VOXEL_M": 0.004,
    "PERCEPTION_HAND_FIT_HUBER_M": 0.010,
    "PERCEPTION_HAND_FIT_W_DATA": 500.0,
    "PERCEPTION_HAND_FIT_W_VEL_TRANSLATION": 40.0,
    "PERCEPTION_HAND_FIT_W_VEL_ROTATION": 4.0,
    "PERCEPTION_HAND_FIT_W_VEL_JOINTS": 0.8,
    "PERCEPTION_HAND_FIT_W_ACC_TRANSLATION": 120.0,
    "PERCEPTION_HAND_FIT_W_ACC_ROTATION": 10.0,
    "PERCEPTION_HAND_FIT_W_ACC_JOINTS": 1.6,
    "PERCEPTION_HAND_FIT_W_PRIOR": 100.0,
    "PERCEPTION_HAND_FIT_W_POSTURE": 0.004,
    "PERCEPTION_HAND_FIT_W_LANDMARK": 4.0,
    "PERCEPTION_HAND_FIT_LANDMARK_DECAY": 0.35,
    "PERCEPTION_HAND_FIT_INSIDE_SCALE": 0.15,
    "PERCEPTION_HAND_FIT_MIN_POINTS": 40,
    "PERCEPTION_HAND_FIT_MAX_POINTS": 400,
    "PERCEPTION_HAND_FIT_MAX_NFEV": 35,
    "PERCEPTION_HAND_FIT_OUTER_ITERATIONS": 4,
    "PERCEPTION_HAND_FIT_WINDOW": 120,
    "PERCEPTION_HAND_FIT_WINDOW_OVERLAP": 30,
    "PERCEPTION_HAND_FIT_WORKERS": 0,
    "PERCEPTION_HAND_FIT_WARM_START_MAD_K": 6.0,
    "PERCEPTION_HAND_FIT_DEADLINE_S": 120.0,
    "KINECT_SYNC": {},
    "RETARGET_IK_CONF_FLOOR": 0.05,
    "WORLD_ANCHOR_FILENAME": "data/world_anchor.json",
    "CALIB_POSE_MIN_ANGLE_DEG": 8.0,
    "CALIB_POSE_MIN_TRANSLATION_M": 0.05,
    "CALIB_TILT_MIN_DEG": 25.0,
    "CALIB_MIN_SETS": 8,
    "CALIB_MIN_COVISIBLE_SETS": 6,
    "CALIB_MIN_TILTED_SETS": 3,
    "CALIB_MIN_FRAME_COVERAGE": 0.45,
    "VALIDATION_FILENAME": "data/validation_report.json",
    "CALIB_VALIDATE_GREEN_NN_MM": 15.0,
    "CALIB_VALIDATE_GREEN_ICP_TRANS_MM": 20.0,
    "CALIB_VALIDATE_GREEN_ICP_ROT_DEG": 2.0,
    "CALIB_VALIDATE_AMBER_NN_MM": 30.0,
    "CALIB_VALIDATE_AMBER_ICP_TRANS_MM": 50.0,
    "CALIB_VALIDATE_AMBER_ICP_ROT_DEG": 5.0,
}


def load(path: str | None = None) -> Config:
    """
    Load configuration into an immutable :class:`Config`.

    Reads ``path`` (default: the user config, copied from the default on first
    run), overlays :data:`_DEFAULTS` for missing keys, and freezes the result.
    """
    raw = dict(_DEFAULTS)
    if path is None:
        raw.update(_load_config())
    else:
        with open(path) as f:
            raw.update(json.load(f))
    return Config(MappingProxyType(raw))


# Keep a reference to the paths for the API
DEFAULT_CONFIG_PATH = DEFAULT_CONFIG_PATH
USER_CONFIG_PATH = USER_CONFIG_PATH
