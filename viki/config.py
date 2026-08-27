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
INTRINSICS_FILENAME: str
EXTRINSICS_FILENAME: str
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
RECORDING_DURATION: int
RECORDING_FPS: int
RETARGET_DEFAULT_ROBOT: str
RETARGET_LANDMARK_SG_WINDOW: int
RETARGET_LANDMARK_SG_POLYORDER: int
RETARGET_IK_POSITION_COST: float
RETARGET_IK_ORIENTATION_COST: float
RETARGET_IK_POSTURE_COST: float
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
    "POSE_BACKEND": "mediapipe",
    "GRIPPER": "binary",
    "EXPORT_FPS": 15,
    "EPISODES_DIR": "data/episodes",
    "DATASETS_DIR": "data/datasets",
    "SKELETON_SAVE_JSON_DEBUG": False,
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
