"""
viki.calibration.models
--------------------
Data models for camera calibration.

This module defines dataclasses that hold calibration parameters,
samples, and results. These are used throughout the calibration pipeline.
"""

from dataclasses import dataclass, field
from cv2.typing import MatLike
import numpy as np
import cv2
from typing import Tuple
from viki.capture.base import Frame


@dataclass
class BoardParameters:
    """
    Physical parameters of a calibration board.

    Attributes
    ----------
    board_size : Tuple[int, int]
        Number of internal corners per row and column (e.g., (9,6) for a 9x6 board).
    square_size : float
        Length of one square side in real-world units (e.g., meters).
    """

    board_size: Tuple[int, int]
    square_size: float


@dataclass
class ArucoBoardParameters(BoardParameters):
    """
    Parameters specific to a ChArUco board.

    Attributes
    ----------
    marker_size : float
        Size of each ArUco marker (in the same units as square_size).
    aruco_dict : int
        OpenCV predefined dictionary ID (e.g., cv2.aruco.DICT_6X6_250).
    """

    marker_size: float
    # aruco_dict: int = cv2.aruco.DICT_6X6_250
    aruco_dict: int


@dataclass
class CalibrationSample:
    """
    A single calibration sample (detected corners from one image).

    Attributes
    ----------
    frame : Frame
        The original captured frame (may include color and depth).
    corners : MatLike
        Detected corner points (image coordinates).
    resolution : Tuple[int, int]
        Image resolution as (width, height).
    board_params : BoardParameters
        Board parameters used for this detection.
    """

    frame: Frame
    corners: MatLike
    resolution: Tuple[int, int]
    board_params: BoardParameters


@dataclass
class ArucoCalibrationSample(CalibrationSample):
    """
    Extended sample for ChArUco boards, containing detected corner data.

    Attributes
    ----------
    c_ids : MatLike
        IDs of detected chessboard corners (from Charuco detection).
    """

    c_ids: MatLike


@dataclass
class CalibrationIntrinsics:
    """
    Camera intrinsic parameters.

    Attributes
    ----------
    fx, fy : float
        Focal lengths in pixels.
    cx, cy : float
        Principal point coordinates.
    dist_coeffs : np.ndarray
        Distortion coefficients (k1, k2, p1, p2, k3) as a 5‑element vector.
    """

    fx: float
    fy: float
    cx: float
    cy: float
    dist_coeffs: np.ndarray = field(default_factory=lambda: np.zeros(5))

    @property
    def camera_matrix(self) -> np.ndarray:
        """Return the 3x3 camera matrix."""
        return np.array(
            [
                [self.fx, 0.0, self.cx],
                [0.0, self.fy, self.cy],
                [0.0, 0.0, 1.0],
            ],
            dtype=np.float32,
        )


@dataclass
class CalibrationExtrinsics:
    """
    Extrinsic parameters (pose of the board relative to the camera).

    Attributes
    ----------
    rvec : np.ndarray
        Rotation vector (3 elements) in Rodrigues form.
    tvec : np.ndarray
        Translation vector (3 elements).
    """

    rvec: np.ndarray = field(default_factory=lambda: np.zeros(3))
    tvec: np.ndarray = field(default_factory=lambda: np.zeros(3))

    @property
    def rotation_matrix(self) -> np.ndarray:
        """Convert rotation vector to a 3x3 rotation matrix."""
        R, _ = cv2.Rodrigues(self.rvec)
        return np.array(R)

    @property
    def transform_matrix(self) -> np.ndarray:
        """Return a 4x4 homogeneous transformation matrix from camera to board."""
        R = self.rotation_matrix
        R_inv = R.T
        t = self.tvec.flatten()
        T = np.eye(4)
        T[:3, :3] = R_inv
        T[:3, 3] = -R_inv @ t
        return np.array(T)


def canonical_board_extrinsics(
    rvec: np.ndarray,
    tvec: np.ndarray,
    board_size: tuple[int, int],
    square_size: float,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Re‑centre the board origin at the board centre and flip the Z axis so the
    camera sits at +Z (board normal points toward the camera).

    Calibration (solvePnP / estimatePoseCharucoBoard) returns the pose of the
    board frame whose origin is the first corner.  We shift the origin to the
    board centre and, if the solved normal points away from the camera, rotate
    the frame 180° about its X axis.  Returns corrected ``(rvec, tvec)``.
    """
    rvec = np.asarray(rvec, dtype=np.float64).reshape(3)
    tvec = np.asarray(tvec, dtype=np.float64).reshape(3)
    R, _ = cv2.Rodrigues(rvec)

    # Shift the origin to the board centre (objp spans 0 .. (n-1)*ss).
    cx_b = (board_size[0] - 1) / 2.0 * square_size
    cy_b = (board_size[1] - 1) / 2.0 * square_size
    t = tvec + R @ np.array([cx_b, cy_b, 0.0])

    # Canonicalise: the camera must sit at +Z in the board frame.
    if float((-R.T @ t)[2]) < 0:
        R = R @ np.diag([1.0, -1.0, -1.0])

    rvec_corr, _ = cv2.Rodrigues(R)
    return rvec_corr.flatten(), t.flatten()
