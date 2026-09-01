"""
viki.server.routes.models
-------------------------
Pydantic models (request/response schemas) for the API endpoints.
"""

from typing import Tuple, List
from pydantic import BaseModel
import numpy as np
import cv2


class BoardParametersData(BaseModel):
    """
    Request model for chessboard parameters.

    Attributes
    ----------
    board_size : Tuple[int, int]
        Number of internal corners (width, height).
    square_size : float
        Physical size of a square in metres.
    """
    board_size: Tuple[int, int]
    square_size: float


class ArucoBoardParametersData(BoardParametersData):
    """
    Request model for ChArUco board parameters.

    Attributes
    ----------
    marker_size : float
        Size of ArUco markers in metres.
    aruco_dict : str
        Name of the OpenCV ArUco dictionary (e.g., "DICT_6X6_250").
    """
    marker_size: float
    aruco_dict: str


# (IntrinsicsResponse removed — intrinsics come straight from the camera SDK,
# there is no calibration/storage endpoint for them any more.)


class ExtrinsicsResponse(BaseModel):
    """
    Response model for camera extrinsic parameters (pose).

    Attributes
    ----------
    device_id : str
        Camera device ID.
    rvec : List[float]
        Rotation vector (3 elements) in Rodrigues form.
    tvec : List[float]
        Translation vector (3 elements).
    """
    device_id: str
    rvec: List[float]
    tvec: List[float]

    @property
    def rvec_np(self) -> np.ndarray:
        """Rotation vector as numpy array."""
        return np.array(self.rvec, dtype=np.float32)

    @property
    def tvec_np(self) -> np.ndarray:
        """Translation vector as numpy array."""
        return np.array(self.tvec, dtype=np.float32)

    @property
    def rotation_matrix(self) -> np.ndarray:
        """3x3 rotation matrix derived from rvec."""
        R, _ = cv2.Rodrigues(self.rvec_np)
        return R

    @property
    def transform_matrix(self) -> np.ndarray:
        """4x4 homogeneous transformation matrix."""
        R = self.rotation_matrix
        R_inv = R.T
        t = self.tvec_np.flatten()
        T = np.eye(4)
        T[:3, :3] = R_inv
        T[:3, 3] = -R_inv @ t
        return T
