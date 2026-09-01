"""
viki.calibration.aurco_worker
--------------------
ChArUco board calibration worker.

This module implements a calibration worker that uses a ChArUco board
(combination of ArUco markers and chessboard corners) for robust detection.
It supports intrinsic and extrinsic calibration using OpenCV's Charuco functions.
"""
import cv2
import numpy as np
from typing import List

from cv2.typing import MatLike
from viki.cameras.base import Frame
from viki.cameras.manager import CameraManager
from viki.calibration.geometry import canonical_board_extrinsics
from viki.contracts import (
    ArucoBoardParameters,
    BoardParameters,
    ArucoCalibrationSample,
    CalibrationSample,
    CalibrationIntrinsics,
    CalibrationExtrinsics,
)
from viki.calibration.worker import _CalibrationWorker


class ArucoWorker(_CalibrationWorker):
    """
    Calibration worker for ChArUco boards.

    Detects both ArUco markers and chessboard corners to obtain sub‑pixel
    accurate points. Collects `ArucoCalibrationSample` objects that contain
    both marker and corner information.

    Attributes
    ----------
    dictionary : cv2.aruco.Dictionary
        OpenCV ArUco dictionary.
    aruco_detector : cv2.aruco.ArucoDetector
        Detector for ArUco markers (used for debugging).
    board : cv2.aruco.CharucoBoard
        The ChArUco board object (reused for all detections).
    detector : cv2.aruco.CharucoDetector
        The ChArUco detector.
    """
    def __init__(
        self,
        mgr: CameraManager,
        device_id: str,
        aruco_board_params: ArucoBoardParameters,
    ):
        """
        Parameters
        ----------
        mgr : CameraManager
            Camera manager.
        device_id : str
            Camera identifier.
        aruco_board_params : ArucoBoardParameters
            Parameters for the ChArUco board.
        """
        super().__init__(mgr, device_id, aruco_board_params)
        self.device_id = device_id

        dict_id = aruco_board_params.aruco_dict
        self.dictionary = cv2.aruco.getPredefinedDictionary(dict_id)
        self.aruco_detector = cv2.aruco.ArucoDetector(self.dictionary)

        # Create the ChArUco board object once (reused for all detections)
        self.board = cv2.aruco.CharucoBoard(
            aruco_board_params.board_size,
            aruco_board_params.square_size,
            aruco_board_params.marker_size,
            self.dictionary,
        )

        self.detector = cv2.aruco.CharucoDetector(self.board)

    def set_board_params(self, board_params: BoardParameters) -> None:
        """
        Override to update the ChArUco board and detector when board params change.

        Note: This does not call super() because the base class only stores the
        parameters; here we need to rebuild the OpenCV objects.
        """
        # super().set_board_params(board_params)
        if not isinstance(board_params, ArucoBoardParameters):
            return
        with self._lock:
            self.board = cv2.aruco.CharucoBoard(
                board_params.board_size,
                board_params.square_size,
                board_params.marker_size,
                self.dictionary,
            )
            self.detector = cv2.aruco.CharucoDetector(self.board)

    def add_sample(self, frame: Frame) -> None:
        """
        Detect the ChArUco board in the frame and store a sample if successful.

        The sample includes both ArUco markers and chessboard corners, which
        are required for accurate calibration.

        Logs debug information about the number of markers and corners found.
        """
        gray = cv2.cvtColor(frame.color, cv2.COLOR_BGR2GRAY)

        # Debug: detect markers separately to see what's actually visible
        markers_raw, ids_raw, _ = self.aruco_detector.detectMarkers(gray)
        if ids_raw is not None:
            self._logger.debug(f"{self.device_id} markers visible: {len(ids_raw)}")
        else:
            self._logger.debug(f"{self.device_id} no markers visible at all")

        # 1. Detect ArUco markers
        corners, c_ids, _, m_ids = self.detector.detectBoard(gray)

        if m_ids is None or len(m_ids) == 0:
            self._logger.debug(
                f"{self.device_id} add_sample: detectBoard failed to find markers. Board size: {self.board_params.board_size}. "
                f"Raw markers found: {len(ids_raw) if ids_raw is not None else 0}"
            )
            return
        if c_ids is None or len(c_ids) == 0:
            self._logger.debug(
                f"{self.device_id} add_sample: detectBoard found markers but no corners. Board size: {self.board_params.board_size}"
            )
            return

        self._logger.debug(
            f"{self.device_id} add_sample: success. Corners: {len(corners)}, Markers: {len(m_ids)}"
        )

        h, w = frame.color.shape[:2]

        # Store the detected corners and IDs inside the sample
        sample = ArucoCalibrationSample(
            frame=frame,
            corners=corners,
            resolution=(w, h),
            board_params=self.board_params,
            c_ids=c_ids,
        )

        with self._lock:
            self._samples.append(sample)

        self._logger.debug(f"{self.device_id} add_sample: success")

    def extrinsics_calibration(
        self,
        intrinsics: CalibrationIntrinsics,
        sample: CalibrationSample | None = None,
    ) -> CalibrationExtrinsics:
        """
        Compute the pose (rotation and translation) of the board relative to the camera.

        Uses `cv2.aruco.estimatePoseCharucoBoard` (fallback to `cv2.solvePnP`).
        If no sample is provided, the most recent collected sample is used.

        Parameters
        ----------
        intrinsics : CalibrationIntrinsics
            Known camera intrinsics.
        sample : Optional[CalibrationSample]
            Sample to use; if None, uses the last sample.

        Returns
        -------
        CalibrationExtrinsics
            Rotation vector and translation vector.

        Raises
        ------
        RuntimeError
            If no sample available, sample is not a CharUco sample,
            or pose estimation fails.
        """
        # Solve one rigid board->camera pose from ChArUco 2D<->3D
        # correspondences with ``cv2.solvePnP`` over ``getChessboardCorners()``.
        # We deliberately do NOT use ``cv2.aruco.estimatePoseCharucoBoard``: its
        # pose is expressed in the CharucoBoard's native object frame, whose
        # origin/axes differ from ``getChessboardCorners()`` in OpenCV >= 4.7, so
        # the re-centre in ``canonical_board_extrinsics`` (which assumes the
        # first-corner origin) is applied from the wrong point and the board
        # lands ~0.7 m off the real plane. This mirrors
        # ``viki.calibration.samples.solve_extrinsics`` exactly.
        if sample is not None:
            if not type(sample) is ArucoCalibrationSample:
                msg = "ArucoWorker extrinsics_calibration: sample is not CharUco sample"
                self._logger.debug(msg)
                raise RuntimeError(msg)
            solve_samples = [sample]
        else:
            solve_samples = list(self._samples)
        if not solve_samples:
            msg = f"{self.device_id} extrinsics: no sample available"
            self._logger.debug(msg)
            raise RuntimeError(msg)

        camera_matrix = intrinsics.camera_matrix
        dist_coeffs = intrinsics.dist_coeffs
        obj_all = np.asarray(self.board.getChessboardCorners(), dtype=np.float32)
        obj_pts, img_pts = [], []
        for s in solve_samples:
            if s.corners is None or s.c_ids is None:
                continue
            ids = np.asarray(s.c_ids, dtype=int).reshape(-1)
            cor = np.asarray(s.corners, dtype=np.float32).reshape(-1, 2)
            if ids.size < 4 or ids.size != len(cor) or int(ids.max(initial=-1)) >= len(obj_all):
                continue
            obj_pts.append(obj_all[ids])
            img_pts.append(cor)
        if not obj_pts:
            msg = f"ArucoWorker extrinsics_calibration: no usable corners"
            self._logger.debug(msg)
            raise RuntimeError(msg)

        obj_v, img_v = np.vstack(obj_pts), np.vstack(img_pts)
        ret, rvec, tvec = cv2.solvePnP(
            obj_v, img_v, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ret:
            msg = f"{self.device_id} extrinsics: pose estimation failed"
            self._logger.debug(msg)
            raise RuntimeError(msg)

        # Reprojection RMS — a healthy solve is ~1 px. A large value means the
        # intrinsics don't match the images (wrong resolution) or the board
        # moved between captures; either way the pose is not trustworthy.
        proj, _ = cv2.projectPoints(obj_v, rvec, tvec, camera_matrix, dist_coeffs)
        rms = float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - img_v) ** 2, axis=1))))
        msg = (f"{self.device_id} extrinsics: {len(obj_v)} pts from {len(obj_pts)} "
               f"set(s), reproj RMS {rms:.2f} px")
        (self._logger.warning if rms > 3.0 else self._logger.info)(
            msg + ("  — POSE UNRELIABLE, check intrinsics resolution / board" if rms > 3.0 else "")
        )

        rvec, tvec = canonical_board_extrinsics(
            rvec, tvec, self.board_params.board_size, self.board_params.square_size
        )
        return CalibrationExtrinsics(rvec=rvec, tvec=tvec)

    def mark_board(self, frame: Frame) -> np.ndarray:
        """
        Generate a debug image with detected ArUco markers and Charuco corners overlaid.

        Parameters
        ----------
        frame : Frame
            Input frame.

        Returns
        -------
        np.ndarray
            Annotated BGR image (original if no detection).
        """
        gray = cv2.cvtColor(frame.color, cv2.COLOR_BGR2GRAY)
        markers_raw, ids_raw, _ = self.aruco_detector.detectMarkers(gray)
        if ids_raw is None:
            return frame.color
        corners, c_ids, markers, m_ids = self.detector.detectBoard(gray)
        if m_ids is None or len(m_ids) == 0 or c_ids is None or len(c_ids) == 0:
            return frame.color
        debug_img = frame.color.copy()
        try:
            cv2.aruco.drawDetectedMarkers(
                debug_img, markers, m_ids, borderColor=(0, 255, 0)
            )
            corners_pts = np.asarray(corners, dtype=np.float32).reshape(-1, 1, 2)
            corner_ids = np.asarray(c_ids, dtype=np.int32).reshape(-1, 1)
            cv2.aruco.drawDetectedCornersCharuco(
                debug_img, corners_pts, corner_ids, (0, 0, 255)
            )
        except Exception as e:
            self._logger.debug(f"{self.device_id} mark_board overlay failed: {e}")
        return debug_img
