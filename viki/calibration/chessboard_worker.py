"""
viki.calibration.chessboard_worker
--------------------
Chessboard calibration worker.

This module provides a calibration worker that uses a standard chessboard
pattern. It detects chessboard corners and performs intrinsic/extrinsic
calibration using OpenCV's `findChessboardCorners` and `calibrateCamera`.
"""
import cv2
import numpy as np
from typing import List
from viki.cameras.base import Frame
from viki.calibration.geometry import canonical_board_extrinsics
from viki.contracts import (
    CalibrationSample,
    CalibrationIntrinsics,
    CalibrationExtrinsics,
)
from viki.calibration.worker import _CalibrationWorker


class ChessboardWorker(_CalibrationWorker):
    """
    Calibration worker for standard chessboard patterns.

    Detects chessboard corners (with sub‑pixel refinement) and stores
    samples. Requires at least 20 samples for intrinsic calibration.
    """

    def add_sample(self, frame: Frame) -> None:
        """
        Detect chessboard corners in the frame and store a sample if successful.

        Uses sub‑pixel refinement to improve corner accuracy.

        Parameters
        ----------
        frame : Frame
            Input frame.
        """
        board_params = self.board_params
        board_size = board_params.board_size

        subpix_criteria = (
            cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER,
            30,
            0.001,
        )

        gray = cv2.cvtColor(frame.color, cv2.COLOR_BGR2GRAY)
        ret, corners = cv2.findChessboardCorners(gray, board_size, None)
        if not ret:
            self._logger.debug(
                f"{self.device_id} add sample: cv2.findChessboardCorners failed, chessboard_size: {board_size}"
            )
            return

        refined_corners = cv2.cornerSubPix(
            gray, corners, (11, 11), (-1, -1), subpix_criteria
        )

        w, h = frame.color.shape[:2]

        sample = CalibrationSample(frame, refined_corners, (w, h), board_params)
        with self._lock:
            self._samples.append(sample)

        self._logger.debug(f"{self.device_id} add sample: success")

    def extrinsics_calibration(
        self,
        intrinsics: CalibrationIntrinsics,
        sample: CalibrationSample | None = None,
    ) -> CalibrationExtrinsics:
        """
        Compute the pose (rotation and translation) of the chessboard relative to the camera.

        Uses `cv2.solvePnP` with the object points derived from board parameters.

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
            If no sample available or solvePnP fails.
        """
        if not sample:
            if self.samples_count < 1:
                msg = f"{self.device_id} extrinsics_calibration: no sample"
                self._logger.debug(msg)
                raise RuntimeError(msg)
            sample = self.samples[-1]
 
        square_size = sample.board_params.square_size
        w, h = sample.board_params.board_size

        objp = np.zeros((w * h, 3), np.float32)
        objp[:, :2] = np.mgrid[0:w, 0:h].T.reshape(-1, 2)
        objp *= square_size

        camera_matrix = intrinsics.camera_matrix
        dist_coeffs = intrinsics.dist_coeffs
        ret, rvec, tvec = cv2.solvePnP(objp, sample.corners, camera_matrix, dist_coeffs)

        if not ret:
            msg = f"{self.device_id} extrinsics calibration: cv2.solvePnP failed"
            self._logger.debug(msg)
            raise RuntimeError(msg)

        rvec, tvec = canonical_board_extrinsics(
            rvec, tvec, sample.board_params.board_size, sample.board_params.square_size
        )

        self._logger.debug(f"{self.device_id} extrinsics calibration: success")

        return CalibrationExtrinsics(rvec=rvec, tvec=tvec)

    def mark_board(self, frame: Frame) -> np.ndarray:
        """
        Generate a debug image with detected chessboard corners overlaid.

        Parameters
        ----------
        frame : Frame
            Input frame.

        Returns
        -------
        np.ndarray
            Annotated BGR image (original if no detection).
        """
        pattern_size = self._board_params.board_size
        frm = frame.color
        gray = cv2.cvtColor(frm, cv2.COLOR_BGR2GRAY)
        flags = (
            cv2.CALIB_CB_ADAPTIVE_THRESH
            + cv2.CALIB_CB_NORMALIZE_IMAGE
            + cv2.CALIB_CB_FAST_CHECK
        )
        ret, corners = cv2.findChessboardCorners(gray, pattern_size, None, flags)
        if ret:
            criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)
            corners_sub = cv2.cornerSubPix(gray, corners, (11, 11), (-1, -1), criteria)
            debug_frm = frm.copy()
            cv2.drawChessboardCorners(debug_frm, pattern_size, corners_sub, ret)
            frm = debug_frm
        return frm
