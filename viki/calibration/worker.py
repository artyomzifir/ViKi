"""
viki.calibration.worker
--------------------
Abstract base class for camera calibration workers.

This module defines the common interface for collecting calibration samples
(chessboard or ChArUco) and computing intrinsic/extrinsic parameters.
Each worker runs in its own thread when `mode='auto'` and continuously
captures frames from the associated camera.
"""

from abc import ABC, abstractmethod
import threading
import cv2
import numpy as np
import logging
from typing import List
from viki.capture.base import Frame
from viki.capture.manager import CameraManager
from viki.calibration.models import (
    BoardParameters,
    CalibrationSample,
    CalibrationIntrinsics,
    CalibrationExtrinsics,
)


class _CalibrationWorker(ABC):
    """
    Base class for calibration workers.

    Manages a thread that periodically captures frames and adds valid
    calibration samples. Subclasses must implement board detection,
    sample addition, and calibration routines.

    Attributes
    ----------
    device_id : str
        Unique identifier for the camera.
    _mgr : CameraManager
        Manager to retrieve the latest frame.
    _board_params : BoardParameters
        Current board configuration.
    _samples : List[CalibrationSample]
        Collected samples (thread-safe).
    _lock : threading.Lock
        Protects access to _samples and _board_params.
    _stop_event : threading.Event
        Signals the capture thread to stop.
    _thread : threading.Thread
        Background thread for auto-capture.
    """

    def __init__(
        self, mgr: CameraManager, device_id: str, board_params: BoardParameters
    ):
        """
        Parameters
        ----------
        mgr : CameraManager
            Camera manager providing frame access.
        device_id : str
            Identifier of the camera to calibrate.
        board_params : BoardParameters
            Physical parameters of the calibration board.
        """
        self._mgr = mgr
        self.device_id = device_id
        self._logger = logging.getLogger(__name__)
        self._samples: List[CalibrationSample] = []
        self._board_params = board_params

        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        """Start the background capture thread (auto mode)."""
        self._thread.start()

    def stop(self) -> None:
        """Signal the background thread to stop and join."""
        self._stop_event.set()

    def set_board_params(self, board_params: BoardParameters) -> None:
        """Update the board parameters in a thread-safe manner."""
        with self._lock:
            self._board_params = board_params

    @property
    def board_params(self) -> BoardParameters:
        """Return a copy of the current board parameters (thread-safe)."""
        with self._lock:
            return self._board_params

    @property
    def samples_count(self) -> int:
        """Number of collected calibration samples."""
        with self._lock:
            return len(self._samples)

    @property
    def samples(self) -> List[CalibrationSample]:
        """Return a copy of the collected samples list (thread-safe)."""
        with self._lock:
            return self._samples.copy()

    @abstractmethod
    def add_sample(self, frame: Frame) -> None:
        """
        Detect the calibration board in the frame and, if successful,
        append a new sample to the internal list.

        This method is called by `capture()` and should be implemented
        by subclasses to perform board‑specific detection.

        Parameters
        ----------
        frame : Frame
            The captured frame to process.
        """
        pass

    @abstractmethod
    def intrinsics_calibration(
        self, samples: List[CalibrationSample] | None = None
    ) -> CalibrationIntrinsics:
        """
        Perform intrinsic calibration using the collected (or provided) samples.
        Parameters
        ----------
        samples : Optional[List[CalibrationSample]]
            Optional list of samples; if None, uses internal samples.
        Returns
        -------
        CalibrationIntrinsics
            Focal lengths, principal point, distortion coefficients.
        Raises
        ------
        RuntimeError
            If not enough valid samples or calibration fails.
        """
        pass

    @abstractmethod
    def extrinsics_calibration(
        self,
        intrinsics: CalibrationIntrinsics,
        sample: CalibrationSample | None = None,
    ) -> CalibrationExtrinsics:
        """
        Compute extrinsic parameters (rotation and translation) for a given sample.
        Parameters
        ----------
        intrinsics : CalibrationIntrinsics
            Known camera intrinsics.
        sample : Optional[CalibrationSample]
            The sample to use; if None, uses the most recent one.
        Returns
        -------
        CalibrationExtrinsics
            Rotation vector and translation vector.
        Raises
        ------
        RuntimeError
            If no sample available or pose estimation fails.
        """
        pass

    def clear(self):
        """Remove all collected samples."""
        with self._lock:
            self._samples = []

    def capture(self) -> None:
        """
        Fetch the latest frame from the camera and attempt to add a sample.
        If no frame is available, this method does nothing.
        """
        frame = self._mgr.latest_frame(self.device_id)
        if frame is None:
            return
        self.add_sample(frame)

    @abstractmethod
    def mark_board(self, frame: Frame) -> np.ndarray:
        """
        Produce a debug image with the detected board overlaid.
        Parameters
        ----------
        frame : Frame
            Input frame.
        Returns
        -------
        np.ndarray
            Annotated image (BGR format).
        """
        pass

    def _loop(self) -> None:
        """
        Background thread loop: repeatedly call `capture()` until stopped.
        Exceptions are logged and ignored (except TimeoutError which is silent).
        """
        while not self._stop_event.is_set():
            try:
                self.capture()
            except TimeoutError:
                pass
            except Exception as e:
                self._logger.error(f"{self.device_id} calibration worker error: {e}")
