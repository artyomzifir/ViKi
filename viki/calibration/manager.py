"""
viki.calibration.manager
--------------------
CalibrationManager – public interface for camera calibration.

This module provides a high-level manager that coordinates calibration workers
for multiple cameras. It handles starting/stopping workers, collecting samples,
computing intrinsics/extrinsics, and persisting results to JSON files.
"""
import cv2
import json
import logging
from typing import Dict, List
from viki.capture.manager import CameraManager
from viki.calibration.models import (
    ArucoBoardParameters,
    BoardParameters,
    CalibrationSample,
    CalibrationIntrinsics,
    CalibrationExtrinsics,
)
from viki.config import (
    INTRINSICS_FILENAME,
    EXTRINSICS_FILENAME,
    CALIB_MODE,
    CALIB_BOARD_TYPE,
    CALIB_CHESS_BOARD_SIZE,
    CALIB_CHESS_SQUARE_SIZE,
    CALIB_ARUCO_BOARD_SIZE,
    CALIB_ARUCO_SQUARE_SIZE,
    CALIB_ARUCO_MARKER_SIZE,
    CALIB_ARUCO_DICT,
)
from viki.calibration.file import (
    read_device_intrinsics,
    read_device_extrinsics,
    write_device_intrinsics,
    write_device_extrinsics,
)
from viki.calibration.worker import _CalibrationWorker
from viki.calibration.chessboard_worker import ChessboardWorker
from viki.calibration.aruco_worker import ArucoWorker


class CalibrationManager:
    """
    Central manager for multi‑camera calibration.

    This class is intended to be a singleton in the application. It manages
    one `_CalibrationWorker` per camera device, handles sample collection,
    and provides methods to compute, load, and save calibration parameters.

    Attributes
    ----------
    _mgr : CameraManager
        CameraManager used to fetch frames.
    _intrinsics : Dict[str, CalibrationIntrinsics]
        Dict mapping device_id to loaded/computed intrinsics.
    _extrinsics : Dict[str, CalibrationExtrinsics]
        Dict mapping device_id to loaded/computed extrinsics.
    _workers : Dict[str, _CalibrationWorker]
        Dict mapping device_id to its active worker.
    """
    def __init__(self, mgr: CameraManager):
        """
        Parameters
        ----------
        mgr : CameraManager
            CameraManager instance providing frame access.
        """
        self._mgr = mgr
        self._logger = logging.getLogger(__name__)
        self._intrinsics: Dict[str, CalibrationIntrinsics] = {}
        self._extrinsics: Dict[str, CalibrationExtrinsics] = {}
        self._workers: Dict[str, _CalibrationWorker] = {}

    def start(
        self,
        device_id: str,
        mode=CALIB_MODE,
        board_type=CALIB_BOARD_TYPE,
        board_size=None,
        square_size=None,
        marker_size=None,
        aruco_dict=CALIB_ARUCO_DICT,
    ) -> None:
        """
        Start a calibration worker for a given camera.

        The worker will run in the background if `mode='auto'`, automatically
        capturing frames and adding samples. If `mode='manual'`, samples are
        added only when `capture()` is called explicitly.

        Parameters
        ----------
        device_id : str
            Camera identifier.
        mode : str
            "auto" or "manual". If "auto", the worker starts its own thread.
        board_type : str
            "chess" or "aruco".
        board_size : Optional[Tuple[int, int]]
            Number of internal corners (width, height) of the board.
        square_size : Optional[float]
            Physical size of a square (in meters or mm).
        marker_size : Optional[float]
            Physical size of ArUco markers (only for ChArUco).
        aruco_dict : int
            OpenCV ArUco dictionary ID (only for ChArUco).

        Raises
        ------
        ValueError
            If board_type is unknown.
        """
        if device_id in self._workers:
            self._logger.warning(
                f"CalibrationManager start: {device_id} has already started"
            )
            return

        if board_type == "chess":
            bs = board_size or CALIB_CHESS_BOARD_SIZE
            ss = square_size or CALIB_CHESS_SQUARE_SIZE
            board_params = BoardParameters(bs, ss)
            worker = ChessboardWorker(self._mgr, device_id, board_params)
        else:
            bs = board_size or CALIB_ARUCO_BOARD_SIZE
            ss = square_size or CALIB_ARUCO_SQUARE_SIZE
            ms = marker_size or CALIB_ARUCO_MARKER_SIZE
            board_params = ArucoBoardParameters(bs, ss, ms, aruco_dict)
            worker = ArucoWorker(self._mgr, device_id, board_params)
        if mode == "auto":
            worker.start()
        self._workers[device_id] = worker

    def stop(self, device_id: str) -> None:
        """Stop and remove the worker for the given device."""
        worker = self._workers.pop(device_id, None)
        if worker:
            worker.stop()
            return
        self._logger.warning(
            f"CalibrationManager stop: {device_id} is not in worker list"
        )

    def stop_all(self) -> None:
        """Stop all active workers."""
        for device_id in list(self._workers):
            self.stop(device_id)

    def sync_params(
        self,
        board_type: str,
        board_size: tuple[int, int],
        square_size: float,
        marker_size: float = CALIB_ARUCO_MARKER_SIZE,
        aruco_dict: int = CALIB_ARUCO_DICT,
    ) -> None:
        """
        Update board parameters for all active workers.

        Useful when the physical board configuration changes (e.g., using a
        different board). All workers will use the new parameters for subsequent
        detections.

        Parameters
        ----------
        board_type : str
            "chess" or "aruco".
        board_size : Tuple[int, int]
            New board size.
        square_size : float
            New square size.
        marker_size : float
            New marker size (only for ChArUco).
        aruco_dict : int
            New dictionary ID (only for ChArUco).
        """
        if board_type == "chess":
            params = BoardParameters(board_size, square_size)
        else:
            params = ArucoBoardParameters(
                board_size, square_size, marker_size, aruco_dict
            )

        for worker in self._workers.values():
            worker.set_board_params(params)

    def is_device_active(self, device_id: str) -> bool:
        """Return True if a worker exists for the given device."""
        if self._workers.get(device_id):
            return True
        return False

    def clear(self, device_id: str) -> None:
        """Clear all collected samples for the given device."""
        worker = self._workers.get(device_id)
        if not worker:
            self._logger.warning(
                f"CalibrationManager status: {device_id} is not in worker list"
            )
            return
        worker.clear()

    def intrinsics_calibration(
        self,
        device_id: str,
        results_path: str = INTRINSICS_FILENAME,
        samples: List[CalibrationSample] | None = None,
    ) -> CalibrationIntrinsics:
        """
        Compute intrinsic parameters for the given device and persist them.

        The computed intrinsics are saved to `results_path` (JSON) and also
        stored in the internal cache (`_intrinsics`).

        Parameters
        ----------
        device_id : str
            Camera identifier.
        results_path : str
            Path to the JSON file for saving.
        samples : Optional[List[CalibrationSample]]
            Optional list of samples; if None, uses the worker's internal samples.

        Returns
        -------
        CalibrationIntrinsics
            The computed intrinsics.

        Raises
        ------
        RuntimeError
            If no worker exists or calibration fails.
        """
        worker = self._workers.get(device_id)
        if not worker:
            msg = f"CalibrationManager intrinsics_calibration: {device_id} is not in worker list"
            self._logger.warning(msg)
            raise RuntimeError(msg)

        intrinsics = (
            worker.intrinsics_calibration(samples)
            if samples
            else worker.intrinsics_calibration()
        )

        write_device_intrinsics(device_id, intrinsics, results_path)

        return intrinsics

    def write_intrinsics(
        self,
        device_id: str,
        intrinsics: CalibrationIntrinsics,
        path: str = INTRINSICS_FILENAME,
    ):
        """Write intrinsics to a JSON file without recomputing."""
        write_device_intrinsics(device_id, intrinsics, path)

    def load_intrinsics(self, device_id: str, path: str = INTRINSICS_FILENAME) -> None:
        """Load intrinsics from a JSON file into the internal cache."""
        intrinsics = read_device_intrinsics(device_id, path)
        if not intrinsics:
            return
        self._intrinsics[device_id] = intrinsics

    def set_intrinsics(
        self, device_id: str, intrinsics: CalibrationIntrinsics, path: str = ""
    ) -> None:
        """
        Manually set intrinsics (and optionally persist them).

        If `path` is non‑empty, the intrinsics are written to that file.
        The intrinsics are also stored in the internal cache.

        Parameters
        ----------
        device_id : str
            Camera identifier.
        intrinsics : CalibrationIntrinsics
            The intrinsic parameters to set.
        path : str
            If provided, save to this JSON file.
        """
        if path != "":
            self.write_intrinsics(device_id, intrinsics, path)
        self._intrinsics[device_id] = intrinsics

    def get_intrinsics(
        self, device_id: str, path: str = ""
    ) -> CalibrationIntrinsics | None:
        """
        Retrieve intrinsics for a device.

        If `path` is provided, it loads from that file first.
        Otherwise, it checks the internal cache, then falls back to the
        default intrinsics file (`INTRINSICS_FILENAME`).

        Parameters
        ----------
        device_id : str
            Camera identifier.
        path : str
            Optional specific file to load from.

        Returns
        -------
        Optional[CalibrationIntrinsics]
            Intrinsics or None if not found.
        """
        if path != "":
            self.load_intrinsics(device_id, path)
        elif device_id not in self._intrinsics:
            self.load_intrinsics(device_id, INTRINSICS_FILENAME)
            
        return self._intrinsics.get(device_id)

    def extrinsics_calibration(
        self,
        device_id: str,
        results_path: str = EXTRINSICS_FILENAME,
        sample: CalibrationSample | None = None,
        intrinsics: CalibrationIntrinsics | None = None,
    ) -> CalibrationExtrinsics:
        """
        Compute extrinsic parameters (pose) for the given device.

        Uses a single sample (or the most recent one) and the provided or
        cached intrinsics. Results are saved to `results_path` and cached.

        Parameters
        ----------
        device_id : str
            Camera identifier.
        results_path : str
            Path to JSON file for saving.
        sample : Optional[CalibrationSample]
            Specific sample to use; if None, uses the worker's last sample.
        intrinsics : Optional[CalibrationIntrinsics]
            Intrinsics to use; if None, fetches from cache/default file.

        Returns
        -------
        CalibrationExtrinsics
            Rotation and translation.

        Raises
        ------
        RuntimeError
            If no worker, no intrinsics, or pose estimation fails.
        """
        worker = self._workers.get(device_id)
        if not worker:
            msg = f"CalibrationManager extrinsics_calibration: {device_id} is not in worker list"
            self._logger.warning(msg)
            raise RuntimeError(msg)

        if not intrinsics:
            intrinsics = self.get_intrinsics(device_id)
            if not intrinsics:
                msg = f"CalibrationManager extrinsics_calibration: no intrinsics"
                self._logger.warning(msg)
                raise RuntimeError(msg)

        extrinsics = (
            worker.extrinsics_calibration(intrinsics, sample)
            if sample
            else worker.extrinsics_calibration(intrinsics)
        )

        write_device_extrinsics(device_id, extrinsics, results_path)
        self.set_extrinsics(device_id, extrinsics)

        return extrinsics

    def write_extrinsics(
        self,
        device_id: str,
        extrinsics: CalibrationExtrinsics,
        path: str = EXTRINSICS_FILENAME,
    ):
        """Write extrinsics to a JSON file without recomputing."""
        write_device_extrinsics(device_id, extrinsics, path)

    def load_extrinsics(self, device_id: str, path: str = EXTRINSICS_FILENAME) -> None:
        """Load extrinsics from a JSON file into the internal cache."""
        extrinsics = read_device_extrinsics(device_id, path)
        if not extrinsics:
            return
        self._extrinsics[device_id] = extrinsics

    def load_all_extrinsics(self, path: str = EXTRINSICS_FILENAME) -> None:
        """Load ALL extrinsics entries from file into the internal cache."""
        try:
            with open(path) as f:
                data = json.load(f)
        except (FileNotFoundError, json.JSONDecodeError):
            self._logger.info("No extrinsics file at %s, skipping", path)
            return

        if not isinstance(data, list):
            return

        count = 0
        from viki.calibration.file import read_device_extrinsics
        for entry in data:
            dev_id = entry.get("device_id") if isinstance(entry, dict) else None
            if dev_id:
                self.load_extrinsics(dev_id, path)
                count += 1

        if count:
            self._logger.info(
                "Loaded extrinsics for %d device(s): %s",
                count, [e.get("device_id") for e in data if isinstance(e, dict)],
            )

    def set_extrinsics(
        self, device_id: str, extrinsics: CalibrationExtrinsics, path: str = ""
    ) -> None:
        """
        Manually set extrinsics (and optionally persist them).

        If `path` is non‑empty, the extrinsics are written to that file.
        The extrinsics are also stored in the internal cache.

        Parameters
        ----------
        device_id : str
            Camera identifier.
        extrinsics : CalibrationExtrinsics
            The extrinsic parameters to set.
        path : str
            If provided, save to this JSON file.
        """
        if path != "":
            self.write_extrinsics(device_id, extrinsics, path)
        self._extrinsics[device_id] = extrinsics

    def get_extrinsics(
        self, device_id: str, path: str = ""
    ) -> CalibrationExtrinsics | None:
        """
        Retrieve extrinsics for a device.

        If `path` is provided, it loads from that file first.
        Otherwise, it checks the internal cache, then falls back to the
        default extrinsics file (`EXTRINSICS_FILENAME`).

        Parameters
        ----------
        device_id : str
            Camera identifier.
        path : str
            Optional specific file to load from.

        Returns
        -------
        Optional[CalibrationExtrinsics]
            Extrinsics or None if not found.
        """
        if path != "":
            self.load_extrinsics(device_id, path)
        elif device_id not in self._extrinsics:
            self.load_extrinsics(device_id, EXTRINSICS_FILENAME)
            
        extrinsics = self._extrinsics.get(device_id)
        if not extrinsics:
            self._logger.debug(
                f"CalibrationManager get_extrinsics: {device_id} not in extrinsics list"
            )
        return extrinsics

    def capture_all(self) -> None:
        """Manually trigger sample capture for all active workers."""
        for device_id, worker in self._workers.items():
            worker.capture()

    def capture(self, device_id: str) -> None:
        """Manually trigger sample capture for a specific device."""
        worker = self._workers.get(device_id)
        if not worker:
            self._logger.warning(
                f"CalibrationManager capture: {device_id} not in workers list"
            )
            return
        worker.capture()

    def samples_count(self, device_id: str) -> int:
        """Return the number of collected samples for the device."""
        worker = self._workers.get(device_id)
        if not worker:
            self._logger.debug(
                f"CalibrationManager samples_amount: {device_id} is not in worker list"
            )
            return 0
        return worker.samples_count

    def get_board_params(self):
        """Return board parameters from the first active worker, or None."""
        for w in self._workers.values():
            return w.board_params
        return None

    def status(self, device_id: str) -> dict:
        """
        Return a status dictionary for the device.

        Returns
        -------
        dict
            Example: {"samples_count": 42, "started": True}
        """
        worker = self._workers.get(device_id)
        if not worker:
            return {"samples_count": 0, "started": False}
        return {"samples_count": worker.samples_count, "started": True}
