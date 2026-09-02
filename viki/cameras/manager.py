"""
viki.cameras.manager
--------------------
CameraManager: detects, starts, and owns camera backends.
Each camera runs in a background thread; the latest frame is always
available for non-blocking reads by the MJPEG streamer.
"""

from __future__ import annotations

import logging
import threading
import time
import numpy as np
from collections import deque
from typing import Optional

logger = logging.getLogger(__name__)
from concurrent.futures import ThreadPoolExecutor

from viki.cameras.base import Frame, CameraBackend
from viki.config import (
    FRAME_BUFFER_SIZE,
    DEFAULT_FPS,
    DEFAULT_COLOR_WIDTH,
    DEFAULT_COLOR_HEIGHT,
    DEFAULT_DEPTH_MODE,
)

# Frames kept per camera for timestamp-based sync queries.
_FRAME_BUFFER_SIZE = FRAME_BUFFER_SIZE


class _CameraWorker:
    """Background thread that continuously reads frames from one camera."""

    def __init__(self, backend: CameraBackend) -> None:
        self.backend = backend
        self._buffer: deque = deque(maxlen=_FRAME_BUFFER_SIZE)
        self._lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread = threading.Thread(target=self._loop, daemon=True)

    def start(self) -> None:
        self.backend.start()
        self._thread.start()

    def stop(self) -> None:
        """Signal the loop thread to stop. Returns immediately."""
        self._stop_event.set()
        # backend.stop() is called inside _loop's finally block so it is
        # never concurrent with backend.get_frame() on the same handle.

    def join(self, timeout: float = 8.0) -> None:
        """Wait for the loop thread and backend cleanup to finish."""
        self._thread.join(timeout=timeout)

    def latest(self) -> Optional[Frame]:
        with self._lock:
            return self._buffer[-1] if self._buffer else None

    def nearest_to(self, host_timestamp_us: int) -> Optional[Frame]:
        """Return the buffered frame whose host_timestamp_us is closest to the given value."""
        with self._lock:
            if not self._buffer:
                return None
            return min(
                self._buffer, key=lambda f: abs(f.host_timestamp_us - host_timestamp_us)
            )

    def _loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    frame = self.backend.get_frame()
                    # monotonic, not wall clock: an NTP step / VM resume / manual
                    # clock change mid-recording must not make timestamps jump.
                    frame.host_timestamp_us = time.monotonic_ns() // 1000
                    with self._lock:
                        self._buffer.append(frame)
                except TimeoutError:
                    logger.warning(
                        "[%s] get_frame timed out — frame dropped",
                        self.backend.device_id,
                    )
                except Exception as exc:
                    print(f"[worker:{self.backend.device_id}] error: {exc}")
                    time.sleep(0.1)
        finally:
            # Always called from this thread, so it is never concurrent with
            # get_frame() — eliminates the deadlock from calling backend.stop()
            # on a handle that another thread is actively using.
            try:
                self.backend.stop()
            except Exception as exc:
                print(f"[worker:{self.backend.device_id}] stop error: {exc}")


class CameraManager:
    """Manages multiple camera backends and their worker threads."""

    def __init__(self) -> None:
        self._workers: dict[str, _CameraWorker] = {}
        self.calibration: dict[str, dict] = {}
        # FastAPI runs sync route handlers on a threadpool, so two clients (or two
        # browser tabs) can land concurrent start/stop for the *same* device.
        # Without mutual exclusion the second start races the first: both build a
        # backend, the second ``self._workers[id] = worker`` overwrites the first
        # without stopping it, and the orphaned worker keeps the USB/k4a handle
        # claimed forever — the next start then fails with k4a_device_open=1.
        # One reentrant lock per device (start() may re-enter via stop()); calls
        # for different devices still run in parallel.
        self._locks_guard = threading.Lock()
        self._dev_locks: dict[str, threading.RLock] = {}

    def _dev_lock(self, device_id: str) -> "threading.RLock":
        with self._locks_guard:
            lk = self._dev_locks.get(device_id)
            if lk is None:
                lk = self._dev_locks[device_id] = threading.RLock()
            return lk

    # ── Device discovery ──────────────────────────────────────────────────────

    def active_device_ids(self) -> list:
        """Return the device IDs of all currently running cameras."""
        return list(self._workers.keys())

    def list_devices(self) -> dict:
        """Return all detected camera device IDs grouped by type."""
        devices: dict = {
            "realsense": [],
            "kinect": [],
            "active": list(self._workers.keys()),
        }

        try:
            from .realsense import RealSenseBackend

            devices["realsense"] = RealSenseBackend.list_devices()
        except Exception as e:
            devices["realsense_error"] = str(e)

        try:
            from .kinect import KinectBackend

            count = KinectBackend.device_count()
            devices["kinect"] = [f"kinect_{i}" for i in range(count)]
        except Exception as e:
            devices["kinect_error"] = str(e)

        return devices

    # ── Lifecycle ─────────────────────────────────────────────────────────────

    def start(
        self,
        device_id: str,
        fps: int = DEFAULT_FPS,
        color_width: int = DEFAULT_COLOR_WIDTH,
        color_height: int = DEFAULT_COLOR_HEIGHT,
        depth_mode: str = DEFAULT_DEPTH_MODE,
        **kwargs,
    ) -> str:
        want = {
            "color_width": int(color_width),
            "color_height": int(color_height),
            "fps": int(fps),
            "depth_mode": depth_mode,
        }
        with self._dev_lock(device_id):
            existing = self._workers.get(device_id)
            if existing is not None:
                have = existing.backend.config
                # only the keys we set here, and only where the backend actually
                # reports one — a ``None`` means "this backend has no such knob"
                # (RealSense has no depth-mode enum), so it must not force a restart.
                if all(
                    have.get(k) == v
                    for k, v in want.items()
                    if k in have and have.get(k) is not None
                ):
                    return "unchanged"
                # config changed → restart so the request actually takes effect
                self.stop(device_id)

            backend = self._make_backend(
                device_id, fps, color_width, color_height, depth_mode, **kwargs
            )
            worker = _CameraWorker(backend)
            worker.start()
            self._workers[device_id] = worker
            return "restarted" if existing is not None else "started"

    def start_kinect_sync(
        self,
        master_id: str,
        subordinate_ids: list,
        fps: int = DEFAULT_FPS,
        color_width: int = DEFAULT_COLOR_WIDTH,
        color_height: int = DEFAULT_COLOR_HEIGHT,
        depth_mode: str = DEFAULT_DEPTH_MODE,
        subordinate_delay_us: int = 0,
    ) -> None:
        """
        Start Azure Kinects in hardware-sync mode with correct startup order.

        The subordinate(s) must be started before the master so they are
        already listening for trigger pulses when the master begins sending them.

        Parameters
        ----------
        master_id : str
            Device ID of the master Kinect (SYNC OUT connected to cable).
        subordinate_ids : list[str]
            Device IDs of subordinate Kinects (SYNC IN connected to cable).
        subordinate_delay_us : int
            Delay added to each subordinate's capture relative to the master's
            trigger, in microseconds.  A non-zero value staggers the depth IR
            projectors and reduces inter-device interference even further.
        """
        from .kinect import K4A_WIRED_SYNC_MODE_MASTER, K4A_WIRED_SYNC_MODE_SUBORDINATE

        for sub_id in subordinate_ids:
            self.start(
                sub_id,
                fps=fps,
                color_width=color_width,
                color_height=color_height,
                depth_mode=depth_mode,
                wired_sync_mode=K4A_WIRED_SYNC_MODE_SUBORDINATE,
                subordinate_delay_us=subordinate_delay_us,
                synchronized_images_only=True,
            )

        self.start(
            master_id,
            fps=fps,
            color_width=color_width,
            color_height=color_height,
            depth_mode=depth_mode,
            wired_sync_mode=K4A_WIRED_SYNC_MODE_MASTER,
            synchronized_images_only=True,
        )

    def stop(self, device_id: str) -> None:
        # Hold the device lock across the full teardown: a concurrent start() for
        # the same device must wait until the old backend has actually released
        # its handle, or KinectBackend.start() hits k4a_device_open=1. Different
        # devices have different locks, so stop_all() still tears down in parallel.
        with self._dev_lock(device_id):
            worker = self._workers.pop(device_id, None)
            if worker:
                worker.stop()
                worker.join()

    def stop_all(self) -> None:
        with ThreadPoolExecutor() as executor:
            executor.map(self.stop, list(self._workers))

    # ── Frame access ──────────────────────────────────────────────────────────

    def nearest_frame(self, device_id: str, host_timestamp_us: int) -> Optional[Frame]:
        """Return the buffered frame from device_id nearest to host_timestamp_us."""
        worker = self._workers.get(device_id)
        return worker.nearest_to(host_timestamp_us) if worker else None

    def latest_frame(self, device_id: str) -> Optional[Frame]:
        worker = self._workers.get(device_id)
        return worker.latest() if worker else None

    def get_backend(self, device_id: str) -> Optional[CameraBackend]:
        """Return the backend instance for the given device_id."""
        worker = self._workers.get(device_id)
        return worker.backend if worker else None

    def get_info(self, device_id: str) -> Optional[dict]:
        worker = self._workers.get(device_id)
        if not worker:
            return None
        frame = worker.latest()
        info: dict = {
            "device_id": device_id,
            "running": True,
            "has_frame": frame is not None,
            "config": worker.backend.config,  # what it was actually started with
        }
        if frame:
            info["color_shape"] = list(frame.color.shape)
            info["depth_shape"] = list(frame.depth.shape)
            info["timestamp_us"] = frame.timestamp_us
            if frame.color_intrinsics:
                ci = frame.color_intrinsics
                info["color_intrinsics"] = {
                    "fx": ci.fx,
                    "fy": ci.fy,
                    "cx": ci.cx,
                    "cy": ci.cy,
                    "width": ci.width,
                    "height": ci.height,
                }
        return info

    # ── Backend factory ───────────────────────────────────────────────────────

    @staticmethod
    def _make_backend(
        device_id: str,
        fps: int,
        color_width: int,
        color_height: int,
        depth_mode: str,
        **kwargs,
    ) -> CameraBackend:
        if device_id.startswith("kinect_"):
            from .kinect import KinectBackend

            idx = int(device_id.split("_")[1])
            return KinectBackend(
                device_index=idx,
                color_resolution=(color_width, color_height),
                depth_mode=depth_mode,
                fps=fps,
                **kwargs,
            )
        else:
            from .realsense import RealSenseBackend

            return RealSenseBackend(
                serial=device_id,
                color_resolution=(color_width, color_height),
                depth_resolution=(color_width, color_height),
                fps=fps,
            )
