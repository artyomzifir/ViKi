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
from viki.cameras.hw_sync import (
    HardwareSyncError,
    SyncRole,
    WIRED_MASTER,
    WIRED_STANDALONE,
    WIRED_SUBORDINATE,
    build_sync_plan,
)
from viki.config import (
    FRAME_BUFFER_SIZE,
    DEFAULT_FPS,
    DEFAULT_COLOR_WIDTH,
    DEFAULT_COLOR_HEIGHT,
    DEFAULT_DEPTH_MODE,
)

# Frames kept per camera for timestamp-based sync queries.
_FRAME_BUFFER_SIZE = FRAME_BUFFER_SIZE
_HW_SYNC_TIMESTAMP_TOLERANCE_US = 500


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

    def snapshot(self) -> list[Frame]:
        """Stable copy of buffered frames for cross-device timestamp checks."""
        with self._lock:
            return list(self._buffer)

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
        # A Kinect rig is one lifecycle unit: subordinate(s) must be started
        # before the master, and a partial failure must leave no half-live rig.
        self._kinect_rig_lock = threading.RLock()

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

        try:
            devices["hardware_sync"] = self.hardware_sync_status(
                detected_ids=devices["kinect"]
            )
        except Exception as exc:  # discovery must still return the devices
            devices["hardware_sync"] = {
                "required": len(devices["kinect"]) >= 2,
                "ready": False,
                "error": str(exc),
            }

        return devices

    @staticmethod
    def _detected_kinect_ids() -> list[str]:
        from .kinect import KinectBackend

        return [f"kinect_{i}" for i in range(KinectBackend.device_count())]

    def kinect_sync_plan(
        self, detected_ids: list[str] | None = None,
    ) -> dict[str, SyncRole]:
        """Validated role assignment for the currently connected Kinect rig."""
        from viki import config

        detected = (
            list(detected_ids)
            if detected_ids is not None
            else self._detected_kinect_ids()
        )
        return build_sync_plan(detected, getattr(config, "KINECT_SYNC", {}) or {})

    def hardware_sync_status(
        self, detected_ids: list[str] | None = None,
    ) -> dict[str, object]:
        """Describe whether a multi-Kinect rig is completely HW-synchronised."""
        detected = (
            list(detected_ids)
            if detected_ids is not None
            else self._detected_kinect_ids()
        )
        required = len(detected) >= 2
        try:
            plan = self.kinect_sync_plan(detected)
        except HardwareSyncError as exc:
            return {
                "required": required,
                "ready": False,
                "detected": sorted(detected),
                "roles": {},
                "error": str(exc),
            }

        roles = {
            device_id: {"role": role.name, "mode": role.mode, "delay_us": role.delay_us}
            for device_id, role in plan.items()
        }
        if not required:
            return {
                "required": False,
                "ready": True,
                "detected": sorted(detected),
                "roles": roles,
            }

        active = set(self.active_device_ids())
        missing = sorted(set(plan) - active)
        problems: list[str] = []
        if missing:
            problems.append("not running: " + ", ".join(missing))
        for device_id, role in plan.items():
            backend = self.get_backend(device_id)
            if backend is None:
                continue
            try:
                jack_reader = getattr(backend, "get_sync_jack_status", None)
                jack = jack_reader() if jack_reader is not None else {}
            except Exception as exc:  # a live cable read is part of readiness
                problems.append(f"{device_id} sync-jack check failed: {exc}")
                jack = {}
            cfg = backend.config or {}
            if int(cfg.get("wired_sync_mode", WIRED_STANDALONE)) != role.mode:
                problems.append(f"{device_id} is not running as {role.name}")
            if not bool(cfg.get("synchronized_images_only", False)):
                problems.append(f"{device_id} synchronized_images_only is disabled")
            sync_in = jack.get("sync_in_connected", cfg.get("sync_in_connected", False))
            sync_out = jack.get("sync_out_connected", cfg.get("sync_out_connected", False))
            if role.mode == WIRED_MASTER and not bool(sync_out):
                problems.append(f"{device_id} SYNC OUT is disconnected")
            if role.mode == WIRED_SUBORDINATE and not bool(sync_in):
                problems.append(f"{device_id} SYNC IN is disconnected")
        alignment = self._hardware_timestamp_alignment(plan)
        if not alignment["verified"]:
            problems.append(str(alignment["error"]))
        return {
            "required": True,
            "ready": not problems,
            "detected": sorted(detected),
            "active": sorted(active & set(plan)),
            "roles": roles,
            "timestamp_alignment": alignment,
            "error": "; ".join(problems) if problems else None,
        }

    def _hardware_timestamp_alignment(
        self, plan: dict[str, SyncRole],
    ) -> dict[str, object]:
        """Verify the wire on actual K4A device timestamps, not host arrival."""
        master_id = next(
            (device_id for device_id, role in plan.items()
             if role.mode == WIRED_MASTER),
            None,
        )
        if master_id is None:
            return {"verified": False, "error": "HW_SYNC master is missing"}
        master_worker = self._workers.get(master_id)
        master_frames = master_worker.snapshot() if master_worker else []
        if not master_frames:
            return {
                "verified": False,
                "tolerance_us": _HW_SYNC_TIMESTAMP_TOLERANCE_US,
                "error": f"{master_id} has no frame for HW timestamp verification",
            }

        offsets: dict[str, dict[str, int]] = {}
        for subordinate_id, role in plan.items():
            if role.mode != WIRED_SUBORDINATE:
                continue
            worker = self._workers.get(subordinate_id)
            subordinate_frames = worker.snapshot() if worker else []
            if not subordinate_frames:
                return {
                    "verified": False,
                    "tolerance_us": _HW_SYNC_TIMESTAMP_TOLERANCE_US,
                    "offsets": offsets,
                    "error": (
                        f"{subordinate_id} has no frame for HW timestamp verification"
                    ),
                }
            candidates = (
                (
                    int(sub.timestamp_us) - int(master.timestamp_us),
                    abs(
                        (int(sub.timestamp_us) - int(master.timestamp_us))
                        - role.delay_us
                    ),
                )
                for master in master_frames
                for sub in subordinate_frames
            )
            actual, residual = min(candidates, key=lambda pair: pair[1])
            offsets[subordinate_id] = {
                "actual_us": int(actual),
                "expected_us": int(role.delay_us),
                "residual_us": int(residual),
            }
            if residual > _HW_SYNC_TIMESTAMP_TOLERANCE_US:
                return {
                    "verified": False,
                    "tolerance_us": _HW_SYNC_TIMESTAMP_TOLERANCE_US,
                    "offsets": offsets,
                    "error": (
                        f"{subordinate_id} HW timestamp residual {residual}us exceeds "
                        f"{_HW_SYNC_TIMESTAMP_TOLERANCE_US}us"
                    ),
                }
        return {
            "verified": True,
            "tolerance_us": _HW_SYNC_TIMESTAMP_TOLERANCE_US,
            "offsets": offsets,
            "error": None,
        }

    def require_hardware_sync_ready(self) -> dict[str, object]:
        """Fail unless every connected Kinect in a multi-device rig is synced."""
        status = self.hardware_sync_status()
        if status["required"] and not status["ready"]:
            raise HardwareSyncError(
                "Kinect HW_SYNC rig is not ready: " + str(status.get("error") or "unknown error")
            )
        return status

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
        is_kinect = device_id.startswith("kinect_")
        if is_kinect:
            plan = self.kinect_sync_plan()
            if len(plan) >= 2:
                if device_id not in plan:
                    raise HardwareSyncError(
                        f"{device_id} is not part of the configured Kinect HW_SYNC rig"
                    )
                role = plan[device_id]
                requested_mode = int(kwargs.get("wired_sync_mode", WIRED_STANDALONE))
                if requested_mode not in (WIRED_STANDALONE, role.mode):
                    raise HardwareSyncError(
                        f"{device_id} must run as {role.name}; requested wired_sync_mode="
                        f"{requested_mode}"
                    )
                requested_delay = int(kwargs.get("subordinate_delay_us", 0))
                if requested_delay not in (0, role.delay_us):
                    raise HardwareSyncError(
                        f"{device_id} must use subordinate_delay_us={role.delay_us}; "
                        f"requested {requested_delay}"
                    )
                if kwargs.get("synchronized_images_only") is False:
                    raise HardwareSyncError(
                        "multi-Kinect HW_SYNC requires synchronized_images_only=true"
                    )
                # Zero is the public API's "unspecified" default, never an
                # opt-out when multiple Kinects are connected.
                kwargs["wired_sync_mode"] = role.mode
                kwargs["subordinate_delay_us"] = role.delay_us
                kwargs["synchronized_images_only"] = True

        want = {
            "color_width": int(color_width),
            "color_height": int(color_height),
            "fps": int(fps),
            "depth_mode": depth_mode,
        }
        if is_kinect:
            want.update({
                "wired_sync_mode": int(kwargs.get("wired_sync_mode", WIRED_STANDALONE)),
                "subordinate_delay_us": int(kwargs.get("subordinate_delay_us", 0)),
                "synchronized_images_only": bool(
                    kwargs.get("synchronized_images_only", True)
                ),
            })
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
    ) -> dict[str, str]:
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
        with self._kinect_rig_lock:
            plan = self.kinect_sync_plan()
            configured_master = next(
                (device_id for device_id, role in plan.items()
                 if role.mode == WIRED_MASTER),
                None,
            )
            configured_subordinates = {
                device_id for device_id, role in plan.items()
                if role.mode == WIRED_SUBORDINATE
            }
            if configured_master is None:
                raise HardwareSyncError(
                    "start_kinect_sync requires at least two connected Kinects"
                )
            if master_id != configured_master or set(subordinate_ids) != configured_subordinates:
                raise HardwareSyncError(
                    "requested Kinect roles differ from the validated KINECT_SYNC plan"
                )
            expected_delay = next(iter(configured_subordinates), None)
            if expected_delay is not None:
                expected_delay = plan[expected_delay].delay_us
                if int(subordinate_delay_us) != expected_delay:
                    raise HardwareSyncError(
                        f"configured subordinate_delay_us is {expected_delay}, "
                        f"requested {subordinate_delay_us}"
                    )

            # If the entire rig already matches, do not disturb a live stream.
            status = self.hardware_sync_status(list(plan))
            desired = {
                "fps": int(fps), "color_width": int(color_width),
                "color_height": int(color_height), "depth_mode": depth_mode,
            }
            configs_match = status["ready"] and all(
                all(self.get_backend(device_id).config.get(key) == value
                    for key, value in desired.items())
                for device_id in plan
            )
            if configs_match:
                return {device_id: "unchanged" for device_id in plan}

            # A restart is one transaction. Stop the master first, then all
            # subordinates; start in the inverse order. Any failure tears down
            # the whole rig so no camera remains usable in a partial state.
            was_active = set(self.active_device_ids())
            self.stop(master_id)
            for sub_id in subordinate_ids:
                self.stop(sub_id)

            outcomes: dict[str, str] = {}
            try:
                for sub_id in subordinate_ids:
                    self.start(
                        sub_id,
                        fps=fps,
                        color_width=color_width,
                        color_height=color_height,
                        depth_mode=depth_mode,
                        wired_sync_mode=WIRED_SUBORDINATE,
                        subordinate_delay_us=plan[sub_id].delay_us,
                        synchronized_images_only=True,
                    )
                    outcomes[sub_id] = "restarted" if sub_id in was_active else "started"

                self.start(
                    master_id,
                    fps=fps,
                    color_width=color_width,
                    color_height=color_height,
                    depth_mode=depth_mode,
                    wired_sync_mode=WIRED_MASTER,
                    synchronized_images_only=True,
                )
                outcomes[master_id] = "restarted" if master_id in was_active else "started"
                deadline = time.monotonic() + 5.0
                status = self.hardware_sync_status()
                while not status["ready"] and time.monotonic() < deadline:
                    time.sleep(0.05)
                    status = self.hardware_sync_status()
                if not status["ready"]:
                    raise HardwareSyncError(
                        "Kinect HW_SYNC did not verify after startup: "
                        + str(status.get("error") or "unknown error")
                    )
                return outcomes
            except Exception:
                self.stop(master_id)
                for sub_id in subordinate_ids:
                    self.stop(sub_id)
                raise

    def start_configured_kinect_rig(
        self,
        fps: int = DEFAULT_FPS,
        color_width: int = DEFAULT_COLOR_WIDTH,
        color_height: int = DEFAULT_COLOR_HEIGHT,
        depth_mode: str = DEFAULT_DEPTH_MODE,
    ) -> dict[str, str]:
        """Start the detected rig from KINECT_SYNC, subordinate(s) first."""
        plan = self.kinect_sync_plan()
        if len(plan) < 2:
            if not plan:
                raise HardwareSyncError("no Kinect devices detected")
            device_id = next(iter(plan))
            return {device_id: self.start(
                device_id, fps=fps, color_width=color_width,
                color_height=color_height, depth_mode=depth_mode,
            )}
        master = next(
            device_id for device_id, role in plan.items() if role.mode == WIRED_MASTER
        )
        subordinates = [
            device_id for device_id, role in plan.items()
            if role.mode == WIRED_SUBORDINATE
        ]
        delay = plan[subordinates[0]].delay_us
        return self.start_kinect_sync(
            master, subordinates, fps=fps, color_width=color_width,
            color_height=color_height, depth_mode=depth_mode,
            subordinate_delay_us=delay,
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

            # Record raw at the backend's default depth res (848x480, the D4xx
            # sweet spot) — the colour↔depth registration is replayed offline
            # from raw/<dev>_rs_calib.json, like the Kinect's k4a blob. Don't
            # mirror the colour resolution here.
            return RealSenseBackend(
                serial=device_id,
                color_resolution=(color_width, color_height),
                fps=fps,
            )
