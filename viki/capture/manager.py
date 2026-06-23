"""
viki.capture.manager
--------------------
CameraManager: detects, starts, and owns camera backends.
Each camera runs in a background thread; the latest frame is always
available for non-blocking reads by the MJPEG streamer.
"""
from __future__ import annotations

import json
import shutil
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from .base import CameraBackend, CameraIntrinsics, Frame

# Frames kept per camera for timestamp-based sync queries.
_FRAME_BUFFER_SIZE = 2


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
            return min(self._buffer, key=lambda f: abs(f.host_timestamp_us - host_timestamp_us))

    def _loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                try:
                    frame = self.backend.get_frame()
                    frame.host_timestamp_us = time.time_ns() // 1000
                    with self._lock:
                        self._buffer.append(frame)
                except TimeoutError:
                    pass  # short timeout, check stop_event and retry
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
        fps: int = 30,
        color_width: int = 640,
        color_height: int = 480,
        depth_mode: str = "NFOV_UNBINNED",
        **kwargs,
    ) -> None:
        if device_id in self._workers:
            return  # already running

        backend = self._make_backend(device_id, fps, color_width, color_height, depth_mode, **kwargs)
        worker = _CameraWorker(backend)
        worker.start()
        self._workers[device_id] = worker

    def start_kinect_sync(
        self,
        master_id: str,
        subordinate_ids: list,
        fps: int = 30,
        color_width: int = 1280,
        color_height: int = 720,
        depth_mode: str = "NFOV_UNBINNED",
        subordinate_delay_us: int = 0,
        align_depth_to_color: bool = False,
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
                align_depth_to_color=align_depth_to_color,
            )

        self.start(
            master_id,
            fps=fps,
            color_width=color_width,
            color_height=color_height,
            depth_mode=depth_mode,
            wired_sync_mode=K4A_WIRED_SYNC_MODE_MASTER,
            synchronized_images_only=True,
            align_depth_to_color=align_depth_to_color,
        )

    def stop(self, device_id: str) -> None:
        worker = self._workers.pop(device_id, None)
        if worker:
            worker.stop()

    def stop_all(self) -> None:
        workers = [self._workers.pop(did) for did in list(self._workers)]
        for w in workers:
            w.stop()   # signal all first (non-blocking)
        for w in workers:
            w.join()   # then wait for all backend cleanup to finish

    # ── Frame access ──────────────────────────────────────────────────────────

    def nearest_frame(self, device_id: str, host_timestamp_us: int) -> Optional[Frame]:
        """Return the buffered frame from device_id nearest to host_timestamp_us."""
        worker = self._workers.get(device_id)
        return worker.nearest_to(host_timestamp_us) if worker else None

    def latest_frame(self, device_id: str) -> Optional[Frame]:
        worker = self._workers.get(device_id)
        return worker.latest() if worker else None

    def get_info(self, device_id: str) -> Optional[dict]:
        worker = self._workers.get(device_id)
        if not worker:
            return None
        frame = worker.latest()
        info: dict = {
            "device_id": device_id,
            "running": True,
            "has_frame": frame is not None,
        }
        serial = getattr(worker.backend, "serial_number", None)
        if serial:
            info["serial_number"] = serial
        align_depth = getattr(worker.backend, "align_depth_to_color", None)
        if align_depth is not None:
            info["align_depth_to_color"] = bool(align_depth)
        color_format = getattr(worker.backend, "color_format", None)
        if color_format:
            info["color_format"] = color_format
        depth_mode = getattr(worker.backend, "depth_mode", None)
        if depth_mode:
            info["depth_mode"] = depth_mode
            info["requested_depth_mode"] = depth_mode
        sdk_depth_mode = getattr(worker.backend, "sdk_depth_mode", None)
        if sdk_depth_mode is not None:
            info["actual_depth_mode"] = sdk_depth_mode
            info["sdk_depth_mode"] = sdk_depth_mode
        expected_raw_depth_resolution = getattr(
            worker.backend,
            "expected_raw_depth_resolution",
            None,
        )
        if expected_raw_depth_resolution:
            ew, eh = expected_raw_depth_resolution
            info["expected_raw_depth_shape"] = [eh, ew]
        color_intrinsics = self._serialize_intrinsics(
            getattr(worker.backend, "color_intrinsics", None)
        )
        if color_intrinsics:
            info["color_intrinsics"] = color_intrinsics
        if frame:
            color_shape = frame.color_shape or frame.color.shape
            info["color_shape"] = list(color_shape)
            info["depth_shape"] = list(frame.depth.shape)
            info["depth_is_aligned"] = bool(frame.depth_is_aligned)
            info["color_jpeg"] = frame.color_jpeg is not None
            info["timestamp_us"] = frame.timestamp_us
            color_intrinsics = self._serialize_intrinsics(frame.color_intrinsics)
            if color_intrinsics:
                info["color_intrinsics"] = color_intrinsics
        return info

    # ── Snapshot capture ─────────────────────────────────────────────────────

    def snapshot(
        self,
        device_id: str,
        aligned_depth: bool = True,
        save: bool = True,
        snapshot_id: Optional[str] = None,
        root_dir: str | Path = "data/snapshots",
    ) -> dict:
        worker = self._workers.get(device_id)
        if not worker:
            raise RuntimeError(f"Camera {device_id} is not running")
        frame = worker.latest()
        if frame is None:
            raise RuntimeError(f"Camera {device_id} has no frame yet")

        timestamp = snapshot_id or self._snapshot_timestamp()
        target_dir = Path(root_dir) / timestamp / device_id
        return self._save_snapshot(
            device_id=device_id,
            worker=worker,
            frame=frame,
            target_dir=target_dir,
            aligned_depth=aligned_depth,
            save=save,
        )

    def pair_snapshot(
        self,
        device_ids: list[str],
        aligned_depth: bool = True,
        save: bool = True,
        root_dir: str | Path = "data/snapshots",
    ) -> dict:
        timestamp = self._snapshot_timestamp(prefix="pair")
        root_path = Path(root_dir) / timestamp
        snapshots = []
        try:
            for device_id in device_ids:
                snapshots.append(
                    self.snapshot(
                        device_id,
                        aligned_depth=aligned_depth,
                        save=save,
                        snapshot_id=timestamp,
                        root_dir=root_dir,
                    )
                )
        except Exception:
            if save and root_path.exists():
                shutil.rmtree(root_path, ignore_errors=True)
            raise
        return {
            "snapshot_id": timestamp,
            "root": str(root_path),
            "device_ids": device_ids,
            "snapshots": snapshots,
        }

    @staticmethod
    def _snapshot_timestamp(prefix: str = "snapshot") -> str:
        return f"{prefix}_{datetime.now().strftime('%Y%m%d_%H%M%S_%f')}"

    def _save_snapshot(
        self,
        device_id: str,
        worker: _CameraWorker,
        frame: Frame,
        target_dir: Path,
        aligned_depth: bool,
        save: bool,
    ) -> dict:
        raw_depth = self._raw_depth_from_frame(frame)
        aligned = None
        if aligned_depth:
            if frame.depth_is_aligned:
                aligned = frame.depth
            elif hasattr(worker.backend, "align_depth_snapshot"):
                aligned = worker.backend.align_depth_snapshot(raw_depth)
            else:
                raise RuntimeError(f"Backend for {device_id} cannot align depth snapshots")

        color_shape = frame.color_shape or frame.color.shape
        metadata = {
            "device_id": device_id,
            "serial_number": getattr(worker.backend, "serial_number", frame.device_id),
            "timestamp_us": frame.timestamp_us,
            "host_timestamp_us": frame.host_timestamp_us,
            "color_shape": list(color_shape),
            "raw_depth_shape": list(raw_depth.shape),
            "aligned_depth_shape": list(aligned.shape) if aligned is not None else None,
            "depth_shape": list(frame.depth.shape),
            "depth_is_aligned": bool(frame.depth_is_aligned),
            "depth_mode": getattr(worker.backend, "depth_mode", None),
            "sdk_depth_mode": getattr(worker.backend, "sdk_depth_mode", None),
            "color_resolution": list(getattr(worker.backend, "color_resolution", (color_shape[1], color_shape[0]))),
            "color_format": getattr(worker.backend, "color_format", None),
            "align_depth_requested": bool(aligned_depth),
        }
        color_intrinsics = self._serialize_intrinsics(
            frame.color_intrinsics or getattr(worker.backend, "color_intrinsics", None)
        )
        if color_intrinsics:
            metadata["color_intrinsics"] = color_intrinsics
        if hasattr(worker.backend, "snapshot_calibration_summary"):
            metadata["calibration"] = worker.backend.snapshot_calibration_summary()

        files: dict[str, str] = {}
        if save:
            target_dir.mkdir(parents=True, exist_ok=True)
            files.update(self._save_color(frame, target_dir))
            raw_npy = target_dir / "raw_depth.npy"
            np.save(raw_npy, raw_depth)
            files["raw_depth_npy"] = str(raw_npy)
            raw_png = target_dir / "raw_depth.png"
            if cv2.imwrite(str(raw_png), raw_depth):
                files["raw_depth_png"] = str(raw_png)

            if aligned is not None:
                aligned_npy = target_dir / "aligned_depth.npy"
                np.save(aligned_npy, aligned)
                files["aligned_depth_npy"] = str(aligned_npy)
                aligned_png = target_dir / "aligned_depth.png"
                if cv2.imwrite(str(aligned_png), aligned):
                    files["aligned_depth_png"] = str(aligned_png)

            metadata_path = target_dir / "metadata.json"
            metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            files["metadata"] = str(metadata_path)

        return {
            "status": "saved" if save else "captured",
            "device_id": device_id,
            "path": str(target_dir) if save else None,
            "files": files,
            "metadata": metadata,
        }

    @staticmethod
    def _raw_depth_from_frame(frame: Frame) -> np.ndarray:
        if frame.raw_depth is not None:
            return frame.raw_depth
        if not frame.depth_is_aligned:
            return frame.depth
        raise RuntimeError("Frame has aligned depth but no raw_depth copy")

    @staticmethod
    def _save_color(frame: Frame, target_dir: Path) -> dict[str, str]:
        if frame.color_jpeg is not None:
            path = target_dir / "color.jpg"
            path.write_bytes(frame.color_jpeg)
            return {"color": str(path)}

        path = target_dir / "color.jpg"
        if not cv2.imwrite(str(path), frame.color):
            raise RuntimeError(f"Failed to write {path}")
        return {"color": str(path)}

    @staticmethod
    def _serialize_intrinsics(intrinsics: Optional[CameraIntrinsics]) -> Optional[dict]:
        if intrinsics is None:
            return None

        dist = intrinsics.dist_coeffs
        if dist is None:
            dist_coeffs = []
        elif hasattr(dist, "tolist"):
            dist_coeffs = dist.tolist()
        else:
            dist_coeffs = list(dist)

        fx = float(intrinsics.fx)
        fy = float(intrinsics.fy)
        cx = float(intrinsics.cx)
        cy = float(intrinsics.cy)
        return {
            "fx": fx,
            "fy": fy,
            "cx": cx,
            "cy": cy,
            "width": int(intrinsics.width),
            "height": int(intrinsics.height),
            "dist_coeffs": [float(v) for v in dist_coeffs],
            "camera_matrix": [
                [fx, 0.0, cx],
                [0.0, fy, cy],
                [0.0, 0.0, 1.0],
            ],
        }

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
