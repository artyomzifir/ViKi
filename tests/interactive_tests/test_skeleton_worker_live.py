"""
Interactive live demo test for SkeletonWorker.

Opens a webcam via OpenCV, wraps it in mock CameraManager and MultiCameraSync,
and runs SkeletonWorker in the background. Displays a live preview window.

Controls:
    q / Esc — quit
    e       — toggle skeleton estimation
    r       — toggle recording
"""

from __future__ import annotations

import argparse
import time
from typing import Optional

import cv2
import numpy as np

from viki.capture.base import Frame, SyncedFrameGroup, CameraIntrinsics
from viki.calibration.manager import CalibrationManager
from viki.server.skeleton_worker import SkeletonWorker
from viki.skeleton.pipeline import SkeletonPipeline
from viki.skeleton.recorder import SkeletonRecorder
from viki.skeleton.models import LM, HandDetection


class MockDepthBackend:
    """Duck-typed backend for a single webcam (no separate depth sensor)."""

    @staticmethod
    def project_color_to_depth(u: float, v: float, z: float) -> tuple[float, float]:
        """Identity mapping — same pixel for colour and depth."""
        return (float(u), float(v))


class MockCameraManager:
    """
    Fake CameraManager that reads from an OpenCV webcam.

    Only the worker thread calls ``nearest_frame`` (reads + caches a frame).
    The display reads from that cache via ``latest_frame`` so both see the
    same frame, eliminating the skeleton-vs-display desync.
    """

    def __init__(self, camera_id: int = 0, width: int = 640, height: int = 480):
        self._cap = cv2.VideoCapture(camera_id)
        self._cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        self._cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if not self._cap.isOpened():
            raise RuntimeError(f"Cannot open camera {camera_id}")
        ret, bgr = self._cap.read()
        if not ret:
            raise RuntimeError("Cannot read from camera")
        self._h, self._w = bgr.shape[:2]
        self._device_id = "webcam_0"
        self._started = True
        self._cached_frame: Optional[Frame] = None

    def list_devices(self) -> dict:
        return {self._device_id: {"type": "webcam"}}

    def active_device_ids(self) -> list:
        return [self._device_id] if self._started else []

    def start(self, device_id: str, **kwargs) -> None:
        self._started = True

    def stop(self, device_id: str) -> None:
        self._started = False

    def stop_all(self) -> None:
        self._started = False

    def nearest_frame(self, device_id: str, host_timestamp_us: int) -> Optional[Frame]:
        ret, bgr = self._cap.read()
        if not ret:
            return None
        frame = Frame(
            color=bgr,
            depth=np.full((self._h, self._w), 700, dtype=np.uint16),
            timestamp_us=host_timestamp_us,
            device_id=device_id,
            host_timestamp_us=host_timestamp_us,
            color_intrinsics=CameraIntrinsics(
                fx=self._w * 0.8, fy=self._w * 0.8,
                cx=self._w / 2, cy=self._h / 2,
                width=self._w, height=self._h,
            ),
            depth_intrinsics=CameraIntrinsics(
                fx=self._w * 0.8, fy=self._w * 0.8,
                cx=self._w / 2, cy=self._h / 2,
                width=self._w, height=self._h,
            ),
        )
        self._cached_frame = frame
        return frame

    def latest_frame(self, device_id: str) -> Optional[Frame]:
        return self._cached_frame

    def get_info(self, device_id: str) -> Optional[dict]:
        return {
            "device_id": device_id,
            "color_width": self._w,
            "color_height": self._h,
        }

    def get_backend(self, device_id: str):
        return MockDepthBackend()


class MockMultiCameraSync:
    """Fake MultiCameraSync that returns SyncedFrameGroup from MockCameraManager."""

    def __init__(self, manager: MockCameraManager):
        self._manager = manager

    def get_synced_frame(self) -> Optional[SyncedFrameGroup]:
        dev_ids = self._manager.active_device_ids()
        if not dev_ids:
            return None
        tick_us = time.time_ns() // 1000
        frames: dict[str, Frame] = {}
        offsets: dict[str, int] = {}
        for dev_id in dev_ids:
            frame = self._manager.nearest_frame(dev_id, tick_us)
            if frame is None:
                return None
            frames[dev_id] = frame
            offsets[dev_id] = 0
        return SyncedFrameGroup(
            frames=frames,
            sync_timestamp_us=tick_us,
            offsets_us=offsets,
        )


# Finger chains for drawing skeleton overlay
CHAINS = [
    [LM.WRIST, LM.THUMB_CMC, LM.THUMB_MCP, LM.THUMB_IP, LM.THUMB_TIP],
    [LM.WRIST, LM.INDEX_MCP, LM.INDEX_PIP, LM.INDEX_DIP, LM.INDEX_TIP],
    [LM.WRIST, LM.MIDDLE_MCP, LM.MIDDLE_PIP, LM.MIDDLE_DIP, LM.MIDDLE_TIP],
    [LM.WRIST, LM.RING_MCP, LM.RING_PIP, LM.RING_DIP, LM.RING_TIP],
    [LM.WRIST, LM.PINKY_MCP, LM.PINKY_PIP, LM.PINKY_DIP, LM.PINKY_TIP],
    [LM.SHOULDER, LM.ELBOW, LM.WRIST],
]

CHAIN_COLORS = [
    (0, 165, 255),    # thumb   — orange
    (0, 255, 0),      # index   — green
    (255, 255, 0),    # middle  — yellow
    (255, 0, 255),    # ring    — magenta
    (0, 255, 255),    # pinky   — cyan
    (255, 100, 100),  # arm     — light blue
]


def draw_skeleton(img: np.ndarray, detection: HandDetection) -> np.ndarray:
    """Draw 2D skeleton overlay from HandDetection pixel coords."""
    display = img.copy()
    px = np.full((LM.N, 2), np.nan, dtype=np.float32)
    for landmark, point in detection.points.items():
        px[int(landmark)] = point

    for chain, color in zip(CHAINS, CHAIN_COLORS):
        pts = px[chain]
        if np.isnan(pts).any():
            continue
        for i in range(len(pts) - 1):
            p1 = tuple(pts[i].astype(int))
            p2 = tuple(pts[i + 1].astype(int))
            cv2.line(display, p1, p2, color, 2, cv2.LINE_AA)

    for i, (u, v) in enumerate(px):
        if np.isnan(u) or np.isnan(v):
            continue
        cv2.circle(display, (int(u), int(v)), 4, (255, 255, 255), -1)
        cv2.circle(display, (int(u), int(v)), 4, (0, 0, 0), 1)

    return display


def main():
    parser = argparse.ArgumentParser(
        description="Live SkeletonWorker demo with webcam"
    )
    parser.add_argument("--camera", type=int, default=0, help="OpenCV camera index")
    parser.add_argument("--width", type=int, default=640, help="Capture width")
    parser.add_argument("--height", type=int, default=480, help="Capture height")
    parser.add_argument("--fps", type=float, default=15.0, help="Worker target FPS")
    args = parser.parse_args()

    mock_manager = MockCameraManager(args.camera, args.width, args.height)
    mock_sync = MockMultiCameraSync(mock_manager)

    calibrator = CalibrationManager(mock_manager)
    pipeline = SkeletonPipeline(calibrator, mock_manager)
    recorder = SkeletonRecorder()

    worker = SkeletonWorker(
        manager=mock_manager,
        sync=mock_sync,
        pipeline=pipeline,
        recorder=recorder,
        target_fps=args.fps,
    )
    worker.start()

    print("Controls: q/Esc=quit  e=toggle skeleton  r=toggle recording")
    print(f"Webcam: {args.camera} ({mock_manager._w}x{mock_manager._h})")

    frame_count = 0
    fps_start = time.perf_counter()
    display_fps = 0.0

    try:
        while True:
            frame = mock_manager.latest_frame("webcam_0")
            if frame is None:
                time.sleep(0.01)
                continue

            display = frame.color.copy()

            detections = worker.get_latest_detections()
            detection = detections.get("webcam_0")
            if detection is not None:
                display = draw_skeleton(display, detection)
                cv2.putText(
                    display, f"conf={detection.confidence:.2f}",
                    (10, display.shape[0] - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
                )

            frame_count += 1
            now = time.perf_counter()
            elapsed = now - fps_start
            if elapsed >= 1.0:
                display_fps = frame_count / elapsed
                frame_count = 0
                fps_start = now

            lines = [
                f"FPS: {display_fps:.1f}",
                f"Skeleton: {'ON' if worker.is_enabled else 'OFF'}",
                f"Recording: {'ON' if worker.is_recording else 'OFF'}",
            ]
            for i, line in enumerate(lines):
                cv2.putText(
                    display, line, (10, 24 + i * 24),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2,
                )

            cv2.imshow("SkeletonWorker Live Demo", display)
            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
            elif key == ord("e"):
                new_state = not worker.is_enabled
                worker.set_enabled(new_state)
                print(f"Skeleton {'enabled' if new_state else 'disabled'}")
            elif key == ord("r"):
                new_state = not worker.is_recording
                worker.set_recording(new_state)
                print(f"Recording {'started' if new_state else 'stopped'}")
    finally:
        worker.stop()
        mock_manager.stop_all()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
