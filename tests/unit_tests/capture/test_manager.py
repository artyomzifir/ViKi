"""
Tests for CameraManager and _CameraWorker.
Verifies device lifecycle, frame buffering, and timestamp-based retrieval.
"""

import pytest

# ...
from unittest.mock import MagicMock, patch
import numpy as np
from viki.capture.manager import CameraManager
from viki.capture.base import CameraBackend, Frame


class MockBackend(CameraBackend):
    def __init__(self, device_id="mock_dev"):
        self._device_id = device_id
        self._is_running = False
        self.last_frame: Frame | None = None

    @property
    def device_id(self) -> str:
        return self._device_id

    def start(self):
        self._is_running = True

    def stop(self):
        self._is_running = False

    def get_frame(self) -> Frame:
        if not self._is_running:
            raise RuntimeError("Backend not started")
        return self.last_frame or Frame(
            color=np.zeros((480, 640, 3), dtype=np.uint8),
            depth=np.zeros((480, 640), dtype=np.uint16),
            timestamp_us=1000,
            device_id=self.device_id,
        )

    @property
    def is_running(self) -> bool:
        return self._is_running


def test_manager_start_stop():
    """Verify that cameras can be started and stopped correctly."""
    manager = CameraManager()

    # Mock _make_backend to return our MockBackend
    with patch.object(
        CameraManager, "_make_backend", return_value=MockBackend("mock_dev")
    ):
        manager.start("mock_dev")
        assert "mock_dev" in manager.active_device_ids()

        manager.stop("mock_dev")
        assert "mock_dev" not in manager.active_device_ids()


def test_manager_latest_frame():
    """Verify that the manager returns the most recent frame from the worker's buffer."""
    manager = CameraManager()
    mock_backend = MockBackend("mock_dev")

    # Setup a specific frame
    frame = Frame(
        color=np.ones((480, 640, 3), dtype=np.uint8),
        depth=np.ones((480, 640), dtype=np.uint16),
        timestamp_us=5000,
        device_id="mock_dev",
    )
    mock_backend.last_frame = frame

    with patch.object(CameraManager, "_make_backend", return_value=mock_backend):
        manager.start("mock_dev")
        # The worker thread will now be calling get_frame() in a loop
        # We need to wait a bit for the first frame to be buffered
        import time

        time.sleep(0.1)

        latest = manager.latest_frame("mock_dev")
        assert latest is not None
        assert np.array_equal(latest.color, frame.color)


def test_manager_nearest_frame():
    """Verify that the manager finds the frame closest to a given host timestamp."""
    manager = CameraManager()
    mock_backend = MockBackend("mock_dev")

    # We'll manually populate the worker's buffer to avoid timing issues with threads
    with patch.object(CameraManager, "_make_backend", return_value=mock_backend):
        manager.start("mock_dev")
        worker = manager._workers["mock_dev"]

        f1 = Frame(
            color=np.zeros((1, 1, 3), dtype=np.uint8),
            depth=np.zeros((1, 1), dtype=np.uint16),
            timestamp_us=1000,
            device_id="mock_dev",
            host_timestamp_us=1000,
        )
        f2 = Frame(
            color=np.zeros((1, 1, 3), dtype=np.uint8),
            depth=np.zeros((1, 1), dtype=np.uint16),
            timestamp_us=2000,
            device_id="mock_dev",
            host_timestamp_us=2000,
        )
        f3 = Frame(
            color=np.zeros((1, 1, 3), dtype=np.uint8),
            depth=np.zeros((1, 1), dtype=np.uint16),
            timestamp_us=3000,
            device_id="mock_dev",
            host_timestamp_us=3000,
        )

        # Manually add to buffer
        worker._buffer.append(f1)
        worker._buffer.append(f2)
        worker._buffer.append(f3)

        assert manager.nearest_frame("mock_dev", 1100).timestamp_us == f1.timestamp_us
        assert manager.nearest_frame("mock_dev", 1900).timestamp_us == f2.timestamp_us
        assert manager.nearest_frame("mock_dev", 2900).timestamp_us == f3.timestamp_us
        assert manager.nearest_frame("mock_dev", 5000).timestamp_us == f3.timestamp_us

        manager.stop("mock_dev")
