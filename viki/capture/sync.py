"""
viki.capture.sync
-----------------
MultiCameraSync: software timestamp synchronisation across heterogeneous cameras.

How it works
------------
Each CameraWorker stamps frames with host_timestamp_us = time.time_ns() // 1000
as they arrive from the device.  The worker keeps a short rolling buffer of
recent frames (not just the latest), so frames at the sync tick that landed a
few milliseconds earlier are still reachable.

MultiCameraSync drives a loop at sync_fps (normally the rate of the slowest
camera).  At each tick it calls nearest_to(tick_us) on every active worker and
checks that the returned frame is within max_offset_us of the tick.  If every
required camera passes that check, a SyncedFrameGroup is emitted.

Azure Kinect hardware sync (wired master/subordinate) aligns Kinect captures
by hardware so both devices' frames arrive within ~1 ms of each other.
RealSense cameras are aligned to the host clock only — their grouping is
purely software-based.  At a shared sync_fps of 15 fps the tolerance window
is comfortably wide enough for any USB jitter in practice.

Startup order for hardware-synced Kinects
------------------------------------------
Always start the subordinate before the master:

    manager.start("kinect_1", wired_sync_mode=K4A_WIRED_SYNC_MODE_SUBORDINATE, ...)
    manager.start("kinect_0", wired_sync_mode=K4A_WIRED_SYNC_MODE_MASTER, ...)

The subordinate waits for trigger pulses; if the master fires first the sync
fails.  CameraManager.start_kinect_sync() enforces this order automatically.
"""

from __future__ import annotations

import time
import random
from typing import Callable, Optional

from .base import SyncedFrameGroup
from .manager import CameraManager


class MultiCameraSync:
    """
    Synchronises frames from all active cameras to host-clock ticks.

    Parameters
    ----------
    manager : CameraManager
        Running manager with one or more active cameras.
    sync_fps : int
        Output tick rate.  Must be <= the slowest camera's FPS or groups will
        be dropped whenever the slow camera hasn't produced a fresh frame.
    max_offset_us : int
        A frame is accepted if |frame.host_timestamp_us - tick_us| <= this value.
        Default: half a 30 fps frame (≈ 16.7 ms), which is conservative enough
        for USB delivery jitter at any supported frame rate.
    required_devices : list[str] | None
        Device IDs that must all have an in-tolerance frame for a group to be
        emitted.  None means all currently active devices are required.
    """

    def __init__(
        self,
        manager: CameraManager,
        sync_fps: int = 15,
        max_offset_us: int = 150000,
        required_devices: Optional[list] = None,
    ) -> None:
        self._manager = manager
        self._sync_fps = sync_fps
        self._max_offset_us = max_offset_us
        self._required_devices = required_devices

    def get_synced_frame(self) -> Optional[SyncedFrameGroup]:
        """
        Attempt to build one synchronised frame group at the current host time.
        
        Returns None if any required camera has no frame within the tolerance
        window (camera not yet started, stalled, or running at lower FPS than
        sync_fps).
        """
        import logging
        logger = logging.getLogger(__name__)
        tick_us = time.time_ns() // 1000
        device_ids = (
            self._required_devices
            if self._required_devices is not None
            else self._manager.active_device_ids()
        )

        frames: dict = {}
        offsets: dict = {}

        for dev_id in device_ids:
            frame = self._manager.nearest_frame(dev_id, tick_us)
            if frame is None:
                if random.random() < 0.01:
                    logger.warning(f"Sync: {dev_id} has no buffered frames")
                return None
            offset = frame.host_timestamp_us - tick_us
            if abs(offset) > self._max_offset_us:
                if random.random() < 0.01:
                    logger.warning(f"Sync: {dev_id} frame offset {offset}us exceeds tolerance {self._max_offset_us}us")
                return None
            frames[dev_id] = frame
            offsets[dev_id] = offset

        return SyncedFrameGroup(
            frames=frames,
            sync_timestamp_us=tick_us,
            offsets_us=offsets,
        )

    def record(
        self,
        duration_s: float,
        on_group: Optional[Callable] = None,
    ) -> list:
        """
        Collect synchronised frame groups for duration_s seconds.

        Parameters
        ----------
        duration_s : float
            Recording duration.
        on_group : callable | None
            Optional callback invoked with each SyncedFrameGroup as it arrives.
            Use this to write frames to disk while recording rather than
            accumulating everything in memory.

        Returns
        -------
        list[SyncedFrameGroup]
            All successfully synchronised groups in chronological order.
            Groups where any camera missed the tolerance window are silently
            skipped — check len(result) vs expected (duration_s * sync_fps).
        """
        groups: list = []
        period_s = 1.0 / self._sync_fps
        deadline = time.monotonic() + duration_s

        while time.monotonic() < deadline:
            tick_wall = time.monotonic()

            group = self.get_synced_frame()
            if group is not None:
                groups.append(group)
                if on_group is not None:
                    on_group(group)

            elapsed = time.monotonic() - tick_wall
            sleep_s = period_s - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)

        return groups
