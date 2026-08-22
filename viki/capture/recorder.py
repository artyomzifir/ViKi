import os
import json
import time
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from .base import SyncedFrameGroup
from .sync import MultiCameraSync
from viki.calibration.file import read_device_intrinsics, read_device_extrinsics
from viki.viz.depth import DepthColorizer
import viki.config

class RGBDRecorder:
    """
    Records synchronized RGB-D frames from multiple cameras to disk.
    Saves RGB and colorized Depth as MP4 videos, and raw Depth as .npy files.
    """

    def __init__(self, manager, output_base_dir: str = "data/videos"):
        self.manager = manager
        self.output_base_dir = output_base_dir
        self.current_recording_dir: Optional[str] = self._setup_recording_dir()
        self.frame_idx = 0
        self.timestamps = []
        self._writers: dict[str, dict] = {} # dev_id -> {"color": VideoWriter, "depth": VideoWriter}
        self._colorizers: dict[str, DepthColorizer] = {}

    def _setup_recording_dir(self) -> str:
        # Create a unique directory for this recording
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        recording_dir = os.path.join(self.output_base_dir, f"rec_{timestamp}")
        os.makedirs(recording_dir, exist_ok=True)

        # Setup per-camera directories
        active_ids = self.manager.active_device_ids()
        for dev_id in active_ids:
            cam_dir = os.path.join(recording_dir, dev_id)
            os.makedirs(os.path.join(cam_dir, "depth"), exist_ok=True)

            # Save intrinsics
            frame = self.manager.latest_frame(dev_id)
            calib_intrinsics = read_device_intrinsics(dev_id)
            
            intrinsics = {}
            if frame:
                if frame.color_intrinsics:
                    ci = frame.color_intrinsics
                    # Use calibrated values from JSON if available, else fall back to SDK frame intrinsics
                    intrinsics["color"] = {
                        "fx": calib_intrinsics.fx if calib_intrinsics else ci.fx,
                        "fy": calib_intrinsics.fy if calib_intrinsics else ci.fy,
                        "cx": calib_intrinsics.cx if calib_intrinsics else ci.cx,
                        "cy": calib_intrinsics.cy if calib_intrinsics else ci.cy,
                        "width": ci.width, "height": ci.height,
                        "dist_coeffs": calib_intrinsics.dist_coeffs.tolist() if calib_intrinsics else ci.dist_coeffs.tolist()
                    }
                if frame.depth_intrinsics:
                    di = frame.depth_intrinsics
                    intrinsics["depth"] = {
                        "fx": di.fx, "fy": di.fy, "cx": di.cx, "cy": di.cy,
                        "width": di.width, "height": di.height,
                        "dist_coeffs": di.dist_coeffs.tolist()
                    }
            
            if intrinsics:
                with open(os.path.join(cam_dir, "intrinsics.json"), "w") as f:
                    json.dump(intrinsics, f, indent=4)

        return recording_dir

    def _init_writers(self, group: SyncedFrameGroup, sync_fps: int):
        """Initialize VideoWriters based on the first synced frame group."""
        for dev_id, frame in group.frames.items():
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            
            # Color writer
            color_path = os.path.join(self.current_recording_dir, dev_id, "color.mp4")
            color_size = (frame.color.shape[1], frame.color.shape[0]) if frame.color is not None else (640, 480)
            
            # Depth writer
            depth_path = os.path.join(self.current_recording_dir, dev_id, "depth_viz.mp4")
            depth_size = (frame.depth.shape[1], frame.depth.shape[0]) if frame.depth is not None else (640, 480)
            
            self._writers[dev_id] = {
                "color": cv2.VideoWriter(color_path, fourcc, sync_fps, color_size),
                "depth": cv2.VideoWriter(depth_path, fourcc, sync_fps, depth_size)
            }
            self._colorizers[dev_id] = DepthColorizer()

    def save_group(self, group: SyncedFrameGroup, sync_fps: int):
        if self.current_recording_dir == None:
            return
        
        if not self._writers:
            self._init_writers(group, sync_fps)

        idx_str = f"{self.frame_idx:06d}"
        self.timestamps.append({
            "sync_timestamp_us": group.sync_timestamp_us,
            "offsets_us": group.offsets_us
        })

        for dev_id, frame in group.frames.items():
            # Save color video
            if frame.color is not None:
                self._writers[dev_id]["color"].write(frame.color)
            
            # Save raw depth for metric analysis
            if frame.depth is not None:
                if viki.config.RECORD_DEPTH == True:
                    depth_path = os.path.join(self.current_recording_dir, dev_id, "depth", f"{idx_str}.npy")
                    np.save(depth_path, frame.depth)
                
                # Save colorized depth video
                colorized = self._colorizers[dev_id].colorize(frame.depth)
                if colorized is not None:
                    self._writers[dev_id]["depth"].write(colorized)
                else:
                    # If depth is invalid, write a black frame to maintain sync
                    self._writers[dev_id]["depth"].write(np.zeros_like(frame.color))

        self.frame_idx += 1

    def _save_extrinsics(self):
        """Saves current extrinsics for all active devices to disk."""
        extrinsics_data = {}
        active_ids = self.manager.active_device_ids()
        for dev_id in active_ids:
            ext = read_device_extrinsics(dev_id)
            if ext:
                extrinsics_data[dev_id] = {
                    "rvec": ext.rvec.tolist(),
                    "tvec": ext.tvec.tolist()
                }
        
        with open(os.path.join(self.current_recording_dir, "extrinsics.json"), "w") as f:
            json.dump(extrinsics_data, f, indent=4)

    def stop(self):
        """Release writers and finalize metadata."""
        if not self._writers:
            return
            
        for dev_id in self._writers:
            self._writers[dev_id]["color"].release()
            self._writers[dev_id]["depth"].release()

        with open(os.path.join(self.current_recording_dir, "timestamps.json"), "w") as f:
            json.dump(self.timestamps, f, indent=4)
        
        self._save_extrinsics()
        
        self._writers = {}
        print(f"RGB-D recording finalized. Saved {self.frame_idx} frames.")

    def record(self, duration_s: float, sync_fps: int = 15):
        """
        Records synchronized frames for the given duration.
        """
        self.current_recording_dir = self._setup_recording_dir()
        self.frame_idx = 0
        self.timestamps = []
        self._writers = {}
        self._colorizers = {}

        sync = MultiCameraSync(self.manager, sync_fps=sync_fps)
        
        print(f"Recording to {self.current_recording_dir} for {duration_s}s at {sync_fps}fps...")
        
        # Custom loop to pass sync_fps to _save_group
        period_s = 1.0 / sync_fps
        deadline = time.monotonic() + duration_s
        
        while time.monotonic() < deadline:
            tick_wall = time.monotonic()
            group = sync.get_synced_frame()
            if group is not None:
                self._save_group(group, sync_fps)
            
            elapsed = time.monotonic() - tick_wall
            sleep_s = period_s - elapsed
            if sleep_s > 0:
                time.sleep(sleep_s)
        
        # Release writers
        for dev_id in self._writers:
            self._writers[dev_id]["color"].release()
            self._writers[dev_id]["depth"].release()

        # Finalize metadata
        with open(os.path.join(self.current_recording_dir, "timestamps.json"), "w") as f:
            json.dump(self.timestamps, f, indent=4)
        
        self._save_extrinsics()

        print(f"Recording finished. Saved {self.frame_idx} frames.")
        return self.current_recording_dir
