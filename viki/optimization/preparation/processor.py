"""
viki.optimization.preparation.processor
--------------------------------------
Business logic for preparing skeleton recordings.

Takes recorded landmarks, interpolates, fuses across cameras, smooths, and
computes end-effector poses (rotation + position) for the retarget stage.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import List

import numpy as np
from .smoothing import smooth_landmark_sequence, interpolate_nans
from viki.skeleton.hand_angles import compute_end_effector_pose
from viki.skeleton.models import LM
import viki.config as config


_PALM_MIN_LENGTH_RATIO = 0.5
_PALM_MAX_MIDDLE_LENGTH_RATIO = 1.75
_PALM_MAX_THUMB_LENGTH_RATIO = 2.0
_PALM_MIN_SINE_ANGLE = 0.25
_PALM_MAX_STEP_DEG = 60.0


def stable_palm_orientation_mask(
    points: np.ndarray,
    landmark_ids: np.ndarray,
    rotations: np.ndarray,
    pose_valid: np.ndarray,
) -> np.ndarray:
    """Reject palm frames with implausible hand geometry or temporal flips."""
    ids = {int(landmark_id): idx for idx, landmark_id in enumerate(landmark_ids)}
    required = (int(LM.WRIST), int(LM.THUMB_CMC), int(LM.MIDDLE_MCP))
    if any(landmark_id not in ids for landmark_id in required):
        return np.zeros(len(points), dtype=bool)

    wrist = points[:, ids[int(LM.WRIST)]]
    thumb = points[:, ids[int(LM.THUMB_CMC)]] - wrist
    middle = points[:, ids[int(LM.MIDDLE_MCP)]] - wrist
    thumb_length = np.linalg.norm(thumb, axis=1)
    middle_length = np.linalg.norm(middle, axis=1)
    cross_length = np.linalg.norm(np.cross(middle, thumb), axis=1)
    with np.errstate(divide="ignore", invalid="ignore"):
        sine_angle = cross_length / (middle_length * thumb_length)

    finite_geometry = (
        np.isfinite(thumb_length)
        & np.isfinite(middle_length)
        & np.isfinite(sine_angle)
        & (thumb_length > 0.0)
        & (middle_length > 0.0)
    )
    if not finite_geometry.any():
        return np.zeros(len(points), dtype=bool)

    median_thumb = float(np.median(thumb_length[finite_geometry]))
    median_middle = float(np.median(middle_length[finite_geometry]))
    valid = np.asarray(pose_valid, dtype=bool).copy()
    valid &= finite_geometry
    valid &= thumb_length >= _PALM_MIN_LENGTH_RATIO * median_thumb
    valid &= thumb_length <= _PALM_MAX_THUMB_LENGTH_RATIO * median_thumb
    valid &= middle_length >= _PALM_MIN_LENGTH_RATIO * median_middle
    valid &= middle_length <= _PALM_MAX_MIDDLE_LENGTH_RATIO * median_middle
    valid &= sine_angle >= _PALM_MIN_SINE_ANGLE
    valid &= np.isfinite(rotations).all(axis=(1, 2))

    if len(rotations) > 1:
        relative = np.einsum(
            "tji,tjk->tik",
            rotations[:-1].astype(np.float64),
            rotations[1:].astype(np.float64),
        )
        cosine = np.clip(
            (np.trace(relative, axis1=1, axis2=2) - 1.0) / 2.0,
            -1.0,
            1.0,
        )
        jumps = np.degrees(np.arccos(cosine)) > _PALM_MAX_STEP_DEG
        valid[:-1] &= ~jumps
        valid[1:] &= ~jumps

    return valid

class PreparationPipeline:
    """
    Handles listing and preparation of skeleton recording files.

    Attributes
    ----------
    recs_dir : Path
        Directory containing raw recordings (rec-*.npz).
    smoothed_dir : Path
        Directory for smoothed outputs (cln-*.npz).
    """

    def __init__(self) -> None:
        self.recs_dir = Path(config.SKELETON_RECS_DIR)
        self.smoothed_dir = Path(config.SKELETON_SMOOTHED_DIR)
        
        self.recs_dir.mkdir(parents=True, exist_ok=True)
        self.smoothed_dir.mkdir(parents=True, exist_ok=True)

    def list_recordings(self, page: int = 0, page_size: int = 10) -> List[str]:
        """
        List all NPZ recording files in the recordings directory with pagination.

        Parameters
        ----------
        page : int, default=0
            Page number (zero‑based).
        page_size : int, default=10
            Number of items per page.

        Returns
        -------
        List[str]
            List of filenames (e.g., "rec-123.npz").
        """
        files = sorted([f.name for f in self.recs_dir.glob("rec-*.npz")], reverse=True)
        start = page * page_size
        end = start + page_size
        return files[start:end]

    def smooth_recording(
        self, 
        filename: str, 
        window_length: int = 7, 
        polyorder: int = 2
    ) -> tuple[str, np.ndarray]:
        """
        Load a recording, smooth its landmarks, and compute end‑effector poses.

        The result is saved as a compressed NPZ in the smoothed directory with
        prefix "cln-". If `SKELETON_SAVE_JSON_DEBUG` is True, a JSON version is also saved.

        Parameters
        ----------
        filename : str
            Name of the raw recording file (e.g., "rec-123.npz").
        window_length : int, default=7
            Savitzky‑Golay window length (must be odd > polyorder).
        polyorder : int, default=2
            Savitzky‑Golay polynomial order.

        Returns
        -------
        tuple[str, np.ndarray]
            (path_to_smoothed_file, smoothed_points) where smoothed_points has shape (T, L, 3).

        Raises
        ------
        FileNotFoundError
            If the input file does not exist.
        ValueError
            If the recording is empty.
        """
        input_path = self.recs_dir / filename
        if not input_path.exists():
            raise FileNotFoundError(f"Recording file {filename} not found.")

        with np.load(input_path) as data:
            if "device_ids" not in data:
                # Legacy single-trajectory recording: treat as one camera.
                timestamps = data["timestamps"]
                points = data["points"]
                landmark_ids = data["landmark_ids"]
                device_ids = np.array(["cam0"] * len(timestamps), dtype=object)
            else:
                device_ids = list(data["device_ids"])
                timestamps = data["timestamps"]
                points = data["points"]
                landmark_ids = data["landmark_ids"]

        if points.size == 0:
            raise ValueError("Recording file is empty.")

        # Backward compat: strip arm landmarks (21, 22) from old files
        hand_mask = landmark_ids < LM.N
        if not hand_mask.all():
            points = points[:, hand_mask, :]
            landmark_ids = landmark_ids[hand_mask]

        # Reconstruct per-camera trajectories (each camera may have skipped
        # frames where it did not detect the hand).
        from collections import defaultdict

        groups: dict[object, list[int]] = defaultdict(list)
        for i, dev in enumerate(device_ids):
            groups[dev].append(i)

        trajectories: dict[str, np.ndarray] = {}
        ts_map: dict[str, np.ndarray] = {}
        for dev, idxs in groups.items():
            trajectories[str(dev)] = np.array(
                [points[i] for i in idxs], dtype=np.float32
            )
            ts_map[str(dev)] = np.array(
                [int(timestamps[i]) for i in idxs], dtype=np.int64
            )

        # 1. Interpolation part: per camera, independently fill NaN gaps.
        raw_filled: dict[str, np.ndarray] = {}
        for dev in trajectories:
            raw_filled[dev] = interpolate_nans(trajectories[dev])

        # 2. Fusion part: fuse the interpolated per-camera trajectories onto a
        #    common time grid (deferred from capture time).
        from .fusion import fuse_trajectories

        raw_fused, grid = fuse_trajectories(raw_filled, ts_map, landmark_ids)

        if grid.size == 0:
            raise ValueError("Recording contains no valid trajectories.")

        # 3. Smooth the fused trajectory.
        fused_points = smooth_landmark_sequence(
            raw_fused,
            window_length=window_length,
            polyorder=polyorder,
        )

        # 4. Compute end-effector poses on the smoothed fused trajectory.
        T = fused_points.shape[0]
        L = fused_points.shape[1]

        positions = np.zeros((T, 3), dtype=np.float32)
        rotations = np.zeros((T, 3, 3), dtype=np.float32)
        rpy = np.zeros((T, 3), dtype=np.float32)
        valid = np.zeros(T, dtype=bool)

        for t in range(T):
            current_mapping = {LM(landmark_ids[i]): fused_points[t, i] for i in range(L)}

            pose = compute_end_effector_pose(current_mapping, int(grid[t]))

            positions[t] = pose.position
            rotations[t] = pose.R_world_palm
            rpy[t] = pose.rpy_deg
            valid[t] = pose.valid

        valid = stable_palm_orientation_mask(
            fused_points,
            landmark_ids,
            rotations,
            valid,
        )

        # 5. Save to smoothed directory as cln-*.npz
        output_filename = filename.replace("rec-", "cln-")
        output_path = self.smoothed_dir / output_filename

        np.savez_compressed(
            output_path,
            positions=positions,
            rotations=rotations,
            rpy=rpy,
            valid=valid,
            timestamps=grid,
            raw_points=raw_fused.astype(np.float32),
            smoothed_points=fused_points.astype(np.float32),
            landmark_ids=landmark_ids,
            coordinate_frame=getattr(
                config,
                "SKELETON_COORDINATE_FRAME",
                "viki_world_or_camera",
            ),
        )

        if getattr(config, 'SKELETON_SAVE_JSON_DEBUG', False):
            json_path = output_path.with_suffix(".json")
            json_data = []
            for t in range(T):
                frame_pts = {int(landmark_ids[i]): fused_points[t, i].tolist() for i in range(L)}
                frame = {
                    "ts": int(grid[t]),
                    "landmarks": frame_pts,
                    "end_effector": {
                        "position": positions[t].tolist(),
                        "R_world_palm": rotations[t].tolist(),
                        "rpy_deg": rpy[t].tolist(),
                        "valid": bool(valid[t]),
                        "timestamp_us": int(grid[t]),
                    }
                }
                json_data.append(frame)
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2)

        return str(output_path), fused_points


def estimate_fps(timestamps_us: np.ndarray) -> float:
    """Estimate frame rate (Hz) from a sequence of microsecond timestamps."""
    if len(timestamps_us) < 2:
        return 30.0
    dt = np.diff(timestamps_us.astype(np.float64)) / 1_000_000.0
    dt = dt[np.isfinite(dt) & (dt > 0)]
    if len(dt) == 0:
        return 30.0
    return float(1.0 / np.median(dt))
