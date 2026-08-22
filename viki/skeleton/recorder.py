"""
viki.skeleton.recorder
----------------------
Handles saving skeleton capture sessions to NPZ files.

Each camera that detects a hand is recorded as its own trajectory; capture‑time
fusion is intentionally NOT performed here (see ``viki.skeleton.pipeline``). The
recorded file keeps the per‑camera trajectories so the smooth/optimisation stage
can fuse them later.

    Stored NPZ layout
    -----------------
    device_ids   : (N,)        object array of camera ids (one per recorded frame)
    timestamps   : (N,)        int64 sync timestamps (µs)
    points       : (N, L, 3)  world‑frame landmark positions (NaN where missing)
    landmark_ids : (L,)        int32 landmark id for the second axis of ``points``
    depth_debug_device_ids : (D,)  camera ids for the depth‑debug columns below
    depth_valid_fraction   : (N, D) share of in‑range depth pixels per camera
    depth_median_m         : (N, D) median valid depth (m) per camera
    depth_mean_m           : (N, D) mean valid depth (m) per camera
    hand_wrist_depth_m     : (N, D) depth (m) at the detected wrist (NaN if none)
    hand_detected          : (N, D) bool, whether a hand was found on that camera
    """

from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path
from typing import List, Optional

import json
import numpy as np
from viki.skeleton.models import SkeletonFrame, LM, DepthDebug
import viki.config as config


class SkeletonRecorder:
    """
    Records per‑camera sequences of SkeletonFrames to a compressed NPZ file.

    Attributes
    ----------
    _base_dir : Path
        Directory where recordings are saved.
    _current_file : Path | None
        Path to the currently open recording file.
    _records : List[tuple[str, SkeletonFrame]]
        (device_id, frame) pairs buffered for the current recording session.
    _depth_debug_records : List[Optional[dict[str, DepthDebug]]]
        Per‑frame depth diagnostics (keyed by device id) aligned with ``_records``.
    """

    def __init__(
        self,
        base_dir: str | Path = "data/skeleton_recs",
        filter_indices: list[LM] | None = None,
    ) -> None:
        """
        Parameters
        ----------
        base_dir : str or Path, default="data/skeleton_recs"
            Root directory for recordings.
        filter_indices : list[LM], optional
            Not used; kept for API compatibility.
        """
        self._base_dir = Path(base_dir)
        self._base_dir.mkdir(parents=True, exist_ok=True)
        self._filter_indices = filter_indices
        self._current_file = None
        self._records: List[tuple[str, SkeletonFrame]] = []
        self._depth_debug_records: List[Optional[dict[str, DepthDebug]]] = []

    def start(self) -> str:
        """
        Start a new recording session.

        Returns
        -------
        str
            Filename (e.g., "rec-12.34-12.12.2025.npz") without the full path.
        """
        self._records = []
        self._depth_debug_records = []
        timestamp = datetime.now().strftime("%H.%M-%d.%m.%Y")
        filename = f"rec-{timestamp}.npz"
        self._current_file = self._base_dir / filename
        return filename

    def record(
        self,
        frame: SkeletonFrame,
        depth_debug: Optional[dict[str, DepthDebug]] = None,
    ) -> None:
        """
        Add one camera's frame to the current recording session.

        Parameters
        ----------
        frame : SkeletonFrame
            A single per‑camera frame to append.
        depth_debug : dict[str, DepthDebug] | None, optional
            Per‑camera depth diagnostics for this frame group, captured so we
            can later inspect what the depth cameras were doing during the
            recording.
        """
        if self._current_file is None:
            return
        self._records.append((frame.device_id, frame))
        self._depth_debug_records.append(depth_debug)

    def stop(self) -> str | None:
        """
        Finalise the recording and write to disk as compressed NumPy arrays.

        Saves all 23 landmarks; missing ones become NaN. If
        ``SKELETON_SAVE_JSON_DEBUG`` is True, also saves a JSON version.

        Returns
        -------
        str or None
            Path to the saved NPZ file, or None if no recording was active.
        """
        if self._current_file is None:
            return None

        all_ids = list(range(LM.N))
        landmark_ids = np.array(all_ids, dtype=np.int32)
        nan3 = np.full(3, np.nan, dtype=np.float32)

        device_ids: List[str] = []
        timestamps: List[int] = []
        points_list: List[np.ndarray] = []
        for dev_id, frame in self._records:
            device_ids.append(dev_id)
            timestamps.append(int(frame.timestamp_us))
            pts = np.array(
                [frame.points.get(LM(idx), nan3) for idx in all_ids],
                dtype=np.float32,
            )
            points_list.append(pts)

        points = np.array(points_list, dtype=np.float32) if points_list else np.empty((0, LM.N, 3), dtype=np.float32)
        # Unicode (not object) array so the file loads without allow_pickle.
        device_ids_arr = np.array(device_ids) if device_ids else np.empty((0,), dtype="<U1")
        timestamps_arr = np.array(timestamps, dtype=np.int64)

        # Depth debug markers: per recorded frame, per device present in the
        # group. Columns are aligned with ``depth_debug_device_ids`` so a frame's
        # row gives each camera's depth stats (valid fraction, median/mean depth,
        # and the wrist depth that feeds hand‑position estimation).
        depth_debug_device_ids = np.empty((0,), dtype="<U1")
        depth_valid_fraction = np.empty((0, 0), dtype=np.float32)
        depth_median_m = np.empty((0, 0), dtype=np.float32)
        depth_mean_m = np.empty((0, 0), dtype=np.float32)
        hand_wrist_depth_m = np.empty((0, 0), dtype=np.float32)
        hand_detected = np.empty((0, 0), dtype=bool)
        if self._depth_debug_records:
            all_devs = sorted(
                {dev for d in self._depth_debug_records if d for dev in d.keys()}
            )
            if all_devs:
                dev_index = {dev: i for i, dev in enumerate(all_devs)}
                D = len(all_devs)
                N = len(self._depth_debug_records)
                vf = np.full((N, D), np.nan, dtype=np.float32)
                med = np.full((N, D), np.nan, dtype=np.float32)
                mean = np.full((N, D), np.nan, dtype=np.float32)
                wrist = np.full((N, D), np.nan, dtype=np.float32)
                hand = np.zeros((N, D), dtype=bool)
                for n, d in enumerate(self._depth_debug_records):
                    if not d:
                        continue
                    for dev, dbg in d.items():
                        i = dev_index[dev]
                        vf[n, i] = dbg.depth_valid_fraction
                        med[n, i] = dbg.depth_median_m
                        mean[n, i] = dbg.depth_mean_m
                        wrist[n, i] = dbg.wrist_depth_m
                        hand[n, i] = dbg.hand_detected
                depth_debug_device_ids = np.array(all_devs)
                depth_valid_fraction = vf
                depth_median_m = med
                depth_mean_m = mean
                hand_wrist_depth_m = wrist
                hand_detected = hand

        np.savez_compressed(
            self._current_file,
            device_ids=device_ids_arr,
            timestamps=timestamps_arr,
            points=points,
            landmark_ids=landmark_ids,
            depth_debug_device_ids=depth_debug_device_ids,
            depth_valid_fraction=depth_valid_fraction,
            depth_median_m=depth_median_m,
            depth_mean_m=depth_mean_m,
            hand_wrist_depth_m=hand_wrist_depth_m,
            hand_detected=hand_detected,
        )

        if getattr(config, 'SKELETON_SAVE_JSON_DEBUG', False):
            json_path = self._current_file.with_suffix(".json")
            json_data = []
            for (dev_id, frame), ddbg in zip(self._records, self._depth_debug_records):
                entry = {
                    "device_id": dev_id,
                    "ts": int(frame.timestamp_us),
                    "landmarks": {
                        idx: frame.points.get(LM(idx), nan3).tolist() for idx in all_ids
                    },
                    "end_effector": frame.end_effector.as_dict() if frame.end_effector else None,
                }
                if ddbg:
                    entry["depth_debug"] = {
                        dev: {
                            "depth_valid_fraction": dbg.depth_valid_fraction,
                            "depth_median_m": dbg.depth_median_m,
                            "depth_mean_m": dbg.depth_mean_m,
                            "hand_detected": dbg.hand_detected,
                            "wrist_depth_m": dbg.wrist_depth_m,
                        }
                        for dev, dbg in ddbg.items()
                    }
                json_data.append(entry)
            with open(json_path, "w") as f:
                json.dump(json_data, f, indent=2)

        path = str(self._current_file)
        self._current_file = None
        self._records = []
        self._depth_debug_records = []
        return path

    @property
    def is_recording(self) -> bool:
        """True if a recording session is currently active."""
        return self._current_file is not None
