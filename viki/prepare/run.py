"""
viki.prepare.run
--------------------------------------
Business logic for preparing skeleton recordings.

Takes recorded landmarks, interpolates, fuses across cameras, smooths, and
computes end-effector poses (rotation + position) for the retarget stage.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import List

import numpy as np
from viki.dsp import smooth_landmark_sequence, interpolate_nans
from viki.perception.hand_angles import compute_end_effector_pose
from viki.contracts import HAND_LM_COUNT, LM
import viki.config as config

logger = logging.getLogger(__name__)


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
        # >0 leaves interior gaps longer than this many frames unfilled
        self.interp_max_gap = int(getattr(config, "PERCEPTION_INTERP_MAX_GAP", 0))
        # optional explicit fused-output time grid (µs) — the raw synced-frame
        # timestamps, so cln.npz shares one index with the point cloud. None →
        # fuse onto the union of the per-camera detection timestamps.
        self.grid_ts: np.ndarray | None = None
        # No mkdir here: the episode flow (prepare_episode) overrides recs_dir /
        # smoothed_dir to a tempdir right after construction, so eagerly creating
        # data/skeleton_{recs,smoothed}/ just litters (root-owned under Docker).
        # smooth_recording() creates the output dir it actually writes to.

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
            conf_all = (
                np.asarray(data["confidence"], dtype=np.float64)
                if "confidence" in data
                else np.ones((len(timestamps), points.shape[1]), dtype=np.float64)
            )

        if points.size == 0:
            raise ValueError("Recording file is empty.")

        # Backward compat: strip arm landmarks (21, 22) from old files
        hand_mask = landmark_ids < HAND_LM_COUNT
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
        conf_map: dict[str, np.ndarray] = {}
        for dev, idxs in groups.items():
            trajectories[str(dev)] = np.array(
                [points[i] for i in idxs], dtype=np.float32
            )
            ts_map[str(dev)] = np.array(
                [int(timestamps[i]) for i in idxs], dtype=np.int64
            )
            conf_map[str(dev)] = np.array([conf_all[i] for i in idxs], dtype=np.float64)

        # 1. Interpolation part: per camera, independently fill NaN gaps (linear).
        raw_filled: dict[str, np.ndarray] = {}
        for dev in trajectories:
            raw_filled[dev] = interpolate_nans(trajectories[dev], max_gap=self.interp_max_gap)

        # 2. Fusion: confidence-weighted average across cameras onto a common
        #    grid (paper §3.5, eq. 2 — weights from rec.npz["confidence"]).
        from viki.prepare.fuse import fuse_trajectories

        raw_fused, grid = fuse_trajectories(
            raw_filled, ts_map, landmark_ids, weights=conf_map, grid=self.grid_ts
        )

        if grid.size == 0:
            raise ValueError("Recording contains no valid trajectories.")

        # 2b. Fill remaining gaps in the fused trajectory with a cubic spline
        #     (paper §3.7) before smoothing.
        from viki.prepare.interpolate import fill_se3_spline

        raw_fused = fill_se3_spline(raw_fused)

        # 3. Smooth the fused trajectory.
        fused_points = smooth_landmark_sequence(
            raw_fused,
            window_length=window_length,
            polyorder=polyorder,
        )

        # 3b. Per-frame confidence weight ω_t: mean over the wrist-frame
        #     landmarks of the max-over-cameras weight (paper §3.5, eq. 5).
        _wf = [int(LM.WRIST), int(LM.INDEX_MCP), int(LM.MIDDLE_MCP), int(LM.PINKY_MCP)]
        _cols = [np.where(landmark_ids == lm)[0][0] for lm in _wf if lm in landmark_ids]
        grid_conf = np.zeros((len(grid), points.shape[1]), dtype=np.float64)
        for dev, cam_conf in conf_map.items():
            for k, t in enumerate(ts_map[dev]):
                gi = int(np.argmin(np.abs(grid - t)))
                grid_conf[gi] = np.maximum(grid_conf[gi], cam_conf[k])
        omega = (
            grid_conf[:, _cols].mean(axis=1)
            if _cols
            else np.ones(len(grid), dtype=np.float64)
        )
        # Normalise to [0, 1] across the episode (eq. 2's w is unbounded through
        # the d^-2 factor, so ω_t is only meaningful relative to the episode's
        # best-observed frame), then sharpen with the α exponent (eq. 5): α > 1
        # discounts low-confidence frames harder, α = 1 leaves the ratio linear.
        _omax = float(omega.max()) or 1.0
        _alpha = float(getattr(config, "PERCEPTION_CONF_ALPHA", 1.0))
        omega = np.clip(omega / _omax, 0.0, 1.0) ** _alpha
        omega = omega.astype(np.float32)

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

        # 4b. Gripper state per frame, from the fused hand skeleton.
        from viki.gripper import load_gripper

        gripper_model = load_gripper(getattr(config, "GRIPPER", "binary"))
        gripper = np.zeros(T, dtype=bool)
        _g_prev = None
        for t in range(T):
            _pts = {LM(landmark_ids[i]): fused_points[t, i] for i in range(L)}
            _g_prev = gripper_model.estimate(_pts, _g_prev)
            gripper[t] = _g_prev.closed

        # ω_t computed at step 3b from the fused per-landmark weights; keep only
        # frames whose EE pose survived validation.
        omega = np.where(valid, omega, 0.0).astype(np.float32)

        # 4d. Object-relative representation (paper §3.6). STUB: no object-pose
        #     tracker → returns None, cln.npz stays workspace-anchored.
        from viki.prepare.represent import object_relative

        _wrist_T = np.tile(np.eye(4, dtype=np.float64), (T, 1, 1))
        _wrist_T[:, :3, :3] = rotations
        _wrist_T[:, :3, 3] = positions
        T_obj_hand = object_relative(_wrist_T, None)

        # 5. Save to smoothed directory as cln-*.npz
        output_filename = filename.replace("rec-", "cln-")
        self.smoothed_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.smoothed_dir / output_filename

        _extra = {}
        if T_obj_hand is not None:
            _extra["T_obj_hand"] = T_obj_hand.astype(np.float32)

        np.savez_compressed(
            output_path,
            positions=positions,
            rotations=rotations,
            rpy=rpy,
            valid=valid,
            omega=omega,
            gripper=gripper,
            timestamps=grid,
            raw_points=raw_fused.astype(np.float32),
            smoothed_points=fused_points.astype(np.float32),
            landmark_ids=landmark_ids,
            coordinate_frame=getattr(
                config,
                "SKELETON_COORDINATE_FRAME",
                "viki_world_or_camera",
            ),
            **_extra,
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


def prepare_episode(
    ep, window_length: int = 7, polyorder: int = 2, interp_max_gap: int | None = None,
    report=None,
) -> str:
    """
    Episode-aware wrapper around :meth:`PreparationPipeline.smooth_recording`:
    ``ep.rec_npz`` -> ``ep.cln_npz``. Returns the cln.npz path.
    """
    import shutil
    import tempfile

    from viki.episode import mark_stage

    if not ep.rec_npz.exists():
        raise FileNotFoundError(f"no rec.npz for episode {ep.id}; run extract first")

    with tempfile.TemporaryDirectory() as stage:
        stage_p = Path(stage)
        shutil.copy(ep.rec_npz, stage_p / "rec-ep.npz")
        pp = PreparationPipeline()
        pp.recs_dir = stage_p
        pp.smoothed_dir = stage_p
        if interp_max_gap is not None:
            pp.interp_max_gap = int(interp_max_gap)
        # Fuse onto the raw synced-frame grid so cln.npz has one row per
        # recorded frame, index-aligned with cloud/<i>.bin and everything else.
        ts_path = ep.raw_dir / "timestamps.json"
        if ts_path.exists():
            try:
                _ts = json.loads(ts_path.read_text())
                _sync = [int(e["sync_us"]) for e in _ts if "sync_us" in e]
                if _sync:
                    pp.grid_ts = np.asarray(sorted(_sync), dtype=np.int64)
            except Exception:  # noqa: BLE001 — fall back to the union grid
                pass
        _, _ = pp.smooth_recording("rec-ep.npz", window_length, polyorder)
        shutil.copy(stage_p / "cln-ep.npz", ep.cln_npz)

    # Optional: refine the wrist pose by fitting a capsule hand model to the
    # per-frame point cloud (rewrites cln.npz positions/rotations in place,
    # adds hand_joint_angles). Off by default; also runnable standalone via
    # `viki hand-fit`.
    hand_fit = bool(getattr(config, "PERCEPTION_HAND_FIT", False))
    if hand_fit:
        try:
            from viki.perception.hand_fit import refine_cln

            refine_cln(ep, report=report)
        except Exception:  # noqa: BLE001 — never fail prepare on the refinement
            logger.warning("prepare %s: hand-fit refinement failed", ep.id, exc_info=True)
            hand_fit = False

    with np.load(ep.cln_npz) as d:
        n = len(d["positions"])
        obj_rel = "T_obj_hand" in d
        hand_fit = hand_fit and "hand_joint_angles" in d
    mark_stage(ep, "prepare", frames=int(n), object_relative=bool(obj_rel),
               hand_fit=bool(hand_fit))
    logger.info("prepare %s: %d frames -> %s", ep.id, n, ep.cln_npz)
    return str(ep.cln_npz)
