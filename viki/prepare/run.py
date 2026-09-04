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


def _pose_and_gripper(
    points: np.ndarray, landmark_ids: np.ndarray, grid: np.ndarray,
    gripper_name: str,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Derive the complete consumer pose contract from one landmark stage."""
    T, L = points.shape[:2]
    positions = np.zeros((T, 3), dtype=np.float32)
    rotations = np.zeros((T, 3, 3), dtype=np.float32)
    rpy = np.zeros((T, 3), dtype=np.float32)
    valid = np.zeros(T, dtype=bool)
    mappings = []
    for t in range(T):
        mapping = {LM(landmark_ids[i]): points[t, i] for i in range(L)}
        mappings.append(mapping)
        pose = compute_end_effector_pose(mapping, int(grid[t]))
        positions[t] = pose.position
        rotations[t] = pose.R_world_palm
        rpy[t] = pose.rpy_deg
        valid[t] = pose.valid

    valid = stable_palm_orientation_mask(points, landmark_ids, rotations, valid)

    from viki.gripper import load_gripper

    gripper_model = load_gripper(gripper_name)
    gripper = np.zeros(T, dtype=bool)
    previous = None
    for t, mapping in enumerate(mappings):
        previous = gripper_model.estimate(mapping, previous)
        gripper[t] = previous.closed
    return positions, rotations, rpy, valid, gripper


def _confidence_arrays(
    grid_conf: np.ndarray,
    landmark_ids: np.ndarray,
    valid: np.ndarray,
    alpha: float,
) -> tuple[np.ndarray, np.ndarray]:
    """Normalise per-joint evidence and derive the per-frame palm confidence."""
    wrist_frame = [int(LM.WRIST), int(LM.INDEX_MCP), int(LM.MIDDLE_MCP), int(LM.PINKY_MCP)]
    columns = [
        np.where(landmark_ids == lm)[0][0]
        for lm in wrist_frame if lm in landmark_ids
    ]
    omega = (
        grid_conf[:, columns].mean(axis=1)
        if columns else np.ones(len(grid_conf), dtype=np.float64)
    )
    maximum = float(np.nanmax(omega)) if np.isfinite(omega).any() else 1.0
    if not np.isfinite(maximum) or maximum <= 0.0:
        maximum = 1.0
    omega = np.clip(omega / maximum, 0.0, 1.0) ** alpha
    omega = np.where(valid, omega, 0.0).astype(np.float32)

    confidence_max = float(np.nanmax(grid_conf)) if grid_conf.size else 0.0
    landmark_confidence = np.clip(
        grid_conf / (confidence_max or 1.0), 0.0, 1.0
    ).astype(np.float32)
    return omega, landmark_confidence


def _cln_payload(
    *,
    points: np.ndarray,
    observed_points: np.ndarray,
    filled_points: np.ndarray,
    landmark_ids: np.ndarray,
    grid: np.ndarray,
    grid_conf: np.ndarray,
    fusion_mode: str,
    checkpoint_stage: str,
    interp_max_gap: int,
    window_length: int,
    polyorder: int,
    profile_name: str = "",
    pose_source: str = "landmarks",
    confidence_alpha: float = 1.0,
    gripper_name: str = "binary",
    coordinate_frame: str = "viki_world_or_camera",
) -> dict[str, object]:
    """Build a viewer/retarget-compatible artifact for any prepare boundary."""
    positions, rotations, rpy, valid, gripper = _pose_and_gripper(
        points, landmark_ids, grid, gripper_name
    )
    omega, landmark_confidence = _confidence_arrays(
        grid_conf, landmark_ids, valid, confidence_alpha,
    )
    params = {
        "fusion_mode": fusion_mode,
        "interp_max_gap": int(interp_max_gap),
        "smoothing": "savgol" if checkpoint_stage in {"smoothed", "hand_fit"} else "none",
        "window_length": int(window_length),
        "polyorder": int(polyorder),
        "profile": profile_name or None,
        "pose_source": pose_source,
        "confidence_alpha": confidence_alpha,
        "gripper": gripper_name,
        "coordinate_frame": coordinate_frame,
    }
    observed_mask = np.isfinite(observed_points).all(axis=2)
    filled_mask = np.isfinite(filled_points).all(axis=2)
    payload: dict[str, object] = {
        "positions": positions,
        "rotations": rotations,
        "rpy": rpy,
        "valid": valid,
        "omega": omega,
        "landmark_confidence": landmark_confidence,
        "gripper": gripper,
        "timestamps": np.asarray(grid, np.int64),
        "raw_points": np.asarray(filled_points, np.float32),
        "smoothed_points": np.asarray(points, np.float32),
        "observed_points": np.asarray(observed_points, np.float32),
        "filled_points": np.asarray(filled_points, np.float32),
        "observed_mask": observed_mask,
        "interpolated_mask": (~observed_mask & filled_mask),
        "landmark_ids": np.asarray(landmark_ids),
        "coordinate_frame": np.asarray(coordinate_frame),
        "perception_fuse_mode": np.asarray(fusion_mode),
        "checkpoint_stage": np.asarray(checkpoint_stage),
        "checkpoint_params_json": np.asarray(json.dumps(params, sort_keys=True)),
        "pose_source": np.asarray(pose_source),
    }
    if profile_name:
        payload["perception_profile"] = np.asarray(profile_name)
        payload["active_variant"] = np.asarray(profile_name)
    return payload

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
        self.fusion_mode = str(getattr(config, "PERCEPTION_FUSE_MODE", "xyz_mean"))
        self.profile_name = ""
        self.pose_source = "landmarks"
        self.confidence_alpha = float(getattr(config, "PERCEPTION_CONF_ALPHA", 1.0))
        self.gripper_name = str(getattr(config, "GRIPPER", "binary"))
        self.coordinate_frame = str(getattr(
            config, "SKELETON_COORDINATE_FRAME", "viki_world_or_camera",
        ))
        self.require_triangulation = False
        # Optional episode-owned directory.  When present, every material
        # prepare boundary is persisted instead of disappearing in temporaries.
        self.checkpoints_dir: Path | None = None
        # >0 leaves interior gaps longer than this many frames unfilled
        self.interp_max_gap = int(getattr(config, "PERCEPTION_INTERP_MAX_GAP", 0))
        # optional explicit fused-output time grid (µs) — the raw synced-frame
        # timestamps, so cln.npz shares one index with the point cloud. None →
        # fuse onto the union of the per-camera detection timestamps.
        self.grid_ts: np.ndarray | None = None
        # Multi-view triangulation output (raw/joints3d.npz). When set and
        # PERCEPTION_FUSE_MODE == "triangulate", fusion uses these world-frame
        # joints + their per-joint quality instead of averaging per-camera XYZ.
        self.joints3d_path: Path | None = None
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
            conf_all = conf_all[:, hand_mask]
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

        checkpoint_params = {
            "fusion_mode": self.fusion_mode,
            "interp_max_gap": int(self.interp_max_gap),
            "window_length": int(window_length),
            "polyorder": int(polyorder),
            "profile": self.profile_name or None,
        }
        if self.checkpoints_dir is not None:
            from viki.prepare.checkpoints import save_camera_stage

            save_camera_stage(
                self.checkpoints_dir / "00_per_camera_observed.npz",
                stage="per_camera_observed", trajectories=trajectories,
                timestamps=ts_map, confidence=conf_map,
                landmark_ids=landmark_ids, params=checkpoint_params,
            )
            save_camera_stage(
                self.checkpoints_dir / "05_per_camera_filled.npz",
                stage="per_camera_filled", trajectories=raw_filled,
                timestamps=ts_map, confidence=conf_map,
                landmark_ids=landmark_ids, params=checkpoint_params,
            )

        # 2. Fusion. Two modes (PERCEPTION_FUSE_MODE, kept switchable for A/B):
        #    - "xyz_mean" (legacy): confidence-weighted average of the per-camera
        #      monocular reconstructions onto a common grid (paper §3.5 eq. 2).
        #    - "triangulate": use raw/joints3d.npz — joints solved by multi-view
        #      triangulation directly in the world frame — snapped onto the grid;
        #      the per-joint triangulation `quality` becomes landmark_confidence.
        from viki.prepare.fuse import fuse_trajectories, snap_to_grid

        _mode_requested = self.fusion_mode
        tri_xyz = tri_timestamps = tri_quality = None
        if (_mode_requested == "triangulate" and self.joints3d_path
                and Path(self.joints3d_path).exists()):
            with np.load(self.joints3d_path, allow_pickle=False) as tri:
                tri_xyz = np.asarray(tri["xyz"], np.float32)
                tri_timestamps = np.asarray(tri["timestamps"], np.int64)
                tri_quality = np.asarray(tri["quality"], np.float32)
        elif _mode_requested == "triangulate":
            if self.require_triangulation:
                raise FileNotFoundError(
                    "named baseline requires raw/joints3d.npz; refusing "
                    "the legacy xyz_mean fallback"
                )
            logger.warning("prepare: PERCEPTION_FUSE_MODE=triangulate but no "
                           "joints3d.npz — falling back to xyz_mean")

        tri_lm_conf = None
        if tri_xyz is not None and tri_timestamps is not None and tri_quality is not None:
            _mode = "triangulate"
            grid = (self.grid_ts if self.grid_ts is not None
                    else np.asarray(sorted(tri_timestamps), dtype=np.int64))
            raw_fused = snap_to_grid(tri_xyz, tri_timestamps, grid)
            tri_lm_conf = np.nan_to_num(
                snap_to_grid(tri_quality, tri_timestamps, grid), nan=0.0
            )
            landmark_ids = np.arange(tri_xyz.shape[1], dtype=landmark_ids.dtype)
            logger.info("prepare: fused from triangulation (%d frames, %d joints)",
                        len(grid), landmark_ids.size)
        else:
            _mode = "xyz_mean"
            raw_fused, grid = fuse_trajectories(
                raw_filled, ts_map, landmark_ids, weights=conf_map, grid=self.grid_ts
            )

        if grid.size == 0:
            raise ValueError("Recording contains no valid trajectories.")

        observed_fused = np.asarray(raw_fused, np.float32).copy()

        # 2b. Fill remaining gaps in the fused trajectory with a cubic spline
        #     (paper §3.7) before smoothing.
        from viki.prepare.interpolate import fill_se3_spline

        filled_fused = fill_se3_spline(
            observed_fused, max_gap=self.interp_max_gap
        )

        # 3. Smooth the fused trajectory.
        fused_points = smooth_landmark_sequence(
            filled_fused,
            window_length=window_length,
            polyorder=polyorder,
        )

        # 3b. Preserve the per-joint evidence on the common grid.  It is used
        # unchanged by every checkpoint so A/B differences come only from the
        # selected processing stage.
        if tri_lm_conf is not None:
            grid_conf = tri_lm_conf.astype(np.float64)   # per-joint triangulation quality
        else:
            grid_conf = np.zeros((len(grid), landmark_ids.size), dtype=np.float64)
            for dev, cam_conf in conf_map.items():
                for k, t in enumerate(ts_map[dev]):
                    gi = int(np.argmin(np.abs(grid - t)))
                    grid_conf[gi] = np.maximum(grid_conf[gi], cam_conf[k])

        # 4. Build the complete consumer contract once.  The same builder is
        # used below for observed/filled/smoothed checkpoints, preventing the
        # diagnostics from silently using different pose logic than cln.npz.
        payload = _cln_payload(
            points=fused_points,
            observed_points=observed_fused,
            filled_points=filled_fused,
            landmark_ids=landmark_ids,
            grid=grid,
            grid_conf=grid_conf,
            fusion_mode=_mode,
            checkpoint_stage="smoothed",
            interp_max_gap=self.interp_max_gap,
            window_length=window_length,
            polyorder=polyorder,
            profile_name=self.profile_name,
            pose_source=self.pose_source,
            confidence_alpha=self.confidence_alpha,
            gripper_name=self.gripper_name,
            coordinate_frame=self.coordinate_frame,
        )
        positions = np.asarray(payload["positions"])
        rotations = np.asarray(payload["rotations"])
        rpy = np.asarray(payload["rpy"])
        valid = np.asarray(payload["valid"])
        T, L = fused_points.shape[:2]

        # 4d. Object-relative representation (paper §3.6). STUB: no object-pose
        #     tracker → returns None, cln.npz stays workspace-anchored.
        from viki.prepare.represent import object_relative

        _wrist_T = np.tile(np.eye(4, dtype=np.float64), (T, 1, 1))
        _wrist_T[:, :3, :3] = rotations
        _wrist_T[:, :3, 3] = positions
        T_obj_hand = object_relative(_wrist_T, None)

        # 5. Save to smoothed directory as cln-*.npz
        output_filename = (
            "cln.npz" if filename == "rec.npz" else filename.replace("rec-", "cln-")
        )
        self.smoothed_dir.mkdir(parents=True, exist_ok=True)
        output_path = self.smoothed_dir / output_filename

        if T_obj_hand is not None:
            payload["T_obj_hand"] = T_obj_hand.astype(np.float32)

        from viki.prepare.checkpoints import atomic_savez

        atomic_savez(output_path, payload)

        if self.checkpoints_dir is not None:
            observed_payload = _cln_payload(
                points=observed_fused, observed_points=observed_fused,
                filled_points=filled_fused, landmark_ids=landmark_ids,
                grid=grid, grid_conf=grid_conf, fusion_mode=_mode,
                checkpoint_stage="observed", interp_max_gap=self.interp_max_gap,
                window_length=window_length, polyorder=polyorder,
                profile_name=self.profile_name, pose_source=self.pose_source,
                confidence_alpha=self.confidence_alpha,
                gripper_name=self.gripper_name,
                coordinate_frame=self.coordinate_frame,
            )
            filled_payload = _cln_payload(
                points=filled_fused, observed_points=observed_fused,
                filled_points=filled_fused, landmark_ids=landmark_ids,
                grid=grid, grid_conf=grid_conf, fusion_mode=_mode,
                checkpoint_stage="filled", interp_max_gap=self.interp_max_gap,
                window_length=window_length, polyorder=polyorder,
                profile_name=self.profile_name, pose_source=self.pose_source,
                confidence_alpha=self.confidence_alpha,
                gripper_name=self.gripper_name,
                coordinate_frame=self.coordinate_frame,
            )
            checkpoint_files = [
                self.checkpoints_dir / "10_fused_observed.npz",
                self.checkpoints_dir / "20_fused_filled.npz",
                self.checkpoints_dir / "30_smoothed.npz",
            ]
            atomic_savez(checkpoint_files[0], observed_payload)
            atomic_savez(checkpoint_files[1], filled_payload)
            atomic_savez(checkpoint_files[2], payload)

            from viki.prepare.checkpoints import atomic_write_json, write_comparison

            atomic_write_json(self.checkpoints_dir / "manifest.json", {
                "schema": 1,
                "fusion_mode_requested": _mode_requested,
                "fusion_mode_used": _mode,
                "interp_max_gap": int(self.interp_max_gap),
                "window_length": int(window_length),
                "polyorder": int(polyorder),
                "profile": self.profile_name or None,
                "files": [p.name for p in (
                    self.checkpoints_dir / "00_per_camera_observed.npz",
                    self.checkpoints_dir / "05_per_camera_filled.npz",
                    *checkpoint_files,
                )],
            })
            write_comparison(self.checkpoints_dir / "comparison.json", checkpoint_files)

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


def _configure_episode_inputs(
    pp: PreparationPipeline,
    ep,
    *,
    triangulation: dict[str, object] | None = None,
    force_triangulation: bool = False,
    require_triangulation: bool = False,
) -> None:
    """Attach the recorded frame grid and requested triangulation artifact."""
    pp.require_triangulation = bool(require_triangulation)
    ts_path = ep.raw_dir / "timestamps.json"
    if ts_path.exists():
        try:
            entries = json.loads(ts_path.read_text())
            sync = [int(entry["sync_us"]) for entry in entries if "sync_us" in entry]
            if sync:
                pp.grid_ts = np.asarray(sorted(sync), dtype=np.int64)
        except Exception:  # noqa: BLE001 — fall back to the union grid
            pass

    if pp.fusion_mode != "triangulate":
        return
    joints = ep.raw_dir / "joints3d.npz"
    observations = ep.raw_dir / "observations.npz"
    try:
        if observations.exists() and (
            force_triangulation
            or not joints.exists()
            or joints.stat().st_mtime < observations.stat().st_mtime
        ):
            from viki.perception.triangulate import TriConfig, triangulate_episode

            triangulate_episode(ep.raw_dir, cfg=TriConfig(triangulation))
        if joints.exists():
            pp.joints3d_path = joints
        elif require_triangulation:
            raise FileNotFoundError(
                f"profile requires {observations}; run profile extraction first"
            )
    except Exception:
        if require_triangulation:
            raise
        logger.warning("prepare %s: triangulation step failed", ep.id, exc_info=True)


def generate_stage_checkpoints(
    ep,
    *,
    fusion_modes: tuple[str, ...] = ("triangulate", "xyz_mean"),
    window_length: int = 7,
    polyorder: int = 2,
    interp_max_gap: int | None = None,
) -> dict[str, str]:
    """Generate non-destructive A/B prepare boundaries for one episode.

    Nothing here replaces ``cln.npz``.  Each fusion mode receives its own
    parameter-named directory, and a cross-variant metrics file is written at
    ``intermediates/prepare/comparison.json``.
    """
    if not ep.rec_npz.exists():
        raise FileNotFoundError(f"no rec.npz for episode {ep.id}; run extract first")
    from viki.prepare.checkpoints import run_name, write_comparison

    outputs: dict[str, str] = {}
    for mode in fusion_modes:
        if mode not in {"triangulate", "xyz_mean"}:
            raise ValueError(f"unknown fusion mode {mode!r}")
        pp = PreparationPipeline()
        pp.recs_dir = ep.root
        pp.fusion_mode = mode
        if interp_max_gap is not None:
            pp.interp_max_gap = int(interp_max_gap)
        name = run_name(mode, pp.interp_max_gap, window_length, polyorder)
        run_dir = ep.intermediates_dir / "prepare" / name
        pp.smoothed_dir = run_dir
        pp.checkpoints_dir = run_dir
        _configure_episode_inputs(pp, ep)
        output, _ = pp.smooth_recording("rec.npz", window_length, polyorder)
        outputs[mode] = output
    # Keep the episode-level comparison cumulative: a later run with another
    # gap/window setting must not erase the earlier rows it was meant to compare.
    prepare_root = ep.intermediates_dir / "prepare"
    # Include the active and any preserved root CLNs as controls (notably old
    # hand-fit outputs), but never modify them.
    viewer_variants: list[Path] = sorted(ep.root.glob("cln*.npz"))
    for run_dir in sorted(path for path in prepare_root.iterdir() if path.is_dir()):
        run_variants = sorted(
            path for path in run_dir.glob("*_*.npz")
            if path.name[:2] in {"10", "20", "30", "40"}
        )
        if run_variants:
            write_comparison(run_dir / "comparison.json", run_variants)
            viewer_variants.extend(run_variants)
    write_comparison(prepare_root / "comparison.json", viewer_variants)
    return outputs


def prepare_episode(
    ep, window_length: int = 7, polyorder: int = 2, interp_max_gap: int | None = None,
    report=None, profile: str | None = None,
) -> str:
    """
    Episode-aware wrapper around :meth:`PreparationPipeline.smooth_recording`:
    ``ep.rec_npz`` -> ``ep.cln_npz``. Returns the cln.npz path.
    """
    import shutil
    import tempfile

    from viki.episode import mark_stage
    from viki.perception.profiles import get_profile

    profile_spec = get_profile(profile)
    if profile_spec is not None:
        window_length = profile_spec.sg_window
        polyorder = profile_spec.sg_polyorder
        interp_max_gap = profile_spec.interp_max_gap

    if not ep.rec_npz.exists():
        raise FileNotFoundError(f"no rec.npz for episode {ep.id}; run extract first")

    with tempfile.TemporaryDirectory() as stage:
        stage_p = Path(stage)
        shutil.copy(ep.rec_npz, stage_p / "rec-ep.npz")
        pp = PreparationPipeline()
        pp.recs_dir = stage_p
        pp.smoothed_dir = stage_p
        if profile_spec is not None:
            pp.fusion_mode = profile_spec.fusion_mode
            pp.profile_name = profile_spec.name
            pp.pose_source = profile_spec.pose_source
            pp.confidence_alpha = profile_spec.confidence_alpha
            pp.gripper_name = profile_spec.gripper
            pp.coordinate_frame = profile_spec.coordinate_frame
        if interp_max_gap is not None:
            pp.interp_max_gap = int(interp_max_gap)
        from viki.prepare.checkpoints import run_name

        pp.checkpoints_dir = ep.intermediates_dir / "prepare" / run_name(
            pp.fusion_mode, pp.interp_max_gap, window_length, polyorder
        )
        _configure_episode_inputs(
            pp,
            ep,
            triangulation=(profile_spec.triangulation if profile_spec else None),
            force_triangulation=profile_spec is not None,
            require_triangulation=profile_spec is not None,
        )

        _, _ = pp.smooth_recording("rec-ep.npz", window_length, polyorder)
        shutil.copy(stage_p / "cln-ep.npz", ep.cln_npz)

    # Optional trajectory-level capsule fit. It appends hand_fit_* arrays and
    # deliberately leaves landmark-derived positions/rotations untouched.
    # Also runnable standalone via `viki hand-fit`.
    hand_fit = (
        profile_spec.hand_fit
        if profile_spec is not None
        else bool(getattr(config, "PERCEPTION_HAND_FIT", False))
    )
    if hand_fit:
        try:
            from viki.perception.hand_fit import refine_cln

            refine_cln(ep, report=report)
            from viki.prepare.checkpoints import atomic_savez, write_comparison

            with np.load(ep.cln_npz, allow_pickle=False) as fitted:
                fitted_payload = {key: fitted[key] for key in fitted.files}
            fitted_payload["checkpoint_stage"] = np.asarray("hand_fit")
            hand_fit_path = pp.checkpoints_dir / "40_hand_fit.npz"
            atomic_savez(hand_fit_path, fitted_payload)
            write_comparison(
                pp.checkpoints_dir / "comparison.json",
                [
                    pp.checkpoints_dir / "10_fused_observed.npz",
                    pp.checkpoints_dir / "20_fused_filled.npz",
                    pp.checkpoints_dir / "30_smoothed.npz",
                    hand_fit_path,
                ],
            )
        except Exception:  # noqa: BLE001 — never fail prepare on the refinement
            logger.warning("prepare %s: hand-fit refinement failed", ep.id, exc_info=True)
            hand_fit = False

    baseline = None
    if profile_spec is not None:
        from viki.prepare.baseline import protect_baseline

        baseline = protect_baseline(ep, profile_spec, ep.cln_npz)

    with np.load(ep.cln_npz) as d:
        n = len(d["positions"])
        obj_rel = "T_obj_hand" in d
        hand_fit = hand_fit and "hand_fit_joint_angles" in d
    mark_stage(ep, "prepare", frames=int(n), object_relative=bool(obj_rel),
               hand_fit=bool(hand_fit), profile=profile or "config",
               baseline=baseline)
    logger.info("prepare %s: %d frames -> %s", ep.id, n, ep.cln_npz)
    return str(ep.cln_npz)
