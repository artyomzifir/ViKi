"""
viki.export.trajectory
----------------------
Pipeline stage 6a: retargeted episodes -> a self-contained trajectory bundle.

This is the export that does not need a policy-training framework. It writes
the robot joint trajectory, the commanded and achieved tool pose, the gripper
command, the human hand pose it came from, and the per-frame evidence weight,
in plain ``.npz`` with a JSON manifest. numpy only — no torch, no ``lerobot``,
no parquet.

It exists because the LeRobot exporter cannot run at all without those optional
dependencies, while the trajectories themselves are finished artifacts that a
consumer may legitimately want on their own: to replay on hardware, to inspect,
to convert to some other dataset format, or simply to have a versioned
deliverable.

**Eligibility is deliberately weaker than the LeRobot path's.** Replay is a stub
in this version, so requiring a replay verdict would make every episode
ineligible and export nothing. An episode qualifies here when retarget has
produced a readable ``plan.h5``; whether it was replayed, labelled or screened
is *recorded* per episode rather than used to reject it. The manifest states
plainly what was and was not checked, so a consumer is never misled about the
provenance of what they are holding.

Units are metres, radians, seconds and microseconds throughout, and the
coordinate frame of every pose is named in the manifest rather than assumed.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from viki.contracts import Episode, cln_pose_keys
from viki.episode import read_status, stage_done
from viki.labeling import load_labels
from viki.retarget.archive import load_archive

logger = logging.getLogger(__name__)

SCHEMA = "viki_trajectory_bundle"
SCHEMA_VERSION = 1

#: Arrays copied verbatim from ``plan.h5``. Each is per-frame unless noted.
_PLAN_ARRAYS = (
    "timestamps",                       # (T,) host monotonic microseconds
    "q",                                # (T, nq) joint positions, rad
    "joint_velocity",                   # (T, nq) rad/s
    "joint_acceleration",               # (T, nq) rad/s^2
    "gripper_opening",                  # (T,) normalised 0 closed .. 1 open
    "gripper_opening_m",                # (T,) physical jaw separation, m
    "gripper_joint_position",           # (T,) drive joint, rad
    "omega",                            # (T,) per-frame evidence weight
    "target_position_calibration",      # (T, 3) commanded tool point, m
    "target_rotation_calibration",      # (T, 3, 3)
    "achieved_position_calibration",    # (T, 3) forward kinematics, m
    "achieved_rotation_calibration",    # (T, 3, 3)
    "position_error_m",                 # (T,)
    "orientation_error_rad",            # (T,)
    "q_approach",                       # (A, nq) home -> first frame, not per-frame
)


def _plan_scalar(arc, key, default=None):
    return arc[key] if key in arc else default


def _episode_bundle(ep: Episode) -> tuple[dict, dict]:
    """One episode's arrays and its provenance record."""
    with load_archive(ep.plan_h5) as arc:
        arrays = {}
        for key in _PLAN_ARRAYS:
            if key in arc:
                arrays[key] = np.asarray(arc[key])
        joint_names = json.loads(str(_plan_scalar(arc, "joint_names_json", "[]")))
        record = {
            "robot": str(_plan_scalar(arc, "robot", "")),
            "robot_key": str(_plan_scalar(arc, "robot_key", "")),
            "ee_frame": str(_plan_scalar(arc, "ee_frame", "")),
            "gripper_model": str(_plan_scalar(arc, "gripper_model", "")),
            "coordinate_frame": str(_plan_scalar(arc, "coordinate_frame", "")),
            "target_position_anchor": str(_plan_scalar(arc, "target_position_anchor", "")),
            "fps": float(_plan_scalar(arc, "fps", 0.0)),
            "dt": float(_plan_scalar(arc, "dt", 0.0)),
            "solver_status": str(_plan_scalar(arc, "solver_status", "")),
            "joint_names": joint_names,
            "source_pose": str(_plan_scalar(arc, "source_pose", "")),
            "retarget_config": json.loads(str(_plan_scalar(arc, "config_json", "{}"))),
            "retarget_metrics": json.loads(str(_plan_scalar(arc, "metrics_json", "{}"))),
        }

    # The human trajectory the plan was derived from. Carried so the bundle can
    # be audited against its source without the episode directory.
    if ep.cln_npz.exists():
        with np.load(ep.cln_npz, allow_pickle=False) as cln:
            from viki import config

            p_key, r_key = cln_pose_keys(
                cln.files, getattr(config, "PERCEPTION_HAND_POSE_SOURCE", "landmarks")
            )
            arrays["hand_position_rig"] = np.asarray(cln[p_key], np.float32)
            arrays["hand_rotation_rig"] = np.asarray(cln[r_key], np.float32)
            arrays["hand_valid"] = np.asarray(cln["valid"], bool)
            if "observed_mask" in cln.files:
                arrays["hand_observed_mask"] = np.asarray(cln["observed_mask"], bool)
            if "geometry_support_mask" in cln.files:
                arrays["hand_pose_supported"] = np.asarray(cln["geometry_support_mask"], bool)
            record["perception_profile"] = (
                str(cln["perception_profile"]) if "perception_profile" in cln.files else ""
            )
            record["hand_pose_source"] = (
                "hand_fit" if p_key.startswith("hand_fit") else "landmarks"
            )

    meta = json.loads(ep.meta_path.read_text()) if ep.meta_path.exists() else {}
    labels = load_labels(ep)
    status = read_status(ep)
    record.update({
        "episode_id": ep.id,
        "task": labels.task.strip() or str(meta.get("task", "")),
        "hand": labels.hand,
        "outcome": labels.outcome,
        "demonstrator": meta.get("demonstrator", ""),
        "calibration_preset": meta.get("calibration_preset", ""),
        "frames": int(len(arrays.get("q", []))),
        # Stated, not enforced: this version exports unscreened trajectories.
        "screening": {
            "replay_run": bool(stage_done(ep, "replay")),
            "replay_verdict": status.get("stages", {}).get("replay", {}).get("verdict"),
            "labelled": bool(labels.task.strip()),
            "outcome_rated": labels.outcome != "unrated",
        },
    })
    return arrays, record


def export_trajectories(
    episode_ids: list[str | Path],
    out_dir: str | Path,
    *,
    name: str | None = None,
) -> str:
    """Write a trajectory bundle. Returns the output directory path.

    Layout::

        <out>/dataset.json                     manifest, provenance, per-episode index
        <out>/episodes/<id>/trajectory.npz     the arrays
        <out>/episodes/<id>/meta.json          that episode's provenance record
    """
    out = Path(out_dir)
    (out / "episodes").mkdir(parents=True, exist_ok=True)

    index, skipped = [], []
    for raw_id in episode_ids:
        ep = Episode(root=Path(raw_id))
        if not stage_done(ep, "retarget") or not ep.plan_h5.exists():
            skipped.append({"episode_id": ep.id, "why": "retarget has not produced a plan"})
            logger.warning("skip %s: retarget has not produced a plan", ep.id)
            continue
        try:
            arrays, record = _episode_bundle(ep)
        except Exception as exc:  # noqa: BLE001 - one bad episode must not lose the rest
            skipped.append({"episode_id": ep.id, "why": f"{type(exc).__name__}: {exc}"})
            logger.warning("skip %s: %s", ep.id, exc, exc_info=True)
            continue
        dest = out / "episodes" / ep.id
        dest.mkdir(parents=True, exist_ok=True)
        np.savez_compressed(dest / "trajectory.npz", **arrays)
        (dest / "meta.json").write_text(json.dumps(record, indent=2, sort_keys=True))
        index.append(record)
        logger.info("exported %s (%d frames, task=%r)", ep.id, record["frames"], record["task"])

    if not index:
        raise RuntimeError(
            "no episodes could be exported: none has a retargeted plan.h5"
        )

    from viki.prepare.baseline import code_sha256

    robots = sorted({r["robot_key"] for r in index if r["robot_key"]})
    manifest = {
        "schema": SCHEMA,
        "schema_version": SCHEMA_VERSION,
        "name": name or out.name,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "code_sha256": code_sha256(),
        "units": {
            "position": "m", "rotation": "rotation matrix, row-major",
            "angle": "rad", "time": "s", "timestamps": "us (host monotonic)",
            "gripper_opening": "normalised, 0 closed .. 1 open",
        },
        "frames": {
            "poses": "calibration frame (ChArUco board origin) unless a key says _rig",
            "hand_*": "rig frame (reference camera); T_calibration_rig is in plan.h5",
        },
        "robots": robots,
        "episode_count": len(index),
        "total_frames": int(sum(r["frames"] for r in index)),
        # Say plainly what this version does and does not guarantee.
        "screening": (
            "NONE. Replay is a stub in this version, so trajectories are exported "
            "without being replayed, screened or necessarily labelled. Each "
            "episode's meta.json records what was actually run."
        ),
        "episodes": index,
        "skipped": skipped,
    }
    (out / "dataset.json").write_text(json.dumps(manifest, indent=2, sort_keys=True))
    logger.info("trajectory bundle: %d episode(s), %d frames -> %s",
                len(index), manifest["total_frames"], out)
    return str(out)
