"""
viki.export.run
---------------
Pipeline stage 6: labelled + screened episodes -> one LeRobot dataset.

An episode is eligible when: retarget + replay have run, its replay verdict is
not ``reject``, its label ``outcome`` is not ``bad``, and it has a non-empty
task string. Camera video is read from ``raw/`` and decimated to ``fps``.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np

from viki.contracts import Episode
from viki.episode import read_status, stage_done
from viki.labeling import load_labels, validate_labels
from viki.retarget.archive import load_archive

logger = logging.getLogger(__name__)


def _eligible(ep: Episode) -> tuple[bool, str]:
    if not stage_done(ep, "retarget"):
        return False, "retarget not run"
    if not stage_done(ep, "replay"):
        return False, "replay not run"
    verdict = read_status(ep).get("stages", {}).get("replay", {}).get("verdict")
    if verdict == "reject":
        return False, "replay verdict = reject"
    labels = load_labels(ep)
    if labels.outcome == "bad":
        return False, "outcome = bad"
    if not labels.task.strip():
        return False, "no task label"
    return True, ""


def _episode_frames(ep: Episode, fps: int) -> tuple[list[dict], str, int, list[str]]:
    """Assemble per-frame dicts from cln.npz + plan/replay .h5 (+ raw video refs)."""
    with np.load(ep.cln_npz) as cln:
        wrist_pos = np.asarray(cln["positions"], dtype=np.float32)
        wrist_rot = np.asarray(cln["rotations"], dtype=np.float32)
        omega = np.asarray(cln.get("omega", np.ones(len(wrist_pos))), dtype=np.float32)
        obj_rel = cln["T_obj_hand"] if "T_obj_hand" in cln else None

    replay_path = ep.replay_h5 if ep.replay_h5.exists() else ep.plan_h5
    with load_archive(replay_path) as arc:
        q = np.asarray(
            arc["q_attained"] if "q_attained" in arc else arc["q_scene_smooth"],
            dtype=np.float32,
        )
        grip = np.asarray(
            arc["gripper_attained"] if "gripper_attained" in arc else np.zeros(len(q)),
            dtype=np.float32,
        )
        resid = np.asarray(
            arc["controller_residual"] if "controller_residual" in arc else np.full(len(q), np.nan),
            dtype=np.float32,
        )
        robot = str(arc["robot"]) if "robot" in arc else ""

    n = min(len(q), len(wrist_pos))
    labels = load_labels(ep)
    phases = _phase_ids(labels, n)

    frames: list[dict] = []
    for t in range(n):
        wp = np.eye(4, dtype=np.float32)
        wp[:3, :3] = wrist_rot[t]
        wp[:3, 3] = wrist_pos[t]
        state = np.concatenate([q[t], [grip[t]]]).astype(np.float32)
        nxt = np.concatenate([q[min(t + 1, n - 1)], [grip[min(t + 1, n - 1)]]]).astype(np.float32)
        frames.append(
            {
                "observation.state": state,
                "action": nxt,
                "annotation.wrist_pose": wp.reshape(16),
                "annotation.object_relative": (
                    np.asarray(obj_rel[t], dtype=np.float32).reshape(16)
                    if obj_rel is not None
                    else np.zeros(16, dtype=np.float32)  # placeholder
                ),
                "annotation.confidence": np.float32(omega[min(t, len(omega) - 1)]),
                "annotation.replay_residual": np.float32(resid[t]),
                "annotation.phase": np.int64(phases[t]),
            }
        )
    cams = sorted(p.stem for p in ep.raw_dir.glob("*.mp4"))
    return frames, robot, q.shape[1], cams


def _phase_ids(labels, n: int) -> np.ndarray:
    ids = np.zeros(n, dtype=np.int64)
    for i, seg in enumerate(sorted(labels.segments, key=lambda s: s.start), start=1):
        ids[seg.start : seg.end] = i
    return ids


def export_dataset(
    episode_ids: list[str | Path],
    out_dir: str | Path,
    *,
    fps: int = 15,
) -> str:
    """Write a LeRobot dataset. Returns the output directory path."""
    from viki.export.lerobot import LeRobotWriter

    eps = [Episode(root=Path(e)) for e in episode_ids]
    picked: list[tuple[Episode, list[dict], str]] = []
    robot = ""
    nq = 0
    cams: list[str] = []
    for ep in eps:
        ok, why = _eligible(ep)
        if not ok:
            logger.warning("skip %s: %s", ep.id, why)
            continue
        labels = load_labels(ep)
        frames, robot, nq, cams = _episode_frames(ep, fps)
        validate_labels(labels, len(frames), for_export=True)
        picked.append((ep, frames, labels.task))

    if not picked:
        raise RuntimeError("no eligible episodes to export")

    writer = LeRobotWriter(out_dir, fps=fps, robot_type=robot, cams=cams, nq=nq)
    for ep, frames, task in picked:
        writer.add_episode(frames, task)
        logger.info("exported %s (%d frames, task=%r)", ep.id, len(frames), task)
    writer.finalize()
    return str(out_dir)
