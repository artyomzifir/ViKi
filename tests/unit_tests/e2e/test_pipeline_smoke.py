"""
End-to-end smoke test over a synthetic episode:
    rec.npz -> prepare -> retarget -> replay(dryrun) -> label -> export

Uses a hand-written rec.npz (no camera decode). Retarget is skipped when
Pinocchio/PINK is unavailable. Export is checked both ways: the trajectory
bundle runs for real and closes the chain, and the LeRobot path asserts its
clean "needs lerobot" error.
"""

import json
from pathlib import Path

import numpy as np
import pytest

from viki.contracts import HAND_LM_COUNT, Episode, EpisodeLabels, LM
from viki.episode import new_episode, stage_done
from viki.labeling import save_labels, validate_labels


def _synthetic_rec(ep: Episode, n: int = 24) -> None:
    """One camera, n frames, a plausible open hand drifting along +x."""
    base = {
        LM.WRIST: [0.0, 0.0, 0.5],
        LM.THUMB_CMC: [0.02, 0.03, 0.5],
        LM.INDEX_MCP: [0.03, 0.09, 0.5],
        LM.MIDDLE_MCP: [0.0, 0.10, 0.5],
        LM.RING_MCP: [-0.02, 0.09, 0.5],
        LM.PINKY_MCP: [-0.04, 0.08, 0.5],
    }
    pts = np.zeros((n, HAND_LM_COUNT, 3), dtype=np.float32)
    for t in range(n):
        for i in range(HAND_LM_COUNT):
            p = base.get(LM(i), [0.0, 0.05, 0.5])
            pts[t, i] = [p[0] + 0.001 * t, p[1], p[2]]
    np.savez_compressed(
        ep.rec_npz,
        device_ids=np.array(["cam0"] * n),
        timestamps=(np.arange(n) * 33_000).astype(np.int64),
        points=pts,
        landmark_ids=np.arange(HAND_LM_COUNT, dtype=np.int32),
        confidence=np.ones((n, HAND_LM_COUNT), dtype=np.float32),
    )


def test_pipeline_smoke(tmp_path):
    from viki.prepare.run import prepare_episode

    ep = new_episode(tmp_path)
    _synthetic_rec(ep)

    # prepare: rec.npz -> cln.npz
    cln = prepare_episode(ep)
    assert ep.cln_npz.exists() and stage_done(ep, "prepare")
    with np.load(cln) as d:
        assert {"positions", "rotations", "valid", "omega", "gripper"} <= set(d.files)
        n_frames = len(d["positions"])
        assert d["gripper"].dtype == np.float32
        assert np.all((d["gripper"] >= 0.0) & (d["gripper"] <= 1.0))
    assert n_frames > 0

    # retarget: the real PINK solve needs a network git-clone of the robot
    # description on first run, so it is covered by tests/unit_tests/retarget/
    # test_retarget_episode.py (skips cleanly offline) rather than here. Hand the
    # replay leg a synthetic plan.h5 so the artifact chain stays exercised.
    from viki.episode import mark_stage
    from viki.replay import replay_episode
    from viki.retarget.archive import write_hdf5_archive

    write_hdf5_archive(
        ep.plan_h5,
        {
            "q": np.zeros((n_frames, 6)), "dt": 1 / 30.0, "robot": "",
            "gripper_model": "binary",
            "gripper_command": np.zeros((n_frames, 1), dtype=np.float32),
        },
    )
    mark_stage(ep, "retarget", robot="")

    # replay: plan.h5 -> replay.h5 (dry-run)
    replay_episode(ep, driver="dryrun")
    assert ep.replay_h5.exists() and stage_done(ep, "replay")

    # label
    labels = EpisodeLabels(task="pick up the block", hand="right", outcome="good")
    validate_labels(labels, n_frames, for_export=True)
    save_labels(ep, labels)

    # export, trajectory bundle: the default path, and the one that completes
    # without optional dependencies. This is what closes the artifact chain.
    from viki.export import export_dataset, export_trajectories

    out = export_trajectories([str(ep.root)], str(tmp_path / "bundle"))
    manifest = json.loads((Path(out) / "dataset.json").read_text())
    assert manifest["episode_count"] == 1
    assert manifest["total_frames"] == n_frames
    assert manifest["episodes"][0]["task"] == "pick up the block"
    # Replay did run here, and the manifest must say so rather than blanket-claim
    # that nothing was screened.
    assert manifest["episodes"][0]["screening"]["replay_run"] is True
    traj = np.load(Path(out) / "episodes" / ep.id / "trajectory.npz", allow_pickle=False)
    assert traj["q"].shape == (n_frames, 6)

    # export, LeRobot: no lerobot in the test image -> clean install error
    with pytest.raises(RuntimeError, match="lerobot"):
        export_dataset([str(ep.root)], str(tmp_path / "ds"), fps=15)
