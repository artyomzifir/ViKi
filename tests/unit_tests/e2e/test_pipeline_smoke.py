"""
End-to-end smoke test over a synthetic episode:
    rec.npz -> prepare -> retarget -> replay(dryrun) -> label -> export

Uses a hand-written rec.npz (no camera decode). Retarget is skipped when
Pinocchio/PINK is unavailable; export asserts the clean "needs lerobot" error.
"""

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
    assert n_frames > 0

    # retarget is exercised separately (slow IK + model download); here we hand a
    # synthetic plan.h5 to the replay leg so the artifact chain stays covered.
    from viki.replay import replay_episode
    from viki.retarget.archive import write_hdf5_archive

    write_hdf5_archive(
        ep.plan_h5,
        {
            "q_scene_smooth": np.zeros((n_frames, 6), dtype=np.float64),
            "dt": 1 / 30.0,
            "robot": "",  # empty -> screen skips the model-load joint check (fast)
        },
    )
    from viki.episode import mark_stage

    mark_stage(ep, "retarget", robot="")

    # replay: plan.h5 -> replay.h5 (dry-run)
    replay_episode(ep, driver="dryrun")
    assert ep.replay_h5.exists() and stage_done(ep, "replay")

    # label
    labels = EpisodeLabels(task="pick up the block", hand="right", outcome="good")
    validate_labels(labels, n_frames, for_export=True)
    save_labels(ep, labels)

    # export: no lerobot in the test image -> clean install error
    from viki.export import export_dataset

    with pytest.raises(RuntimeError, match="lerobot"):
        export_dataset([str(ep.root)], str(tmp_path / "ds"), fps=15)
