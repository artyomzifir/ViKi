"""Post-IK object-relative tracks keep rig and calibration poses distinct."""

import json

import numpy as np
import pytest

from viki.episode import new_episode, stage_done
from viki.object_centric import build_object_relative_episode
from viki.retarget.archive import write_hdf5_archive


def _episode_with_tracks(tmp_path, *, clock_mismatch=False):
    ep = new_episode(tmp_path)
    anchor = np.eye(4)
    anchor[:3, :3] = [[0, -1, 0], [1, 0, 0], [0, 0, 1]]
    anchor[:3, 3] = [10, 0, 0]
    (ep.raw_dir / "world_anchor.json").write_text(
        json.dumps({"T_world_display": anchor.tolist()})
    )
    rotations = np.broadcast_to(np.eye(3), (2, 2, 3, 3)).copy()
    positions = np.asarray([
        [[1, 0, 0], [0, 1, 0]],
        [[1, 0, 0], [0, 1, 0]],
    ], dtype=np.float32)
    ep.object_models_npz.parent.mkdir(parents=True)
    np.savez_compressed(
        ep.object_models_npz,
        schema=np.asarray("viki_object_models_v2"),
        object_ids=np.asarray([2, 3], dtype=np.int32),
        object_labels=np.asarray(["manipulated_object", "other_object"]),
        translation_world=positions,
        rotation_world_object=rotations,
        track_confidence=np.ones((2, 2), dtype=np.float32),
        rotation_information=np.ones((2, 2, 3), dtype=np.float32),
        contact=np.zeros((2, 2), dtype=bool),
    )
    hand_rotations = np.broadcast_to(np.eye(3), (2, 3, 3)).copy()
    np.savez_compressed(
        ep.cln_npz,
        coordinate_frame=np.asarray("rig"),
        timestamps=np.asarray([0, 1], dtype=np.int64),
        positions=np.zeros((2, 3)),
        rotations=hand_rotations,
        hand_fit_positions=np.asarray([[2, 0, 0], [3, 0, 0]], dtype=np.float32),
        hand_fit_rotations=hand_rotations,
        valid=np.asarray([True, False]),
    )
    target_rotation = np.broadcast_to(anchor[:3, :3], (2, 3, 3)).copy()
    write_hdf5_archive(
        ep.plan_h5,
        {
            "timestamps": np.asarray([0, 2] if clock_mismatch else [0, 1], dtype=np.int64),
            "target_position_calibration": np.asarray([[10, 3, 0], [10, 4, 0]]),
            "target_rotation_calibration": target_rotation,
            "achieved_position_calibration": np.asarray([[10, 2, 0], [10, 3, 0]]),
            "achieved_rotation_calibration": target_rotation,
            "gripper_opening": np.asarray([1.0, 0.2]),
            "source_pose": "hand_fit",
        },
    )
    return ep


def test_relative_tracks_use_object_frame_and_mask_invalid_hand(tmp_path):
    ep = _episode_with_tracks(tmp_path)
    output = build_object_relative_episode(ep)

    with np.load(output, allow_pickle=False) as tracks:
        assert str(tracks["schema"]) == "viki_object_relative_v1"
        assert tracks["object_ids"].tolist() == [2, 3]
        assert str(tracks["hand_pose_source"]) == "hand_fit"
        assert tracks["T_object_hand"].shape == (2, 2, 4, 4)
        np.testing.assert_allclose(tracks["T_object_hand"][0, 0, :3, 3], [1, 0, 0])
        np.testing.assert_allclose(tracks["T_object_target_tcp"][0, 0, :3, 3], [2, 0, 0])
        np.testing.assert_allclose(tracks["T_object_achieved_tcp"][0, 0, :3, 3], [1, 0, 0])
        assert tracks["relative_valid"].tolist() == [[True, True], [False, False]]
        assert np.isnan(tracks["T_object_target_tcp"][1]).all()
        assert np.isfinite(tracks["T_object_achieved_tcp"]).all()
    assert stage_done(ep, "object_relative")


def test_relative_tracks_reject_misaligned_frame_clocks(tmp_path):
    ep = _episode_with_tracks(tmp_path, clock_mismatch=True)
    with pytest.raises(ValueError, match="one frame clock"):
        build_object_relative_episode(ep)
    assert not ep.object_relative_npz.exists()
    assert not stage_done(ep, "object_relative")
