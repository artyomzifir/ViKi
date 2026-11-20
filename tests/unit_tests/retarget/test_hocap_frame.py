"""HO-Cap world coordinates must not be silently interpreted as rig coordinates."""

import numpy as np
import pytest

from viki.retarget.run import config_from_options, load_targets


def test_hocap_targets_require_explicit_world_frame(tmp_path):
    path = tmp_path / "cln.npz"
    np.savez_compressed(
        path,
        timestamps=np.arange(3, dtype=np.float64) / 30.0,
        positions=np.tile([0.2, 0.3, 0.4], (3, 1)),
        rotations=np.tile(np.eye(3), (3, 1, 1)),
        valid=np.ones(3, dtype=bool),
        gripper=np.ones(3, dtype=np.float32),
        coordinate_frame="hocap_tag_1",
    )
    config = config_from_options("ur10", {"target_position_anchor": "wrist", "pose_source": "landmarks"})
    with pytest.raises(ValueError, match="coordinate_frame"):
        load_targets(path, config)
    targets = load_targets(path, config, source_frame="hocap_tag_1")
    np.testing.assert_allclose(targets.position_rig, [[0.2, 0.3, 0.4]] * 3)


def test_hocap_targets_reject_mislabeled_world_frame(tmp_path):
    path = tmp_path / "cln.npz"
    np.savez_compressed(
        path,
        timestamps=np.arange(3, dtype=np.float64) / 30.0,
        positions=np.tile([0.2, 0.3, 0.4], (3, 1)),
        rotations=np.tile(np.eye(3), (3, 1, 1)),
        valid=np.ones(3, dtype=bool),
        gripper=np.ones(3, dtype=np.float32),
        coordinate_frame="rig",
    )
    config = config_from_options("ur10", {"target_position_anchor": "wrist"})
    with pytest.raises(ValueError, match="hocap_tag_1"):
        load_targets(path, config, source_frame="hocap_tag_1")


def test_hocap_hand_fit_request_rejects_missing_fitted_arrays(tmp_path):
    path = tmp_path / "cln.npz"
    np.savez(path, timestamps=np.arange(2), positions=np.zeros((2, 3)),
             rotations=np.tile(np.eye(3), (2, 1, 1)), valid=np.ones(2, bool),
             coordinate_frame="hocap_tag_1")
    config = config_from_options("ur10", {"pose_source": "hand_fit"})
    with pytest.raises(ValueError, match="requested hand_fit"):
        load_targets(path, config, source_frame="hocap_tag_1")


def test_hocap_hand_fit_rejects_landmark_pinch_override(tmp_path):
    path = tmp_path / "cln.npz"
    np.savez(path, timestamps=np.arange(2), positions=np.zeros((2, 3)),
             rotations=np.tile(np.eye(3), (2, 1, 1)), valid=np.ones(2, bool),
             coordinate_frame="hocap_tag_1",
             hand_fit_positions=np.zeros((2, 3)),
             hand_fit_rotations=np.tile(np.eye(3), (2, 1, 1)))
    config = config_from_options("ur10", {"pose_source": "hand_fit",
                                            "target_position_anchor": "pinch_center"})
    with pytest.raises(ValueError, match="landmark-derived position"):
        load_targets(path, config, source_frame="hocap_tag_1")
