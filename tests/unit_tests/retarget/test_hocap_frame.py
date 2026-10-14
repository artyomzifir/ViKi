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
    config = config_from_options("ur10", {"target_position_anchor": "wrist"})
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
