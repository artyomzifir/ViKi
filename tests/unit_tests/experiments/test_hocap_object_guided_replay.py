import json

import numpy as np
import pytest

pytest.importorskip("gymnasium")

from experiments.hocap_object_guided_replay import (
    _force_command, _position_action, run_search, run_trial,
)


def test_position_action_is_bounded():
    np.testing.assert_allclose(_position_action(np.array([0.01, -0.01, 0.002])),
                               [1.0, -1.0, 0.4, 0.0, 0.0])


def test_force_command_respects_band_and_joint_limits():
    assert _force_command({"left": 0.7, "right": 1.2}, 0.5, 1.0) == 1.0
    assert _force_command({"left": 1.0, "right": 1.2}, 0.5, 1.0) == 0.0
    assert _force_command({"left": 2.5, "right": 2.6}, 0.5, 1.0) == -1.0
    assert _force_command({"left": 0.0, "right": 0.0}, 0.8, 1.0) == 0.0


def test_invalid_trial_parameters_are_rejected_without_physics():
    with pytest.raises(ValueError):
        run_trial(None, 0, (0.0, 0.0), 0.0, 4.0)
    with pytest.raises(ValueError):
        run_trial(None, 0, (0.0, 0.0), 0.5, 4.0, wrist_roll_deg=180.0)
    with pytest.raises(ValueError):
        run_trial(None, 0, (float("nan"), 0.0), 0.5, 4.0)


def test_four_mesh_mode_requires_full_scene(tmp_path):
    source = tmp_path / "source.json"
    source.write_text(json.dumps({"input_files": {}}))
    with pytest.raises(ValueError, match="require --full-scene"):
        run_search(source, tmp_path / "output.json", seed=1,
                   full_scene_meshes=True)
