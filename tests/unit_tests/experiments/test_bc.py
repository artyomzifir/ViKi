"""Split and action-contract checks without requiring the simulator."""

import numpy as np
import pytest

from experiments.bc import action_residual, load_rollouts, split_episodes
from experiments.observations import ACTION_SCHEMA, OBSERVATION_SCHEMA


def test_split_groups_entire_source_episodes_reproducibly():
    names = [f"episode-{index:02d}" for index in range(17)]
    first = split_episodes(names, seed=42)
    second = split_episodes(list(reversed(names)), seed=42)
    assert first == second
    assert {name for group in first.values() for name in group} == set(names)
    assert [len(first[key]) for key in ("train", "validation", "test")] == [12, 2, 3]
    assert len({name for group in first.values() for name in group}) == 17


def test_action_residual_uses_actual_joint_and_finger_positions():
    observation = np.zeros((2, 53), np.float32)
    observation[:, :6] = np.arange(6)
    observation[:, 12] = 0.3
    command = np.column_stack((observation[:, :6] + 0.02,
                               observation[:, 12] - 0.05))
    np.testing.assert_allclose(action_residual(observation, command),
                               [[0.02] * 6 + [-0.05]] * 2, atol=1e-6)
    with pytest.raises(ValueError, match="aligned"):
        action_residual(observation, command[:1])


def test_rollout_loader_requires_manifest_and_matching_schema(tmp_path):
    import json

    folder = tmp_path / "rollouts"
    folder.mkdir()
    (tmp_path / "manifest.json").write_text(json.dumps({
        "successful_count": 1,
        "episodes": [{"episode": "scene", "released_stack_success": True}],
    }))
    archive = folder / "scene.npz"
    np.savez_compressed(
        archive, observation=np.zeros((2, 53), np.float32),
        action=np.zeros((2, 7), np.float32), time_s=np.array([0.0, 0.02]),
        observation_schema=OBSERVATION_SCHEMA, action_schema=ACTION_SCHEMA,
        source_episode="scene",
    )
    assert list(load_rollouts(folder)) == ["scene"]

    np.savez_compressed(
        archive, observation=np.zeros((2, 53), np.float32),
        action=np.zeros((2, 7), np.float32), time_s=np.array([0.0, 0.02]),
        observation_schema="wrong", action_schema=ACTION_SCHEMA,
        source_episode="scene",
    )
    with pytest.raises(ValueError, match="schema"):
        load_rollouts(folder)

    archive.unlink()
    with pytest.raises(ValueError, match="manifest"):
        load_rollouts(folder)
