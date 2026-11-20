"""viki.replay dry-run driver, screening, and replay.h5 write."""

import numpy as np
import pytest

from viki.contracts import Episode, REPLAY_KEYS
from viki.replay.driver import DryRunDriver, load_driver
from viki.replay.run import replay_episode
from viki.replay.screen import screen
from viki.retarget.archive import load_archive, write_hdf5_archive


def test_dryrun_driver_echoes_plan():
    q = np.random.rand(20, 6)
    g = np.linspace(0.0, 1.0, 20)
    log = DryRunDriver().execute(q, g, dt=1 / 30)
    np.testing.assert_array_equal(log.q_attained, q)
    np.testing.assert_array_equal(log.gripper_attained, g)
    assert np.isnan(log.controller_residual).all()


def test_removed_driver_is_rejected():
    with pytest.raises(ValueError, match="known: dryrun"):
        load_driver("ur3")


def test_valid_plan_stays_dryrun():
    q = np.zeros((10, 6))
    v = screen(q, robot_description="")
    assert v.verdict == "dry-run"


@pytest.mark.parametrize("q", [np.zeros((0, 6)), np.array([[np.nan]]), np.zeros(6)])
def test_screen_rejects_invalid_plan(q):
    with pytest.raises(ValueError, match="finite, non-empty"):
        screen(q, robot_description="")


def test_screen_uses_physical_ur_limits_for_continuous_pinocchio_joint():
    q = np.zeros((2, 6))
    q[1, -1] = 2.0 * np.pi + 0.01
    verdict = screen(
        q,
        robot_description="ur3_official_description",
    )
    assert verdict.verdict == "reject"
    assert verdict.cause == "joint_limit"


def test_replay_episode_executes_approach_but_archives_demo_frames(tmp_path, monkeypatch):
    ep = Episode(root=tmp_path / "ep0")
    ep.raw_dir.mkdir(parents=True)
    q_approach = np.full((3, 6), -0.25)
    write_hdf5_archive(
        ep.plan_h5,
        {
            "q": np.zeros((15, 6)), "dt": 1 / 30.0,
            "q_approach": q_approach,
            "robot": "ur3_official_description", "gripper_model": "binary",
            "gripper_command": np.zeros((15, 1), dtype=np.float32),
        },
    )
    driver = DryRunDriver()
    executed = {}
    original_execute = driver.execute

    def record_execute(q, gripper, dt):
        executed["q"] = np.asarray(q).copy()
        executed["gripper"] = np.asarray(gripper).copy()
        return original_execute(q, gripper, dt)

    driver.execute = record_execute
    monkeypatch.setattr("viki.replay.run.load_driver", lambda _name: driver)
    out = replay_episode(ep, driver="dryrun")
    assert out == str(ep.replay_h5) and ep.replay_h5.exists()
    assert executed["q"].shape == (18, 6)
    np.testing.assert_array_equal(executed["q"][:3], q_approach)
    np.testing.assert_array_equal(executed["gripper"], np.ones(18))
    with load_archive(ep.replay_h5) as arc:
        assert set(arc.files) == set(REPLAY_KEYS)
        assert str(arc["verdict"]) == "dry-run"
        assert np.isnan(np.asarray(arc["controller_residual"])).all()
        assert np.asarray(arc["q_attained"]).shape == (15, 6)
        # Legacy binary ``closed=0`` is migrated to continuous fully-open=1.
        np.testing.assert_array_equal(arc["gripper_attained"], np.ones(15))
