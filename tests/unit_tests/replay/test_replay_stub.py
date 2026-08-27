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
    g = np.zeros(20, dtype=bool)
    log = DryRunDriver().execute(q, g, dt=1 / 30)
    np.testing.assert_array_equal(log.q_attained, q)
    assert np.isnan(log.controller_residual).all()


def test_ur3_driver_not_implemented():
    with pytest.raises(NotImplementedError):
        load_driver("ur3")


def test_screen_dryrun_when_residual_unmeasured():
    q = np.zeros((10, 6))
    v = screen(q, np.full(10, np.nan), robot_description="")
    assert v.verdict == "dry-run"


def test_screen_flags_tracking_fault():
    q = np.zeros((10, 6))
    resid = np.full(10, 0.2)  # above default 0.05 threshold
    v = screen(q, resid, robot_description="")
    assert v.verdict == "reject" and v.cause == "tracking_fault"


def test_replay_episode_writes_replay_h5(tmp_path):
    ep = Episode(root=tmp_path / "ep0")
    ep.raw_dir.mkdir(parents=True)
    write_hdf5_archive(
        ep.plan_h5,
        {"q_scene_smooth": np.zeros((15, 6)), "dt": 1 / 30.0, "robot": "ur3_description"},
    )
    out = replay_episode(ep, driver="dryrun")
    assert out == str(ep.replay_h5) and ep.replay_h5.exists()
    with load_archive(ep.replay_h5) as arc:
        assert set(arc.files) == set(REPLAY_KEYS)
        assert str(arc["verdict"]) in ("dry-run", "pass", "reject")
