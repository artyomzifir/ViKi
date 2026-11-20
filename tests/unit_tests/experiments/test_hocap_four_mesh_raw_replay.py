from pathlib import Path

import pytest

pytest.importorskip("mujoco")

from experiments.hocap_four_mesh_raw_replay import replay


def test_raw_replay_rejects_nonpositive_time_scale_without_loading_assets(tmp_path):
    with pytest.raises(ValueError, match="time scale"):
        replay(Path("missing"), Path("missing"), tmp_path / "report.json",
               time_scale=0)
