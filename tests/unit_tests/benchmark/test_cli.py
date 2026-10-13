"""CLI wiring for the benchmark scorer."""

import json

from viki import benchmark
from viki.cli import main


def test_hocap_evaluate_writes_optional_report(tmp_path, monkeypatch):
    episode = tmp_path / "episode"
    output = tmp_path / "score.json"
    monkeypatch.setattr(
        benchmark, "evaluate_episode",
        lambda path: {"episode": str(path), "mpjpe_mm": 12.5},
    )
    assert main(["hocap-evaluate", str(episode), "--out", str(output)]) == 0
    assert json.loads(output.read_text()) == {
        "episode": str(episode),
        "mpjpe_mm": 12.5,
    }
