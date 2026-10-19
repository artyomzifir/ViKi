"""CLI wiring for the benchmark scorer."""

import json

from viki import benchmark
from viki.cli import main


def test_hocap_evaluate_writes_optional_report(tmp_path, monkeypatch):
    episode = tmp_path / "episode"
    output = tmp_path / "score.json"
    called = []
    def evaluate(path, *, expected_profile):
        called.append(expected_profile)
        return {"episode": str(path), "mpjpe_mm": 12.5}
    monkeypatch.setattr(benchmark, "evaluate_episode", evaluate)
    assert main(["hocap-evaluate", str(episode), "--out", str(output)]) == 0
    assert called == ["hocap-stable-fused-hand-v4"]
    assert json.loads(output.read_text()) == {
        "episode": str(episode),
        "mpjpe_mm": 12.5,
    }


def test_hocap_evaluate_legacy_requires_explicit_override(tmp_path, monkeypatch):
    called = []
    def evaluate(_path, *, expected_profile):
        called.append(expected_profile)
        return {}
    monkeypatch.setattr(benchmark, "evaluate_episode", evaluate)
    assert main(["hocap-evaluate", str(tmp_path), "--allow-legacy-landmarks"]) == 0
    assert called == [None]
