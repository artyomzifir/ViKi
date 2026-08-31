"""extract picks the real SDK colour→depth projector when the episode carries a
k4a calibration blob, and falls back to identity (with a warning) when it does
not."""

import logging

from viki.perception import extract as ex


def test_load_projector_none_without_blob(tmp_path):
    (tmp_path / "meta.json").write_text("{}")
    assert ex._load_projector(tmp_path, "kinect_0", {}) is None


def test_load_projector_uses_k4a_when_available(tmp_path, monkeypatch):
    sentinel = object()
    monkeypatch.setattr(
        "viki.perception.k4a_offline.K4ACalibration.from_episode",
        classmethod(lambda cls, raw, dev, meta: sentinel),
    )
    assert ex._load_projector(tmp_path, "kinect_0", {"cameras": {}}) is sentinel


def test_load_projector_swallows_errors(tmp_path, monkeypatch):
    def boom(cls, raw, dev, meta):
        raise RuntimeError("bad blob")

    monkeypatch.setattr(
        "viki.perception.k4a_offline.K4ACalibration.from_episode", classmethod(boom)
    )
    assert ex._load_projector(tmp_path, "kinect_0", {}) is None


def test_extract_warns_on_identity_fallback(tmp_path, caplog):
    import json

    import cv2
    import numpy as np

    from viki.episode import new_episode

    ep = new_episode(tmp_path)
    raw = ep.raw_dir
    (raw / "intrinsics.json").write_text(json.dumps({
        "cam0": {"depth": {"fx": 300, "fy": 300, "cx": 32, "cy": 32, "width": 64, "height": 64}}
    }))
    (raw / "extrinsics.json").write_text(json.dumps({"cam0": {"rvec": [0, 0, 0], "tvec": [0, 0, 1]}}))
    w = cv2.VideoWriter(str(raw / "cam0.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (64, 64))
    w.write(np.zeros((64, 64, 3), np.uint8))
    w.release()

    with caplog.at_level(logging.WARNING):
        ex.extract_episode(ep)
    assert any("identity colour→depth" in r.message for r in caplog.records)
