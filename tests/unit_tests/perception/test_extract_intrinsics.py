"""extract must read both raw/intrinsics.json layouts (regression: PR added the
nested {"color":..,"depth":..} form and broke _depth_K which expected flat)."""

import json

import numpy as np
import pytest

from viki.perception.extract import _depth_K


def test_depth_K_nested_form():
    entry = {
        "color": {"fx": 600, "fy": 600, "cx": 640, "cy": 360, "width": 1280, "height": 720},
        "depth": {"fx": 195.5, "fy": 195.6, "cx": 258.0, "cy": 255.1, "width": 640, "height": 576},
        "source": "sdk",
    }
    K = _depth_K(entry)
    assert K is not None
    assert K[0, 0] == pytest.approx(195.5) and K[0, 2] == pytest.approx(258.0)


def test_depth_K_flat_legacy_form():
    K = _depth_K({"fx": 500.0, "fy": 500.0, "cx": 320.0, "cy": 240.0})
    assert K is not None and K[1, 1] == pytest.approx(500.0)


def test_depth_K_empty_or_missing():
    assert _depth_K({}) is None
    assert _depth_K({"source": "none", "depth": None}) is None
    assert _depth_K({"color": {"width": 1280}}) is None  # no fx


def test_extract_episode_reads_nested_intrinsics(tmp_path):
    """End-to-end past the intrinsics read: synthetic raw/ with the nested form,
    two black frames -> no hand -> 0 records, but no KeyError and a rec.npz."""
    import cv2

    from viki.episode import new_episode, stage_done
    from viki.perception.extract import extract_episode

    ep = new_episode(tmp_path)
    raw = ep.raw_dir
    (raw / "intrinsics.json").write_text(json.dumps({
        "cam0": {
            "color": {"fx": 600, "fy": 600, "cx": 320, "cy": 240, "width": 640, "height": 480},
            "depth": {"fx": 300, "fy": 300, "cx": 256, "cy": 256, "width": 512, "height": 512},
            "source": "sdk",
        }
    }))
    (raw / "extrinsics.json").write_text(json.dumps({
        "cam0": {"rvec": [0.0, 0.0, 0.0], "tvec": [0.0, 0.0, 1.0]}
    }))
    w = cv2.VideoWriter(str(raw / "cam0.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (64, 64))
    for _ in range(2):
        w.write(np.zeros((64, 64, 3), np.uint8))
    w.release()
    ddir = raw / "cam0_depth"
    ddir.mkdir()
    for i in range(2):
        np.save(ddir / f"{i:06d}.npy", np.zeros((512, 512), np.uint16))

    out = extract_episode(ep)
    assert ep.rec_npz.exists() and stage_done(ep, "extract")
    with np.load(out) as d:
        assert set(("device_ids", "timestamps", "points", "confidence")) <= set(d.files)
