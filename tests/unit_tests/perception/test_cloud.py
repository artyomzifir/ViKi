"""viki.perception.cloud.build_cloud — synthetic RealSense-style episode (no k4a
blob → pinhole fallback): writes a parseable per-frame .bin + meta.json, points
land in the world bbox, and a coarser voxel yields fewer points."""

import json
import struct

import cv2
import numpy as np
import pytest

from viki import config
from viki.episode import new_episode, stage_done
from viki.perception.cloud import build_cloud


def _unpack(buf: bytes):
    n = struct.unpack_from("<i", buf, 0)[0]
    xyz = np.frombuffer(buf, np.float32, count=n * 3, offset=4).reshape(n, 3)
    rgb = np.frombuffer(buf, np.uint8, count=n * 3, offset=4 + n * 12).reshape(n, 3)
    return n, xyz, rgb


def _make_episode(tmp_path, frames=2):
    ep = new_episode(tmp_path)
    raw = ep.raw_dir
    (raw / "intrinsics.json").write_text(json.dumps({
        "cam0": {
            "color": {"fx": 500, "fy": 500, "cx": 32, "cy": 32, "width": 64, "height": 64},
            "depth": {"fx": 500, "fy": 500, "cx": 32, "cy": 32, "width": 64, "height": 64},
            "source": "sdk",
        }
    }))
    (raw / "extrinsics.json").write_text(json.dumps({
        "cam0": {"rvec": [0.0, 0.0, 0.0], "tvec": [0.0, 0.0, 0.0]}
    }))
    (raw / "timestamps.json").write_text(json.dumps(
        [{"sync_us": i * 66_000, "offsets_us": {}} for i in range(frames)]
    ))
    w = cv2.VideoWriter(str(raw / "cam0.mp4"), cv2.VideoWriter_fourcc(*"mp4v"), 15, (64, 64))
    for _ in range(frames):
        w.write(np.full((64, 64, 3), 120, np.uint8))
    w.release()
    ddir = raw / "cam0_depth"
    ddir.mkdir()
    for i in range(frames):
        np.save(ddir / f"{i:06d}.npy", np.full((64, 64), 1000, np.uint16))  # 1 m
    return ep


def test_build_cloud_writes_parseable_frames(tmp_path):
    ep = _make_episode(tmp_path)
    out = build_cloud(ep)

    assert stage_done(ep, "cloud")
    meta = json.loads((ep.cloud_dir / "meta.json").read_text())
    assert meta["n_frames"] == 2
    assert meta["fps"] == pytest.approx(1e6 / 66_000, rel=0.1)

    n, xyz, rgb = _unpack((ep.cloud_dir / "000000.bin").read_bytes())
    assert n > 0 and xyz.shape == (n, 3) and rgb.shape == (n, 3)
    # 1 m ahead, within the default workspace AABB
    assert np.all(xyz[:, 2] > 0.5) and np.all(np.abs(xyz[:, :2]) < 0.6)
    assert str(ep.cloud_dir) == out


def test_coarser_voxel_yields_fewer_points(tmp_path, monkeypatch):
    ep = _make_episode(tmp_path, frames=1)

    monkeypatch.setattr(config, "CLOUD_VOXEL_M", 0.0, raising=False)
    build_cloud(ep)
    fine, _, _ = _unpack((ep.cloud_dir / "000000.bin").read_bytes())

    monkeypatch.setattr(config, "CLOUD_VOXEL_M", 0.5, raising=False)
    build_cloud(ep)
    coarse, _, _ = _unpack((ep.cloud_dir / "000000.bin").read_bytes())

    assert coarse < fine
