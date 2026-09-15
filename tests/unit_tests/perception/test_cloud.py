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
from viki.perception.cloud import (
    _bbox_mask,
    _camera_samples,
    _depth_edge_keep_mask,
    _voxel_downsample_indices,
    build_cloud,
)


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


def test_camera_samples_retain_projected_colour_pixels():
    color = np.zeros((4, 5, 3), np.uint8)
    color[2, 3] = [10, 20, 30]
    depth = np.zeros((4, 5), np.uint16)
    depth[2, 3] = 1000
    K = np.array([[100.0, 0.0, 2.0], [0.0, 100.0, 2.0], [0.0, 0.0, 1.0]])

    xyz, rgb, uv = _camera_samples(color, depth, 1, K, None, np.eye(4))

    np.testing.assert_array_equal(uv, [[3, 2]])
    np.testing.assert_array_equal(rgb, [[30, 20, 10]])
    np.testing.assert_allclose(xyz, [[0.01, 0.0, 1.0]])


def test_camera_samples_can_use_exact_color_to_depth_warp():
    class Calibration:
        def color_deproject_maps(self, height, width):
            rays = np.zeros((height, width, 3), np.float64)
            rays[:, :, 2] = 1.0
            return rays, np.zeros_like(rays)

        def align_color_to_depth(self, _color, depth):
            aligned = np.zeros((*depth.shape, 3), np.uint8)
            aligned[1, 1] = [11, 22, 33]
            return aligned

    color = np.zeros((2, 2, 3), np.uint8)
    color[0, 0] = [1, 2, 3]
    depth = np.zeros((2, 2), np.uint16)
    depth[1, 1] = 1000
    K = np.eye(3)

    _xyz, rgb, uv = _camera_samples(
        color,
        depth,
        1,
        K,
        Calibration(),
        np.eye(4),
        exact_color_projection=True,
    )

    np.testing.assert_array_equal(rgb, [[33, 22, 11]])
    np.testing.assert_array_equal(uv, [[0, 0]])


def test_voxel_downsample_indices_preserve_source_rows():
    xyz = np.array([[0.0, 0, 0], [0.001, 0, 0], [0.02, 0, 0]], np.float32)
    idx = _voxel_downsample_indices(xyz, 0.01)
    np.testing.assert_array_equal(idx, [0, 2])


def test_bbox_mask_keeps_only_workspace_points():
    xyz = np.array([[0.0, 0.0, 0.5], [2.0, 0.0, 0.5]], np.float32)
    keep = _bbox_mask(xyz, [-1, 1, -1, 1, 0, 1])
    np.testing.assert_array_equal(keep, [True, False])


def test_depth_edge_filter_rejects_metric_discontinuity_not_flat_surface():
    depth = np.full((9, 13), 1000, np.uint16)
    depth[:, 7:] = 1400
    K = np.array([[100.0, 0, 6.0], [0, 100.0, 4.0], [0, 0, 1.0]])

    keep = _depth_edge_keep_mask(
        depth, K, radius_rad=0.01, jump_mm=30.0, jump_relative=0.02
    )

    assert keep[:, :5].all()
    assert not keep[:, 6:8].any()
    assert keep[:, 8:].all()


def test_depth_edge_filter_scales_pixel_radius_with_focal_length():
    low = np.full((9, 13), 1000, np.uint16)
    low[:, 7:] = 1400
    high = np.repeat(np.repeat(low, 2, axis=0), 2, axis=1)
    K_low = np.array([[100.0, 0, 6.0], [0, 100.0, 4.0], [0, 0, 1.0]])
    K_high = K_low.copy()
    K_high[:2] *= 2.0
    K_high[2, 2] = 1.0

    keep_low = _depth_edge_keep_mask(
        low, K_low, radius_rad=0.01, jump_mm=30.0, jump_relative=0.02
    )
    keep_high = _depth_edge_keep_mask(
        high, K_high, radius_rad=0.01, jump_mm=30.0, jump_relative=0.02
    )

    low_rejected = np.flatnonzero(~keep_low[4])
    high_rejected = np.flatnonzero(~keep_high[8])
    assert np.ptp(high_rejected) == 2 * np.ptp(low_rejected) + 1
