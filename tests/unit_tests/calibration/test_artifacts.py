"""
Artifact store for the split setup flow — ``viki.calibration.artifacts``.

Covers the round-trip + hashing + staleness contract that the wizard and the
record-start guard depend on, and the migration from a legacy single-file
preset into ``calib/<preset>/``.
"""

from __future__ import annotations

import json

import cv2
import numpy as np
import pytest

from viki.calibration import artifacts, presets
from viki.contracts import CalibrationExtrinsics


@pytest.fixture()
def store(tmp_path, monkeypatch):
    d = tmp_path / "calibrations"
    d.mkdir()
    monkeypatch.setattr(presets, "PRESETS_DIR", d)
    monkeypatch.setattr(artifacts, "PRESETS_DIR", d)
    return d


def _rand_T(seed: int) -> np.ndarray:
    rng = np.random.default_rng(seed)
    rvec = rng.normal(0, 0.6, 3)
    R, _ = cv2.Rodrigues(rvec)
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = rng.normal(0, 0.4, 3)
    return T


def test_extrinsics_round_trip_and_hash(store):
    devices = {"cam_a": np.eye(4), "cam_b": _rand_T(1)}
    artifacts.write_extrinsics(
        "rig", reference_device="cam_a", devices=devices,
        sets=[{"set_id": "s0", "captured_at": None, "observations": {}}],
        solve={"method": "bundle", "rms_reproj_px": {"cam_a": 0.4}, "n_sets": 1, "n_points": 40},
    )
    got = artifacts.read_extrinsics("rig")
    assert got["reference_device"] == "cam_a"
    assert got["schema"] == artifacts.EXTRINSICS_SCHEMA
    np.testing.assert_allclose(artifacts.device_transforms("rig")["cam_b"], devices["cam_b"])

    h1 = artifacts.extrinsics_hash("rig")
    assert h1 and len(h1) == 64
    # rewriting identical content -> identical hash (sorted keys, stable dump)
    artifacts.write_extrinsics(
        "rig", reference_device="cam_a", devices=devices,
        sets=[{"set_id": "s0", "captured_at": None, "observations": {}}],
        solve={"method": "bundle", "rms_reproj_px": {"cam_a": 0.4}, "n_sets": 1, "n_points": 40},
    )
    # created_at changes each write, so the hash is expected to move; assert it
    # is a function of bytes by hashing a hand-frozen file instead:
    p = artifacts._extrinsics_path("rig")
    frozen = json.loads(p.read_text())
    frozen["created_at"] = "frozen"
    artifacts._dump(p, frozen)
    assert artifacts.extrinsics_hash("rig") == artifacts._sha256(p)


def test_reference_device_must_be_present(store):
    with pytest.raises(ValueError):
        artifacts.write_extrinsics(
            "rig", reference_device="missing", devices={"cam_a": np.eye(4)},
            sets=[], solve={},
        )


def test_as_camera_extrinsics_matches_contract(store):
    T_ref_cam = _rand_T(7)
    pair = artifacts.as_camera_extrinsics(T_ref_cam)
    tm = CalibrationExtrinsics(
        rvec=np.asarray(pair["rvec"]), tvec=np.asarray(pair["tvec"])
    ).transform_matrix
    np.testing.assert_allclose(tm, T_ref_cam, atol=1e-9)


def test_world_anchor_and_validation_staleness(store):
    artifacts.write_extrinsics(
        "rig", reference_device="cam_a", devices={"cam_a": np.eye(4)},
        sets=[], solve={},
    )
    h0 = artifacts.extrinsics_hash("rig")
    artifacts.write_world_anchor("rig", T_world_display=_rand_T(2), observations={})
    artifacts.write_validation("rig", verdict="green", pairs=[])
    assert not artifacts.world_anchor_stale("rig")
    assert not artifacts.validation_stale("rig")

    # re-solve extrinsics -> both dependents go stale
    import time as _t
    _t.sleep(1.1)  # created_at has second resolution
    artifacts.write_extrinsics(
        "rig", reference_device="cam_a", devices={"cam_a": np.eye(4)},
        sets=[], solve={"method": "bundle"},
    )
    assert artifacts.extrinsics_hash("rig") != h0
    assert artifacts.world_anchor_stale("rig")
    assert artifacts.validation_stale("rig")


def test_background_plate_mask(store):
    depth = np.full((4, 5), 1500.0, np.float32)
    valid = np.ones((4, 5), bool)
    valid[0, 0] = False
    artifacts.write_background("rig", {"cam_a": (depth, valid)})
    d, v = artifacts.read_background("rig", "cam_a")
    np.testing.assert_array_equal(v, valid)
    masked = artifacts.background_depth_masked("rig", "cam_a")
    assert masked[0, 0] == 0.0 and masked[1, 1] == 1500.0
    assert artifacts.background_devices("rig") == ["cam_a"]


def test_record_ready_gate(store):
    ok, why = artifacts.record_ready("rig")
    assert not ok and "extrinsics" in why

    artifacts.write_extrinsics(
        "rig", reference_device="cam_a", devices={"cam_a": np.eye(4)},
        sets=[], solve={},
    )
    ok, why = artifacts.record_ready("rig")
    assert not ok and "anchor" in why.lower()

    artifacts.write_world_anchor("rig", T_world_display=np.eye(4), observations={})
    ok, why = artifacts.record_ready("rig")
    assert not ok and "background" in why.lower()

    artifacts.write_background(
        "rig", {"cam_a": (np.ones((2, 2), np.float32), np.ones((2, 2), bool))}
    )
    ok, why = artifacts.record_ready("rig")
    assert not ok and "validation" in why.lower()

    artifacts.write_validation("rig", verdict="amber", pairs=[])
    ok, why = artifacts.record_ready("rig")
    assert not ok and "amber" in why.lower()
    ok, why = artifacts.record_ready("rig", allow_amber=True)
    assert ok and why == ""

    artifacts.write_validation("rig", verdict="red", pairs=[])
    ok, why = artifacts.record_ready("rig", allow_amber=True)
    assert not ok and "red" in why.lower()

    artifacts.write_validation("rig", verdict="green", pairs=[])
    ok, why = artifacts.record_ready("rig")
    assert ok


def test_migrate_legacy_v2_preset(store):
    # two cameras, each with a board->camera pose; build a legacy v2 file
    rng = np.random.default_rng(0)
    poses = {}
    for dev in ("kinect_0", "kinect_1"):
        rvec = rng.normal(0, 0.5, 3)
        tvec = np.array([rng.normal(0, 0.2), rng.normal(0, 0.2), 1.0 + rng.normal(0, 0.1)])
        poses[dev] = (rvec, tvec)
    legacy = {
        "version": 2,
        "extrinsics": [
            {"device_id": d, "rvec": rv.tolist(), "tvec": tv.tolist()}
            for d, (rv, tv) in poses.items()
        ],
        "sets": {
            d: [{"c_ids": [0, 1, 2, 3], "corners": [[1, 1], [2, 1], [2, 2], [1, 2]],
                 "resolution": [1280, 720]}]
            for d in poses
        },
        "intrinsics": {}, "board": {"type": "aruco"},
    }
    presets.preset_path("old").write_text(json.dumps(legacy))

    artifacts.ensure_migrated("old")
    extr = artifacts.read_extrinsics("old")
    assert extr["reference_device"] == "kinect_0"
    assert extr["solve"]["method"] == "legacy-migrated"
    # reference camera is identity in the rig frame
    np.testing.assert_allclose(
        artifacts.device_transforms("old")["kinect_0"], np.eye(4), atol=1e-9
    )
    # composing T_world_display @ T_ref_cam reproduces the legacy board->world pose
    Twd = artifacts.world_display_matrix("old")
    for dev, (rv, tv) in poses.items():
        legacy_world = CalibrationExtrinsics(rvec=rv, tvec=tv).transform_matrix
        got = Twd @ artifacts.device_transforms("old")[dev]
        np.testing.assert_allclose(got, legacy_world, atol=1e-9)
    # sets carried across
    assert len(extr["sets"]) == 1
    assert set(extr["sets"][0]["observations"]) == {"kinect_0", "kinect_1"}

    # idempotent
    artifacts.ensure_migrated("old")
    assert artifacts.read_extrinsics("old")["reference_device"] == "kinect_0"
