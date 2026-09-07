"""
Unit tests for :mod:`viki.benchmark.hocap` on synthetic fixtures only — no
network, no HO-Cap download, no MANO. Covers the four highest-risk conversions
(extrinsics inversion, joint permutation, depth scale, timestamp generation)
plus an end-to-end import of a tiny synthetic HO-Cap tree and the acceptance
criteria that do not need a real dataset (1, 2, 3, 4).
"""

from __future__ import annotations

import json
import math

import cv2
import numpy as np
import pytest
import yaml

from viki.benchmark.hocap import (
    HOCAP_JOINT_NAMES,
    VIKI_JOINT_NAMES,
    VIKI_LM_FROM_HOCAP,
    _copy_depth,
    assert_extrinsics_roundtrip,
    import_sequence,
    make_timestamps,
    world_camera_pose_to_rvec_tvec,
)
from viki.contracts import HAND_LM_COUNT, LM, CalibrationExtrinsics

# ───────────────────────────── fixture builders ────────────────────────────

_SERIALS = ["111111111111", "222222222222", "333333333333"]
_W, _H = 80, 60
_FX = _FY = 90.0
_CX, _CY = _W / 2.0, _H / 2.0
_N_FRAMES = 4


def _rigid(rx: float, ry: float, rz: float, t: tuple[float, float, float]) -> np.ndarray:
    R = (
        cv2.Rodrigues(np.array([rx, 0, 0], float))[0]
        @ cv2.Rodrigues(np.array([0, ry, 0], float))[0]
        @ cv2.Rodrigues(np.array([0, 0, rz], float))[0]
    )
    T = np.eye(4)
    T[:3, :3] = R
    T[:3, 3] = t
    return T


def _look_at_cam(eye, target) -> np.ndarray:
    """camera -> world (OpenCV axes: x right, y down, z forward) looking from
    ``eye`` at ``target`` in a z-up world."""
    eye = np.asarray(eye, float)
    z = np.asarray(target, float) - eye
    z /= np.linalg.norm(z)
    x = np.cross(np.array([0.0, 0.0, 1.0]), z)
    x /= np.linalg.norm(x)
    y = np.cross(z, x)
    T = np.eye(4)
    T[:3, 0], T[:3, 1], T[:3, 2], T[:3, 3] = x, y, z, eye
    return T


def _mat12(T: np.ndarray) -> list[float]:
    return [float(x) for x in np.asarray(T)[:3, :4].reshape(-1)]


def _K() -> np.ndarray:
    return np.array([[_FX, 0, _CX], [0, _FY, _CY], [0, 0, 1]], float)


def _synthetic_hand_world(frame: int) -> np.ndarray:
    """21 world-frame joints (m): a splayed right hand ~0.4 m in front of the
    world origin, drifting a little per frame so the wrist trajectory is real."""
    base = np.array([0.02 * frame - 0.03, 0.01 * frame, 0.42], float)
    joints = np.zeros((21, 3), float)
    joints[0] = base  # wrist
    # thumb (idx 1..4) and four fingers (5..20), 4 joints each, fanned in x
    fingers = {
        1: (-0.9, 1),  # thumb splays -x
        5: (-0.35, 1),
        9: (-0.12, 1),
        13: (0.12, 1),
        17: (0.35, 1),
    }
    for start, (xdir, ydir) in fingers.items():
        for k in range(4):
            joints[start + k] = base + np.array(
                [0.025 * xdir * (k + 1), 0.03 * ydir * (k + 1), 0.004 * k], float
            )
    return joints


def _world_from_cam(extr_yaml: dict, serial: str) -> np.ndarray:
    def m12(v):
        T = np.eye(4)
        T[:3, :4] = np.asarray(v, float).reshape(3, 4)
        return T

    ext = extr_yaml["extrinsics"]
    tag_1 = m12(ext["tag_1"])
    return np.linalg.inv(tag_1) @ m12(ext[serial])


_HAND_SLOT = {"right": 0, "left": 1}


def build_hocap_tree(root, *, hands=("right",), mano_sides=None, n_frames=_N_FRAMES):
    """Write a minimal HO-Cap-shaped tree under ``root`` and return
    ``(subject, sequence, extr_yaml_dict, world_from_cam_by_serial)``.

    ``hands`` = which of the fixed [right, left] label slots carry joints
    (real HO-Cap always stores a length-2 hand axis with ``-1`` in the absent
    slot). ``mano_sides`` defaults to ``hands`` but can be set independently to
    exercise a mislabelled ``meta.yaml``."""
    if mano_sides is None:
        mano_sides = hands
    subject, sequence = "subject_1", "20231025_000000"
    calib = root / "calibration"
    (calib / "intrinsics").mkdir(parents=True)
    (calib / "extrinsics").mkdir(parents=True)

    # camera -> world poses (look-at, hand well in front of every camera),
    # then camera -> rs_master = tag_1 @ world_from_cam
    _tgt = (0.0, 0.0, 0.42)
    world_from_cam = {
        _SERIALS[0]: _look_at_cam((-0.35, -0.15, 0.05), _tgt),
        _SERIALS[1]: _look_at_cam((0.35, -0.15, 0.05), _tgt),
        _SERIALS[2]: _look_at_cam((0.0, 0.40, 0.10), _tgt),
    }
    tag_1 = _rigid(0.1, -0.2, 0.05, (0.3, -0.1, 0.2))
    tag_0 = _rigid(0.0, 0.1, 0.0, (0.1, 0.1, 0.0))
    ext_entries = {"tag_0": _mat12(tag_0), "tag_1": _mat12(tag_1)}
    for s, T_wc in world_from_cam.items():
        ext_entries[s] = _mat12(tag_1 @ T_wc)
    extr_yaml = {"rs_master": _SERIALS[0], "extrinsics": ext_entries}
    (calib / "extrinsics" / "extrinsics_test.yaml").write_text(yaml.safe_dump(extr_yaml))

    for s in _SERIALS:
        intr = {
            "color": {"fx": _FX, "fy": _FY, "ppx": _CX, "ppy": _CY,
                      "width": _W, "height": _H},
            "depth": {"fx": _FX, "fy": _FY, "ppx": _CX, "ppy": _CY,
                      "width": _W, "height": _H},
        }
        (calib / "intrinsics" / f"{s}.yaml").write_text(yaml.safe_dump(intr))

    seq_dir = root / subject / sequence
    for s in _SERIALS:
        (seq_dir / s).mkdir(parents=True)
    meta = {
        "num_frames": n_frames,
        "mano_sides": list(mano_sides),
        "realsense": {"serials": _SERIALS, "width": _W, "height": _H},
        "extrinsics": "extrinsics_test.yaml",
        "subject_id": subject,
        "task_id": 1,
        "object_ids": ["G01_1"],
    }
    (seq_dir / "meta.yaml").write_text(yaml.safe_dump(meta))
    np.save(seq_dir / "poses_m.npy", np.zeros((2, n_frames, 51), np.float32))

    K = _K()
    for fi in range(n_frames):
        jw = _synthetic_hand_world(fi)
        for s in _SERIALS:
            T_cw = np.linalg.inv(world_from_cam[s])
            jc = (T_cw[:3, :3] @ jw.T + T_cw[:3, 3:4]).T  # camera-frame (21,3)
            uv = (K @ jc.T).T
            uv2 = uv[:, :2] / uv[:, 2:3]

            # colour frame: mid-grey with a bright dot at each joint pixel
            img = np.full((_H, _W, 3), 60, np.uint8)
            for (u, v) in uv2:
                if 0 <= u < _W and 0 <= v < _H:
                    cv2.circle(img, (int(round(u)), int(round(v))), 1,
                               (255, 255, 255), -1)
            cv2.imwrite(str(seq_dir / s / f"color_{fi:06d}.jpg"), img)

            # depth PNG (uint16 mm): filled disk of the joint's camera Z at
            # each joint pixel, zero elsewhere.
            depth = np.zeros((_H, _W), np.uint16)
            for (u, v), z in zip(uv2, jc[:, 2]):
                if 0 <= u < _W and 0 <= v < _H and z > 0:
                    cv2.circle(depth, (int(round(u)), int(round(v))), 3,
                               int(round(z * 1000.0)), -1)
            cv2.imwrite(str(seq_dir / s / f"depth_{fi:06d}.png"), depth)

            # fixed 2-slot [right, left] hand axis, -1 in the absent slot;
            # hand_joints_2d all -1 like the real dataset (importer must not need it)
            hj3 = np.full((2, 21, 3), -1.0, np.float32)
            for side in hands:
                hj3[_HAND_SLOT[side]] = jc.astype(np.float32)
            hj2 = np.full((2, 21, 2), -1, np.int64)

            np.savez(
                seq_dir / s / f"label_{fi:06d}.npz",
                cam_K=K.astype(np.float32),
                hand_joints_3d=hj3,                            # (2, 21, 3)
                hand_joints_2d=hj2,                            # (2, 21, 2)
                seg_mask=np.zeros((_H, _W), np.uint8),
                obj_class_inds=np.array([1]),
                obj_class_names=np.array(["G01_1"]),
            )
    return subject, sequence, extr_yaml, world_from_cam


# ─────────────────────────────── unit tests ────────────────────────────────


def test_joint_map_is_identity_permutation():
    assert len(VIKI_LM_FROM_HOCAP) == HAND_LM_COUNT == 21
    assert sorted(VIKI_LM_FROM_HOCAP) == list(range(21))
    assert VIKI_LM_FROM_HOCAP == list(range(21))  # HO-Cap == MediaPipe layout
    assert VIKI_JOINT_NAMES == [lm.name for lm in LM]
    # positional structure: wrist at 0, thumb 1..4, then mcp/pip/dip/tip x4
    assert HOCAP_JOINT_NAMES[0] == "wrist" and VIKI_JOINT_NAMES[0] == "WRIST"
    for base, tag in ((1, "thumb"), (5, "index"), (9, "middle"),
                      (13, "ring"), (17, "little")):
        assert all(HOCAP_JOINT_NAMES[base + k].startswith(tag) for k in range(4))
    for base, tag in ((1, "THUMB"), (5, "INDEX"), (9, "MIDDLE"),
                      (13, "RING"), (17, "PINKY")):
        assert all(VIKI_JOINT_NAMES[base + k].startswith(tag) for k in range(4))


@pytest.mark.parametrize("seed", [0, 1, 2, 7])
def test_extrinsics_inversion_roundtrip(seed):
    rng = np.random.default_rng(seed)
    T_wc = _rigid(*(rng.uniform(-math.pi, math.pi, 3)),
                  tuple(rng.uniform(-1.0, 1.0, 3)))
    rvec, tvec = world_camera_pose_to_rvec_tvec(T_wc)
    residual = assert_extrinsics_roundtrip(rvec, tvec, T_wc, tol=1e-9)
    assert residual < 1e-9
    # the real reader must reproduce T_world_camera exactly
    T_back = CalibrationExtrinsics(rvec=rvec, tvec=tvec).transform_matrix
    assert np.allclose(T_back, T_wc, atol=1e-9)
    # and it must map the camera origin to the published camera centre
    centre = (T_back @ np.array([0, 0, 0, 1.0]))[:3]
    assert np.allclose(centre, T_wc[:3, 3], atol=1e-9)


def test_make_timestamps_is_synthetic_and_evenly_spaced():
    devs = ["a", "b"]
    ts = make_timestamps(5, 30.0, devs)
    step = round(1e6 / 30.0)
    assert len(ts) == 5
    for i, row in enumerate(ts):
        assert row["sync_us"] == i * step
        assert row["offsets_us"] == {"a": 0, "b": 0}
    # 10 fps subset spacing
    ts10 = make_timestamps(3, 10.0, devs)
    assert [r["sync_us"] for r in ts10] == [0, 100_000, 200_000]


def test_depth_scale_conversion_preserves_millimetres(tmp_path):
    src = tmp_path / "src" / _SERIALS[0]
    src.mkdir(parents=True)
    arr = np.zeros((_H, _W), np.uint16)
    arr[10, 12] = 1234           # 1234 mm
    arr[20, 30] = 65000
    cv2.imwrite(str(src / "depth_000000.png"), arr)

    out = tmp_path / "out_depth"
    written, dtype_seen = _copy_depth(tmp_path / "src", _SERIALS[0], out, 1)
    assert written == 1 and dtype_seen == "uint16"
    saved = np.load(out / "000000.npy")
    assert saved.dtype == np.uint16
    assert np.array_equal(saved, arr)
    # this is exactly what viki.perception.extract does with it
    depth_m = saved.astype(np.float32) / 1000.0
    assert depth_m[10, 12] == pytest.approx(1.234)
    assert depth_m[20, 30] == pytest.approx(65.0)


# ───────────────────────────── integration ─────────────────────────────────


@pytest.fixture()
def imported(tmp_path):
    root = tmp_path / "hocap"
    subject, sequence, extr_yaml, world_from_cam = build_hocap_tree(root)
    res = import_sequence(
        root, subject, sequence, out_root=tmp_path / "datasets" / "hocap",
    )
    return res, extr_yaml, world_from_cam, root


def test_import_writes_capture_time_layout(imported):
    res, extr_yaml, world_from_cam, _root = imported
    ep = res.episode_dir
    a, b = res.cameras
    assert (ep / "raw" / f"{a}.mp4").is_file()
    assert (ep / "raw" / f"{b}.mp4").is_file()
    assert (ep / "raw" / f"{a}_depth" / "000000.npy").is_file()
    assert np.load(ep / "raw" / f"{a}_depth" / "000000.npy").dtype == np.uint16

    intr = json.loads((ep / "raw" / "intrinsics.json").read_text())
    assert set(intr) == {a, b}
    for s in (a, b):
        assert "depth" in intr[s] and "fx" in intr[s]["depth"]
        assert intr[s]["depth"]["fx"] == pytest.approx(_FX)

    ts = json.loads((ep / "raw" / "timestamps.json").read_text())
    assert len(ts) == res.n_frames
    assert all(row["offsets_us"][a] == 0 for row in ts)

    meta = json.loads((ep / "meta.json").read_text())
    assert meta["source"] == "hocap"
    assert meta["timestamps"] == "synthetic"
    assert meta["hocap_license"] == "CC BY 4.0"
    assert "calibration_preset" not in meta
    assert meta["hocap_pair_selection"]["selected_pair"] == [a, b]
    assert len(meta["hocap_pair_selection"]["pair_table"]) == 3  # 3 cams -> 3 pairs
    # no sync_stats.json (synthetic timestamps must not look measured)
    assert not (ep / "raw" / "sync_stats.json").exists()
    assert not (ep / "status.json").read_text().strip() == ""


def test_written_extrinsics_roundtrip_and_extract_can_read_them(imported):
    res, extr_yaml, world_from_cam, _root = imported
    ep = res.episode_dir
    extr_json = json.loads((ep / "raw" / "extrinsics.json").read_text())

    from viki.perception.extract import _depth_K, _extrinsics

    intr = json.loads((ep / "raw" / "intrinsics.json").read_text())
    for s in res.cameras:
        T_expected = _world_from_cam(extr_yaml, s)
        # via the contract type
        ce = CalibrationExtrinsics(
            rvec=np.asarray(extr_json[s]["rvec"]),
            tvec=np.asarray(extr_json[s]["tvec"]),
        )
        assert np.allclose(ce.transform_matrix, T_expected, atol=1e-9)
        # via the real extract helper
        ce2 = _extrinsics(extr_json, s)
        assert np.allclose(ce2.transform_matrix, T_expected, atol=1e-9)
        K = _depth_K(intr[s])
        assert K is not None and K[0, 0] == pytest.approx(_FX)


def test_ground_truth_npz_schema_and_world_transform(imported):
    res, extr_yaml, world_from_cam, root = imported
    ep = res.episode_dir
    with np.load(ep / "gt" / "joints3d_world.npz", allow_pickle=True) as z:
        jw = z["joints_world"]
        valid = z["valid"]
        assert jw.shape == (res.n_frames, 21, 3)
        assert jw.dtype == np.float32
        assert valid.shape == (res.n_frames, 21) and valid.dtype == bool
        assert z["frame_index"].tolist() == list(range(res.n_frames))
        assert str(z["handedness"]) == "right"
        assert str(z["world_frame"]) == "hocap_tag_1"
        assert list(z["joint_order"]) == VIKI_JOINT_NAMES
        assert list(z["source_joint_order"]) == HOCAP_JOINT_NAMES

    # frame 0 wrist: transform synthetic camera-frame joint back to world
    # independently and compare
    a = res.cameras[0]
    with np.load(root / "subject_1" / "20231025_000000" / a / "label_000000.npz") as lz:
        jc = np.asarray(lz["hand_joints_3d"])[0]
    T_wc = _world_from_cam(extr_yaml, a)
    manual = (T_wc[:3, :3] @ jc.T + T_wc[:3, 3:4]).T
    assert np.allclose(jw[0], manual, atol=2e-4)


def test_acceptance_criteria_pass_on_synthetic(imported):
    res, *_ = imported
    acc = res.acceptance

    assert acc["extrinsics_roundtrip"]["passed"] is True
    assert acc["extrinsics_roundtrip"]["max_residual_m"] < 1e-9

    # criterion 4: triangulate GT reprojections back — median well under 1 mm
    tri = acc["triangulation_identity"]
    assert tri["n_joints"] > 0
    assert tri["median_mm"] < 1.0
    assert tri["passed"] is True

    # criterion 2: depth sampled at joint pixels matches camera-frame Z
    for cam, d in acc["depth_consistency"].items():
        assert d["all_joints"]["n"] > 0
        assert d["all_joints"]["median_mm"] < 15.0
        assert d["passed"] is True

    # criterion 3: overlays were rendered for both cameras
    files = acc["reprojection_overlays"]["files"]
    assert len(files) >= 2
    for f in files:
        assert f.endswith(".png")


def test_gt_frame_convention_detected_as_camera(imported):
    res, *_ = imported
    fc = res.acceptance["gt_frame_convention"]
    assert fc["resolved"] == "camera"
    ev = fc["evidence"]
    # cross-camera agreement: the camera-frame hypothesis collapses to a point,
    # the world-frame hypothesis (raw per-camera coords) does not
    assert ev["camera_hypothesis_spread_mm"] < 1.0
    assert ev["world_hypothesis_spread_mm"] > ev["camera_hypothesis_spread_mm"]
    meta = json.loads((res.episode_dir / "meta.json").read_text())
    assert meta["gt_frame_convention"] == "camera"


@pytest.mark.parametrize("hands", [("right", "left"), ("left",)])
def test_non_right_hand_sequences_skipped_not_imported(tmp_path, hands):
    root = tmp_path / "hocap"
    subject, sequence, *_ = build_hocap_tree(root, hands=hands)
    res = import_sequence(
        root, subject, sequence, out_root=tmp_path / "datasets" / "hocap",
    )
    assert res.skipped and "right-hand single-hand" in res.skipped[0]["reason"]
    assert res.skipped[0]["hands_in_labels"] == sorted(hands)
    assert not res.acceptance
    assert not (tmp_path / "datasets" / "hocap").exists() or not any(
        (tmp_path / "datasets" / "hocap").iterdir()
    )


def test_labels_slots_override_mislabelled_meta(tmp_path):
    # meta says bimanual, but the labels only carry the right hand -> import it
    root = tmp_path / "hocap"
    subject, sequence, *_ = build_hocap_tree(
        root, hands=("right",), mano_sides=("right", "left")
    )
    res = import_sequence(root, subject, sequence, out_root=tmp_path / "d")
    assert not res.skipped
    assert res.acceptance["extrinsics_roundtrip"]["passed"] is True


def test_reimport_needs_force_and_respects_separate_out_dirs(tmp_path):
    root = tmp_path / "hocap"
    subject, sequence, *_ = build_hocap_tree(root)
    out1 = tmp_path / "datasets" / "hocap"
    r1 = import_sequence(root, subject, sequence, out_root=out1)
    with pytest.raises(FileExistsError):
        import_sequence(root, subject, sequence, out_root=out1)
    r1b = import_sequence(root, subject, sequence, out_root=out1, overwrite=True)
    assert r1b.episode_dir == r1.episode_dir

    out2 = tmp_path / "datasets" / "hocap_alt"
    r2 = import_sequence(root, subject, sequence, out_root=out2)
    assert r2.episode_dir != r1.episode_dir
    assert r1.episode_dir.is_dir()  # untouched


def test_manual_camera_override(tmp_path):
    root = tmp_path / "hocap"
    subject, sequence, *_ = build_hocap_tree(root)
    pair = (_SERIALS[2], _SERIALS[0])
    res = import_sequence(
        root, subject, sequence, out_root=tmp_path / "d", cameras=pair,
    )
    assert set(res.cameras) == set(pair)
    assert res.selection["override"] is True
    assert res.episode_dir.name.endswith(f"{pair[0]}-{pair[1]}")
