"""Calibration presets v2 (bundled capture sets) + offline extrinsics re-solve."""

import json

import numpy as np
import pytest

from viki.calibration import captures, presets


@pytest.fixture(autouse=True)
def _tmp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(presets, "PRESETS_DIR", tmp_path / "calibrations")
    monkeypatch.setattr(presets, "EXTRINSICS_FILENAME", str(tmp_path / "extrinsics.json"))
    monkeypatch.setattr(presets, "USER_CONFIG_PATH", str(tmp_path / "user.json"))
    monkeypatch.setattr(captures, "ROOT", tmp_path / "calibrations")
    (tmp_path / "user.json").write_text("{}")


def test_captures_save_list_delete_renumber(tmp_path, monkeypatch):
    monkeypatch.setattr(captures, "ROOT", tmp_path / "cap")
    img = np.zeros((8, 8, 3), np.uint8)
    for i in range(3):
        captures.save_set("_live", i, {"cam0": img, "cam1": img})
    assert [r["index"] for r in captures.list_sets("_live")] == [0, 1, 2]

    captures.delete_set("_live", 1)  # -> 0, (was 2 -> 1)
    rows = captures.list_sets("_live")
    assert [r["index"] for r in rows] == [0, 1]
    assert rows[1]["devices"] == ["cam0", "cam1"]

    captures.copy("_live", "rigX")
    assert [r["index"] for r in captures.list_sets("rigX")] == [0, 1]
    captures.wipe("rigX")
    assert captures.list_sets("rigX") == []


def _extr(dev):
    return {"device_id": dev, "rvec": [0.1, 0.2, 0.3], "tvec": [0.0, 0.0, 0.6]}


def test_v1_list_still_reads_and_activates(tmp_path):
    p = presets.PRESETS_DIR
    p.mkdir(parents=True)
    (p / "old.json").write_text(json.dumps([_extr("cam0"), _extr("cam1")]))

    rows = presets.list_presets()
    assert rows[0]["name"] == "old" and rows[0]["cameras"] == ["cam0", "cam1"]

    presets.activate("old")
    assert json.loads(open(presets.EXTRINSICS_FILENAME).read())[0]["device_id"] == "cam0"
    assert presets.current_active() == "old"


def test_v2_save_read_activate_roundtrip():
    presets.save_as(
        "rigA",
        extrinsics=[_extr("cam0")],
        sets={"cam0": [{"corners": [[1, 2]], "c_ids": [0], "resolution": [640, 480]}]},
        intrinsics={"cam0": {"fx": 600, "fy": 600, "cx": 320, "cy": 240}},
        board={"type": "aruco", "board_size": [8, 10], "square_size": 0.05,
               "marker_size": 0.035, "aruco_dict": 4},
    )
    d = presets.read_detail("rigA")
    assert d["version"] == 2 and d["board"]["type"] == "aruco"
    assert len(d["sets"]["cam0"]) == 1

    presets.activate("rigA")
    written = json.loads(open(presets.EXTRINSICS_FILENAME).read())
    assert isinstance(written, list) and written[0]["device_id"] == "cam0"


def _save(name):
    presets.save_as(
        name, extrinsics=[_extr("cam0")],
        sets={"cam0": [{"corners": [[1, 2]], "c_ids": [0], "resolution": [640, 480]}]},
        intrinsics={"cam0": {"fx": 600, "fy": 600, "cx": 320, "cy": 240}},
        board={"type": "aruco", "board_size": [8, 10], "square_size": 0.05,
               "marker_size": 0.035, "aruco_dict": 4},
    )


def test_preset_rename_keeps_active_and_moves_dir():
    _save("rigA")
    presets.attach_background("rigA", {"cam0": np.zeros((4, 4), np.float32)})  # makes rigA/ dir
    presets.activate("rigA")

    presets.rename("rigA", "rigB")

    assert not presets.preset_path("rigA").exists()
    assert presets.preset_path("rigB").exists()
    assert (presets.PRESETS_DIR / "rigB").is_dir()
    assert presets.current_active() == "rigB"
    assert "cam0" in presets.read_detail("rigB")["background_devices"]


def test_preset_rename_conflict_and_missing():
    _save("rigA")
    _save("rigC")
    with pytest.raises(FileExistsError):
        presets.rename("rigA", "rigC")
    with pytest.raises(FileNotFoundError):
        presets.rename("nope", "rigX")


def test_preset_delete_clears_active_and_dir():
    _save("rigA")
    presets.attach_background("rigA", {"cam0": np.zeros((4, 4), np.float32)})
    presets.activate("rigA")

    presets.delete("rigA")

    assert not presets.preset_path("rigA").exists()
    assert not (presets.PRESETS_DIR / "rigA").exists()
    assert presets.current_active() == ""
    with pytest.raises(FileNotFoundError):
        presets.delete("rigA")


def test_attach_and_read_k4a_blob(monkeypatch):
    import base64

    presets.save_as(
        "rigK",
        extrinsics=[_extr("kinect_0"), _extr("kinect_1")],
        sets={}, intrinsics={},
        board={"type": "aruco", "board_size": [8, 10], "square_size": 0.05,
               "marker_size": 0.035, "aruco_dict": 4},
    )
    presets.attach_k4a(
        "rigK",
        {"kinect_0": b'{"raw":"cal0"}\x00', "kinect_1": b'{"raw":"cal1"}\x00'},
        depth_mode_int=2, color_res_int=1,
    )

    d = presets.read_detail("rigK")
    assert d["k4a_devices"] == ["kinect_0", "kinect_1"]
    assert presets.list_presets()[0]["k4a"] == ["kinect_0", "kinect_1"]
    stored = json.loads((presets.PRESETS_DIR / "rigK.json").read_text())
    assert base64.b64decode(stored["k4a_raw"]["kinect_0"]) == b'{"raw":"cal0"}\x00'
    assert stored["k4a_depth_mode_int"] == 2 and stored["k4a_color_res_int"] == 1

    seen = {}
    def _fake_from_blob(cls, blob, di, ci, tag=""):
        seen.update(blob=blob, di=di, ci=ci)
        return "CAL"
    monkeypatch.setattr(
        "viki.perception.k4a_offline.K4ACalibration.from_blob",
        classmethod(_fake_from_blob),
    )
    assert presets.k4a_calibration("rigK", "kinect_1") == "CAL"
    assert seen["blob"] == b'{"raw":"cal1"}\x00' and seen["di"] == 2 and seen["ci"] == 1
    assert presets.k4a_calibration("rigK", "kinect_9") is None  # unknown device


def test_attach_k4a_rejects_v1_preset(tmp_path):
    p = presets.PRESETS_DIR
    p.mkdir(parents=True, exist_ok=True)
    (p / "legacy.json").write_text(json.dumps([_extr("cam0")]))
    with pytest.raises(ValueError):
        presets.attach_k4a("legacy", {"cam0": b"x"}, 2, 1)


def test_delete_set_resolves_from_real_charuco():
    import cv2

    dictionary = cv2.aruco.getPredefinedDictionary(4)
    board = cv2.aruco.CharucoBoard((8, 10), 0.05, 0.035, dictionary)
    obj = np.asarray(board.getChessboardCorners(), np.float32)  # (N,3)
    K = np.array([[600.0, 0, 320], [0, 600.0, 240], [0, 0, 1]])

    def view(rvec, tvec):
        img, _ = cv2.projectPoints(obj, np.array(rvec, float), np.array(tvec, float), K, np.zeros(5))
        return {"corners": img.reshape(-1, 2).tolist(),
                "c_ids": list(range(len(obj))), "resolution": [640, 480]}

    good = view([0.1, 0.05, 0.02], [-0.2, -0.25, 0.8])
    bad = view([1.5, 1.0, 0.5], [0.5, 0.5, 0.4])  # a wrong capture to prune

    presets.save_as(
        "rigB",
        extrinsics=[_extr("cam0")],
        sets={"cam0": [good, bad]},
        intrinsics={"cam0": {"fx": 600, "fy": 600, "cx": 320, "cy": 240}},
        board={"type": "aruco", "board_size": [8, 10], "square_size": 0.05,
               "marker_size": 0.035, "aruco_dict": 4},
    )
    presets.activate("rigB")

    d = presets.delete_set("rigB", 1)  # drop the bad set, re-solve
    assert len(d["sets"]["cam0"]) == 1
    ex = d["extrinsics"]
    assert ex and ex[0]["device_id"] == "cam0"
    assert all(np.isfinite(ex[0]["rvec"])) and all(np.isfinite(ex[0]["tvec"]))
    # active extrinsics file was refreshed too
    assert json.loads(open(presets.EXTRINSICS_FILENAME).read())[0]["device_id"] == "cam0"
