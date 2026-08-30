"""Calibration presets v2 (bundled capture sets) + offline extrinsics re-solve."""

import json

import numpy as np
import pytest

from viki.calibration import presets


@pytest.fixture(autouse=True)
def _tmp_paths(tmp_path, monkeypatch):
    monkeypatch.setattr(presets, "PRESETS_DIR", tmp_path / "calibrations")
    monkeypatch.setattr(presets, "EXTRINSICS_FILENAME", str(tmp_path / "extrinsics.json"))
    monkeypatch.setattr(presets, "USER_CONFIG_PATH", str(tmp_path / "user.json"))
    (tmp_path / "user.json").write_text("{}")


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
