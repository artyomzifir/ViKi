"""viki.perception.run.perceive_episode — the full perception stage in one call
on a synthetic 2-camera episode (fake hand backend, constant-depth planes)."""

import json

import numpy as np
import pytest

from viki.contracts import HAND_LM_COUNT, HandDetection, LM
from viki.episode import new_episode, stage_done
from viki.perception.run import PerceiveOpts, perceive_episode


class _FakeBackend:
    """Returns a small plausible hand at the image centre for every frame."""

    def detect(self, frame, hand):
        h, w = frame.rgb.shape[:2]
        pts = {
            LM(i): np.array([w / 2 + (i % 5) * 3, h / 2 + (i // 5) * 3], np.float32)
            for i in range(HAND_LM_COUNT)
        }
        return HandDetection(
            points=pts, lm_z_rel=np.zeros(HAND_LM_COUNT, np.float32),
            confidence=0.9, device_id=frame.device_id, timestamp_us=frame.timestamp_us,
        )

    def close(self):
        pass


def _synthetic_episode(tmp_path, frames=8):
    import cv2

    ep = new_episode(tmp_path, {"task": "syn", "calibration_preset": ""})
    raw = ep.raw_dir
    K = {"fx": 200, "fy": 200, "cx": 32, "cy": 24, "width": 64, "height": 48,
         "dist_coeffs": [0, 0, 0, 0, 0]}
    intr, extr, ts = {}, {}, []
    for c, dev in enumerate(("kinect_0", "kinect_1")):
        intr[dev] = {"color": K, "depth": K, "source": "sdk"}
        extr[dev] = {"rvec": [0.01 * c, 0.0, 0.0], "tvec": [0.0, 0.0, 0.5 + 0.02 * c]}
        w = cv2.VideoWriter(str(raw / f"{dev}.mp4"), cv2.VideoWriter_fourcc(*"mp4v"),
                            30, (64, 48))
        for _ in range(frames):
            w.write(np.full((48, 64, 3), 90, np.uint8))
        w.release()
        d = raw / f"{dev}_depth"
        d.mkdir()
        for i in range(frames):
            np.save(d / f"{i:06d}.npy", np.full((48, 64), 1500, np.uint16))
    (raw / "intrinsics.json").write_text(json.dumps(intr))
    (raw / "extrinsics.json").write_text(json.dumps(extr))
    for i in range(frames):
        ts.append({"sync_us": 1_000_000 + i * 33_000,
                   "offsets_us": {"kinect_0": -1000, "kinect_1": -2500}})
    (raw / "timestamps.json").write_text(json.dumps(ts))
    return ep


def test_perceive_end_to_end(tmp_path, monkeypatch):
    monkeypatch.setattr("viki.perception.extract.load_backend",
                        lambda *a, **k: _FakeBackend())
    ep = _synthetic_episode(tmp_path)

    seen = []
    perceive_episode(
        ep, PerceiveOpts(track_lm=[0, 5, 9, 13, 17, 4, 8], build_cloud=False),
        report=lambda **kw: seen.append(kw),
    )

    assert stage_done(ep, "extract") and stage_done(ep, "prepare")
    assert any(s.get("stage") == "done" for s in seen)
    assert any((s.get("frame") or 0) > 0 for s in seen)

    with np.load(ep.rec_npz) as d:
        assert len(d["device_ids"]) > 0
        assert set(str(x) for x in d["device_ids"]) == {"kinect_0", "kinect_1"}
        # real host-monotonic timestamps (~1e6 µs), not the frame index (0..7)
        assert int(d["timestamps"][0]) > 900_000
        pts = np.asarray(d["points"])
        tracked = {0, 5, 9, 13, 17, 4, 8}
        finite = np.isfinite(pts).all(axis=2).any(axis=0)  # per-landmark, any row
        for lm in range(HAND_LM_COUNT):
            if lm not in tracked:
                assert not finite[lm], f"landmark {lm} should be dropped"

    with np.load(ep.cln_npz) as d:
        for k in ("positions", "rotations", "valid", "omega", "gripper",
                  "smoothed_points", "timestamps"):
            assert k in d
        assert len(d["positions"]) > 0


def test_perceive_opts_from_dict_defaults():
    o = PerceiveOpts.from_dict({})
    assert o.model and isinstance(o.track_lm, list) and len(o.track_lm) >= 6
    # legacy 'backend' key still maps to model
    assert PerceiveOpts.from_dict({"backend": "rtmpose-m-hand5"}).model == "rtmpose-m-hand5"
    o2 = PerceiveOpts.from_dict({"hand": "left", "sg_window": 9, "track_lm": [0, 4, 8]})
    assert o2.hand == "left" and o2.sg_window == 9 and o2.track_lm == [0, 4, 8]
