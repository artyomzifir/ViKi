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


def test_clean_baseline_profile_locks_pipeline_and_protects_output(tmp_path, monkeypatch):
    from viki import config
    from viki.perception.profiles import CLEAN_LANDMARKS_V1
    from viki.prepare.baseline import file_sha256

    monkeypatch.setattr("viki.perception.extract.load_backend",
                        lambda *a, **k: _FakeBackend())
    # Deliberately hostile experiment config: the named profile must override
    # every one of these accuracy-affecting choices.
    monkeypatch.setattr(config, "PERCEPTION_FUSE_MODE", "xyz_mean", raising=False)
    monkeypatch.setattr(config, "PERCEPTION_HAND_FIT", True, raising=False)
    monkeypatch.setattr(config, "PERCEPTION_SAVE_OBSERVATIONS", False, raising=False)
    monkeypatch.setattr(config, "TRI_REPROJ_INLIER_PX", 99.0, raising=False)
    ep = _synthetic_episode(tmp_path, frames=12)

    perceive_episode(ep, PerceiveOpts(
        profile=CLEAN_LANDMARKS_V1,
        model="rtmpose-m-hand5",
        track_lm=[0],
        interp_max_gap=3,
        sg_window=9,
        sg_polyorder=1,
    ))

    obs_meta = json.loads((ep.raw_dir / "observations_meta.json").read_text())
    assert obs_meta["sampler"] == {
        "depth_radius_px": 15,
        "model": "mediapipe",
        "min_confidence": 0.5,
        "profile": CLEAN_LANDMARKS_V1,
        "flip": False,
    }
    tri_summary = json.loads((ep.raw_dir / "joints3d_summary.json").read_text())
    assert tri_summary["config"]["reproj_inlier_px"] == 4.0

    with np.load(ep.cln_npz, allow_pickle=False) as clean:
        assert clean["perception_profile"].item() == CLEAN_LANDMARKS_V1
        assert clean["active_variant"].item() == CLEAN_LANDMARKS_V1
        assert clean["perception_fuse_mode"].item() == "triangulate"
        assert clean["pose_source"].item() == "landmarks"
        assert "hand_fit_positions" not in clean.files
        params = json.loads(clean["checkpoint_params_json"].item())
        assert params["interp_max_gap"] == 0
        assert params["window_length"] == 7
        assert params["polyorder"] == 2

    protected = (
        ep.intermediates_dir / "baselines" / CLEAN_LANDMARKS_V1 / "cln.npz"
    )
    manifest = json.loads(protected.with_name("manifest.json").read_text())
    assert protected.read_bytes() == ep.cln_npz.read_bytes()
    assert manifest["artifact_sha256"] == file_sha256(protected)
    # Input provenance must be real hashes, not a dropped/None field.
    assert manifest["inputs_sha256"]["rec.npz"] == file_sha256(ep.rec_npz)
    assert manifest["inputs_sha256"]["raw/timestamps.json"] == file_sha256(
        ep.raw_dir / "timestamps.json"
    )

    # Adding provenance to the active file does not make its trajectory differ
    # from the protected baseline, while the byte-level distinction stays clear.
    from viki.prepare.baseline import protect_baseline
    from viki.perception.profiles import get_profile

    with np.load(ep.cln_npz, allow_pickle=False) as current:
        payload = {key: current[key] for key in current.files}
    payload["test_only_provenance"] = np.asarray("new metadata")
    np.savez_compressed(ep.cln_npz, **payload)
    match = protect_baseline(ep, get_profile(CLEAN_LANDMARKS_V1), ep.cln_npz)
    assert match["matches_active"] is True
    assert match["matches_active_core"] is True
    assert match["matches_active_bytes"] is False
