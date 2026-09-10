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
        ep, PerceiveOpts(
            profile=None,
            track_lm=[0, 5, 9, 13, 17, 4, 8],
            build_cloud=False,
        ),
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
    from viki.perception.profiles import STABLE_FUSED_HAND_V4

    o = PerceiveOpts.from_dict({})
    assert o.model and isinstance(o.track_lm, list) and len(o.track_lm) >= 6
    assert o.profile == STABLE_FUSED_HAND_V4
    assert PerceiveOpts.from_dict({"profile": None}).profile is None
    # legacy 'backend' key still maps to model
    assert PerceiveOpts.from_dict({"backend": "rtmpose-m-hand5"}).model == "rtmpose-m-hand5"
    # The angular gate override belongs to the custom path only, and an empty
    # or zero value means "leave it to config" rather than "gate at zero".
    assert o.reproj_inlier_mrad is None
    assert PerceiveOpts.from_dict({"reproj_inlier_mrad": 12.5}).reproj_inlier_mrad == 12.5
    for blank in (None, "", 0):
        assert PerceiveOpts.from_dict({"reproj_inlier_mrad": blank}).reproj_inlier_mrad is None
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
        assert params["fused_interpolation"] == "cubic"
        assert params["fused_extrapolate_edges"] is True
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


def test_stable_profile_routes_clean_to_fused_and_articulated_to_hand_fit(
    tmp_path, monkeypatch,
):
    from viki.episode import read_status
    from viki.perception.profiles import (
        CLEAN_LANDMARKS_V1,
        STABLE_FUSED_HAND_V1,
        get_profile,
    )
    from viki.prepare.baseline import _same_core_arrays

    monkeypatch.setattr(
        "viki.perception.extract.load_backend", lambda *a, **k: _FakeBackend(),
    )
    calls = []

    def fake_generate(ep, *, cfg, report=None):
        baseline = (
            ep.intermediates_dir / "baselines" / cfg.source_profile / "cln.npz"
        )
        assert baseline.is_file()
        calls.append(("generate", cfg.name, cfg.source_profile))
        return {
            "report": str(ep.intermediates_dir / "geometry" / cfg.name / "report.json"),
            "metrics": {"optimized": {"quality_gate": {
                "accepted": True,
                "structural_pass": True,
                "fidelity_pass": True,
                "temporal_pass": True,
            }}},
        }

    def fake_install(ep, *, cfg, variant):
        baseline = (
            ep.intermediates_dir / "baselines" / cfg.source_profile / "cln.npz"
        )
        assert _same_core_arrays(baseline, ep.cln_npz)
        with np.load(ep.cln_npz, allow_pickle=False) as current:
            payload = {key: current[key] for key in current.files}
        T = len(payload["timestamps"])
        payload.update({
            "hand_fit_positions": np.asarray(payload["positions"]).copy(),
            "hand_fit_rotations": np.asarray(payload["rotations"]).copy(),
            "hand_fit_joint_angles": np.zeros((T, 27), np.float32),
            "hand_fit_capsules": np.zeros((T, 16, 2, 3), np.float32),
            "hand_fit_overlay_variant": np.asarray(f"{cfg.name}:optimized"),
        })
        np.savez_compressed(ep.cln_npz, **payload)
        calls.append(("install", cfg.name, cfg.source_profile, variant))
        return {"clean_core_unchanged": _same_core_arrays(baseline, ep.cln_npz)}

    monkeypatch.setattr(
        "viki.perception.articulated.generate_articulated_variants", fake_generate,
    )
    monkeypatch.setattr(
        "viki.perception.articulated.install_articulated_overlay", fake_install,
    )
    ep = _synthetic_episode(tmp_path, frames=12)

    perceive_episode(ep, PerceiveOpts(profile=STABLE_FUSED_HAND_V1))

    assert calls == [
        ("generate", "articulated-landmarks-v1", STABLE_FUSED_HAND_V1),
        ("install", "articulated-landmarks-v1", STABLE_FUSED_HAND_V1, "optimized"),
    ]
    baseline = (
        ep.intermediates_dir / "baselines" / STABLE_FUSED_HAND_V1 / "cln.npz"
    )
    assert _same_core_arrays(baseline, ep.cln_npz)
    with np.load(baseline, allow_pickle=False) as fused, np.load(
        ep.cln_npz, allow_pickle=False,
    ) as active:
        np.testing.assert_array_equal(active["smoothed_points"], fused["smoothed_points"])
        assert active["hand_fit_capsules"].shape == (12, 16, 2, 3)

    stage = read_status(ep)["stages"]["prepare"]
    assert stage["profile"] == STABLE_FUSED_HAND_V1
    assert stage["hand_fit"] is True
    assert stage["routing"] == {
        "fused": "smoothed_points",
        "hand_fit": "hand_fit_capsules",
    }
    assert stage["baseline"]["matches_active_core"] is True
    # The articulated overlay is an output this profile declares, and it is
    # produced after the landmark trajectory has been protected. The baseline
    # takes it by additive refresh, so it now matches the active artifact
    # exactly - previously it was frozen without it, and anything measured off
    # the baseline silently used landmark poses while production used the fit.
    assert stage["baseline"]["matches_active_bytes"] is True
    assert stage["articulated"]["quality_gate"]["accepted"] is True

    baseline_path = (
        ep.intermediates_dir / "baselines" / STABLE_FUSED_HAND_V1 / "cln.npz"
    )
    with np.load(baseline_path, allow_pickle=False) as protected:
        assert "hand_fit_positions" in protected.files
        assert "hand_fit_rotations" in protected.files
    refreshed = json.loads(
        (baseline_path.parent / "manifest.json").read_text()
    )["additive_refresh"]
    # The refresh is auditable: what it added, and what it superseded.
    assert "hand_fit_positions" in refreshed["added_keys"]
    assert refreshed["previous_artifact_sha256"]

    # Adding the composed profile must not mutate the historical clean-profile
    # manifest used by already protected baselines.
    assert "articulated_hand_fit" not in get_profile(CLEAN_LANDMARKS_V1).manifest()
    assert get_profile(STABLE_FUSED_HAND_V1).manifest()["articulated_hand_fit"] == (
        "articulated-landmarks-v1"
    )


def test_stable_profiles_version_gripper_and_confidence_without_mutating_v1():
    from viki.perception.profiles import (
        DEFAULT_PERCEPTION_PROFILE,
        FUSED_HAND_NO_EXTRAP_V1,
        STABLE_FUSED_HAND_V1,
        STABLE_FUSED_HAND_V2,
        STABLE_FUSED_HAND_V3,
        STABLE_FUSED_HAND_V4,
        get_profile,
    )

    assert DEFAULT_PERCEPTION_PROFILE == STABLE_FUSED_HAND_V4
    assert get_profile(STABLE_FUSED_HAND_V1).gripper == "binary"
    assert get_profile(STABLE_FUSED_HAND_V1).fused_interpolation == "cubic"
    assert get_profile(STABLE_FUSED_HAND_V1).fused_extrapolate_edges is True
    assert get_profile(STABLE_FUSED_HAND_V2).gripper == "linear"
    assert get_profile(STABLE_FUSED_HAND_V2).fused_interpolation == "linear"
    assert get_profile(STABLE_FUSED_HAND_V2).fused_extrapolate_edges is True
    assert get_profile(FUSED_HAND_NO_EXTRAP_V1).fused_interpolation == "linear"
    assert get_profile(FUSED_HAND_NO_EXTRAP_V1).fused_extrapolate_edges is False
    assert get_profile(FUSED_HAND_NO_EXTRAP_V1).gripper == "linear"
    for field_name in get_profile(STABLE_FUSED_HAND_V2).__dataclass_fields__:
        if field_name not in {"name", "description", "fused_extrapolate_edges"}:
            assert getattr(get_profile(FUSED_HAND_NO_EXTRAP_V1), field_name) == getattr(
                get_profile(STABLE_FUSED_HAND_V2), field_name
            )
    assert get_profile(STABLE_FUSED_HAND_V2).confidence_calibration == "episode_max"
    assert get_profile(STABLE_FUSED_HAND_V3).gripper == "linear"

    # V4 is V2 with every scale-dependent knob restated in a unit that does not
    # change meaning when the capture configuration does. The default must not
    # silently acquire a pixel- or frame-counted parameter again.
    v2, v4 = get_profile(STABLE_FUSED_HAND_V2), get_profile(STABLE_FUSED_HAND_V4)
    assert v4.triangulation["reproj_inlier_mrad"] == 25.0
    assert v4.depth_radius_mrad is not None and v4.sg_window_ms is not None
    # 0 means "no limit", which is already scale-free.
    assert v4.interp_max_gap == 0
    for field_name in v2.__dataclass_fields__:
        if field_name not in {
            "name", "description", "triangulation",
            "depth_radius_mrad", "sg_window_ms", "interp_max_gap_ms",
        }:
            assert getattr(v4, field_name) == getattr(v2, field_name), field_name
    # V1 and V2 keep their frozen counted forms: they are the record of what was
    # actually run, and every protected baseline validates against them.
    assert v2.depth_radius_mrad is None and v2.sg_window_ms is None
    assert v2.triangulation.get("reproj_inlier_mrad") is None
    assert get_profile(STABLE_FUSED_HAND_V3).fused_interpolation == "linear"
    assert get_profile(STABLE_FUSED_HAND_V3).fused_extrapolate_edges is True
    assert get_profile(STABLE_FUSED_HAND_V3).confidence_calibration == "absolute"
    assert get_profile(STABLE_FUSED_HAND_V2).detector_model == "mediapipe"
    assert get_profile(STABLE_FUSED_HAND_V2).min_confidence == 0.5
    assert get_profile(STABLE_FUSED_HAND_V3).detector_model == "mediapipe"
    assert get_profile(STABLE_FUSED_HAND_V3).min_confidence == 0.5
    assert get_profile(STABLE_FUSED_HAND_V3).tracking_confidence is None
    assert get_profile(STABLE_FUSED_HAND_V3).triangulation["min_score"] == 0.3
    assert get_profile(STABLE_FUSED_HAND_V2).articulated_hand_fit == (
        get_profile(STABLE_FUSED_HAND_V1).articulated_hand_fit
    )
    assert "confidence_calibration" not in get_profile(STABLE_FUSED_HAND_V2).manifest()
    assert "fused_interpolation" not in get_profile(STABLE_FUSED_HAND_V1).manifest()
    assert "fused_extrapolate_edges" not in get_profile(STABLE_FUSED_HAND_V1).manifest()
    assert get_profile(STABLE_FUSED_HAND_V2).manifest()["fused_interpolation"] == "linear"
    assert "fused_extrapolate_edges" not in get_profile(STABLE_FUSED_HAND_V2).manifest()
    assert get_profile(FUSED_HAND_NO_EXTRAP_V1).manifest()["fused_extrapolate_edges"] is False
    assert "tracking_confidence" not in get_profile(STABLE_FUSED_HAND_V2).manifest()
    assert get_profile(STABLE_FUSED_HAND_V3).manifest()["confidence_calibration"] == (
        "absolute"
    )
