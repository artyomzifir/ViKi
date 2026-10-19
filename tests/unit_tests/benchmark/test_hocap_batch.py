"""Archive staging stays bounded and preserves HO-Cap labels for all views."""

from __future__ import annotations

import io
import json
import tarfile

import numpy as np

from viki.contracts import Episode
from viki.perception.observations import calib_id
from viki.perception.profiles import (CLEAN_LANDMARKS_V1, DEFAULT_PERCEPTION_PROFILE,
                                      HOCAP_MEDIAPIPE_TRI_V1, HOCAP_STABLE_FUSED_HAND_V4, get_profile)

from scripts.run_hocap_batch import reuse_camera_tracks, stage_sequence


def _archive(path, files):
    with tarfile.open(path, "w") as archive:
        for name, content in files.items():
            encoded = content.encode()
            member = tarfile.TarInfo(name)
            member.size = len(encoded)
            archive.addfile(member, io.BytesIO(encoded))


def test_stages_flat_archive_without_hololens_or_unselected_media(tmp_path):
    archives = tmp_path / "archives"
    archives.mkdir()
    sequence_id = "subject_1-20231025_165502"
    _archive(archives / "calibration.tar", {
        "intrinsics.111111111111.yaml": "serial: 111111111111",
        "extrinsics.extrinsics_20231014.yaml": "extrinsics: {}",
    })
    _archive(archives / "sequence_labels.tar", {
        f"{sequence_id}.meta.yaml": "num_frames: 2",
    })
    _archive(archives / f"{sequence_id}.tar", {
        "111111111111-000000.color.jpg": "color",
        "111111111111-000000.depth.png": "depth",
        "111111111111-000000.label.npy": "label 1",
        "222222222222-000000.color.jpg": "other color",
        "222222222222-000000.label.npy": "label 2",
        "111111111111-000001.label.npy": "later label",
        "hololens_kv5h72-000000.color.jpg": "hololens",
    })
    target = tmp_path / "staged"
    assert stage_sequence(archives, sequence_id, target, ("111111111111",), max_frames=1) == (
        "subject_1", "20231025_165502"
    )
    sequence = target / "subject_1" / "20231025_165502"
    assert (sequence / "111111111111" / "color_000000.jpg").read_text() == "color"
    assert (sequence / "111111111111" / "depth_000000.png").read_text() == "depth"
    assert (sequence / "222222222222" / "label_000000.npz").read_text() == "label 2"
    assert not (sequence / "222222222222" / "color_000000.jpg").exists()
    assert not (sequence / "111111111111" / "label_000001.npz").exists()
    assert not (sequence / "hololens_kv5h72").exists()


def test_reused_pair_contains_only_selected_rows(tmp_path):
    source = Episode(root=tmp_path / "eight")
    target = Episode(root=tmp_path / "pair")
    for episode in (source, target):
        episode.raw_dir.mkdir(parents=True)
        episode.status_path.write_text('{"stages": {}}')
        (episode.raw_dir / "intrinsics.json").write_text('{}')
        (episode.raw_dir / "extrinsics.json").write_text('{}')
    cameras = np.array(["camera_a", "camera_b", "camera_c"])
    np.savez(source.rec_npz, device_ids=cameras, timestamps=np.arange(3),
             points=np.arange(189, dtype=np.float32).reshape(3, 21, 3),
             confidence=np.ones((3, 21), dtype=np.float32),
             landmark_ids=np.arange(21, dtype=np.int32))
    np.savez(source.raw_dir / "observations.npz", schema=np.int32(1),
             camera_id=cameras, frame_index=np.arange(3),
             uv=np.arange(126, dtype=np.float32).reshape(3, 21, 2))
    (source.raw_dir / "observations_meta.json").write_text(json.dumps({
        "schema": 1,
        "cameras": {camera: {"calib_id": "old"} for camera in cameras},
        "sampler": {"model": "mediapipe"},
    }))
    reuse_camera_tracks(source, target, ("camera_c", "camera_a"))
    with np.load(target.rec_npz) as data:
        assert data["device_ids"].tolist() == ["camera_a", "camera_c"]
        assert data["timestamps"].tolist() == [0, 2]
        assert data["points"][:, 0, 0].tolist() == [0, 126]
        assert data["landmark_ids"].tolist() == list(range(21))
    with np.load(target.raw_dir / "observations.npz") as data:
        assert data["camera_id"].tolist() == ["camera_a", "camera_c"]
        assert data["frame_index"].tolist() == [0, 2]
    metadata = json.loads((target.raw_dir / "observations_meta.json").read_text())
    assert set(metadata["cameras"]) == {"camera_a", "camera_c"}
    assert metadata["cameras"]["camera_a"]["calib_id"] == calib_id(target.raw_dir)
    status = json.loads(target.status_path.read_text())
    assert status["stages"]["extract"]["frames"] == 2
    assert status["stages"]["extract"]["done"] is True


def test_hocap_profile_keeps_baseline_geometry_and_correct_world_frame():
    baseline = get_profile(CLEAN_LANDMARKS_V1)
    profile = get_profile(HOCAP_MEDIAPIPE_TRI_V1)
    assert profile.detector_model == "mediapipe"
    assert profile.fusion_mode == "triangulate"
    assert profile.triangulation == baseline.triangulation
    assert profile.coordinate_frame == "hocap_tag_1"
    assert baseline.coordinate_frame == "robot_base"


def test_current_hocap_profile_matches_product_v4_except_world_frame():
    product = get_profile(DEFAULT_PERCEPTION_PROFILE)
    hocap = get_profile(HOCAP_STABLE_FUSED_HAND_V4)
    assert hocap.coordinate_frame == "hocap_tag_1"
    assert hocap.articulated_hand_fit == "articulated-landmarks-v1"
    assert {k: v for k, v in hocap.manifest().items() if k not in ("name", "description", "coordinate_frame")} == {
        k: v for k, v in product.manifest().items() if k not in ("name", "description", "coordinate_frame")
    }
