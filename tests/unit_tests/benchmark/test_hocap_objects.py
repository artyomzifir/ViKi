"""Exact-frame object reference and gauge-aware track metric tests."""

import io
import json
import tarfile

import numpy as np
import pytest
from scipy.spatial.transform import Rotation

from viki.benchmark.hocap_objects import (
    evaluate_object_models,
    import_object_ground_truth,
)


def _archive(path, members):
    with tarfile.open(path, "w") as archive:
        for filename, content in members.items():
            member = tarfile.TarInfo(filename)
            member.size = len(content)
            archive.addfile(member, io.BytesIO(content))


def _label(translation):
    pose = np.eye(4, dtype=np.float32)
    pose[:3, 3] = translation
    buffer = io.BytesIO()
    np.savez(buffer, obj_poses=pose[None])
    return buffer.getvalue()


def test_imports_camera_object_poses_into_world(tmp_path):
    archives = tmp_path / "archives"
    archives.mkdir()
    episode = tmp_path / "episode"
    (episode / "raw").mkdir(parents=True)
    serial = "111111111111"
    (episode / "meta.json").write_text(json.dumps({
        "source": "hocap", "hocap_subject": "subject_1",
        "hocap_sequence": "20231025_000000", "num_frames": 2,
        "cameras": {serial: {}},
    }))
    (episode / "raw/extrinsics.json").write_text(json.dumps({
        serial: {"rvec": [0, 0, 0], "tvec": [-0.1, 0, 0]},
    }))
    sequence_id = "subject_1-20231025_000000"
    _archive(archives / "sequence_labels.tar", {
        f"{sequence_id}.meta.yaml": b"num_frames: 2\nobject_ids: [cube]\n",
    })
    _archive(archives / f"{sequence_id}.tar", {
        f"{serial}-000000.label.npy": _label([0.2, 0, 0]),
        f"{serial}-000001.label.npy": _label([0.3, 0, 0]),
    })
    output = import_object_ground_truth(archives, sequence_id, episode)
    with np.load(output, allow_pickle=False) as data:
        assert data["object_ids"].tolist() == ["cube"]
        assert data["valid"].all()
        np.testing.assert_allclose(data["poses_world"][:, 0, 0, 3], [0.3, 0.4])
        assert str(data["world_frame"]) == "hocap_tag_1"


def test_object_track_is_scored_after_only_initial_frame_alignment(tmp_path):
    truth = np.broadcast_to(np.eye(4), (3, 1, 4, 4)).copy()
    truth[:, 0, 0, 3] = [0.3, 0.31, 0.32]
    truth[1, 0, :3, :3] = Rotation.from_euler("z", 10, degrees=True).as_matrix()
    truth[2, 0, :3, :3] = Rotation.from_euler("z", 20, degrees=True).as_matrix()
    gt = tmp_path / "gt.npz"
    np.savez(gt, poses_world=truth, valid=np.ones((3, 1), bool),
             object_ids=np.array(["cube"]), world_frame=np.asarray("hocap_tag_1"))
    local_rotation = Rotation.from_euler("z", 30, degrees=True).as_matrix()
    local_offset = np.array([0.03, 0.01, 0.02])
    rotations = truth[:, 0, :3, :3] @ local_rotation
    predictions = tmp_path / "models.npz"
    origins = truth[:, 0, :3, 3] + np.einsum(
        "tij,j->ti", truth[:, 0, :3, :3], local_offset
    )
    np.savez(predictions, object_ids=np.array([7]),
             rotation_world_object=rotations[:, None],
             translation_world=origins[:, None],
             track_confidence=np.array([[1.0], [1.0], [0.1]]))
    score = evaluate_object_models(gt, predictions, {7: "cube"})
    item = score["objects"]["cube"]
    assert item["tracked_frames"] == 2
    assert item["coverage"] == pytest.approx(2 / 3)
    assert item["translation_mean_mm"] == pytest.approx(0, abs=1e-8)
    assert item["rotation_mean_deg"] == pytest.approx(0, abs=1e-6)
    with pytest.raises(ValueError, match="mapping is incomplete"):
        evaluate_object_models(gt, predictions, {8: "cube"})
