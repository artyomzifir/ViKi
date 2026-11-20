"""The A/B crop must preserve paired frame alignment without overwriting inputs."""

import json

import numpy as np
import pytest

from scripts.prepare_hocap_grasp_pilot import prepare


def test_prepare_hocap_grasp_pilot_crops_both_tracks(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    (source / "meta.json").write_text(json.dumps({
        "source": "hocap", "hocap_world_frame": "tag_1",
    }))
    np.savez_compressed(
        source / "cln.npz", timestamps=np.arange(10),
        positions=np.arange(30).reshape(10, 3), coordinate_frame="hocap_tag_1",
    )
    track_dir = source / "intermediates/segmentation/hocap-oracle-v1"
    track_dir.mkdir(parents=True)
    np.savez_compressed(
        track_dir / "object_models.npz", schema="sample", object_ids=np.array([4]),
        model_offsets=np.array([0, 1]), model_xyz_object=np.zeros((1, 3)),
        translation_world=np.zeros((10, 1, 3)),
        rotation_world_object=np.broadcast_to(np.eye(3), (10, 1, 3, 3)),
        track_confidence=np.ones((10, 1)),
    )
    hand_output, object_output = tmp_path / "hand", tmp_path / "object"
    prepare(source, hand_output, object_output, 3, 8)
    with np.load(hand_output / "cln.npz") as hand:
        assert hand["timestamps"].tolist() == [3, 4, 5, 6, 7]
    with np.load(object_output / "object_track.npz") as tracked:
        assert tracked["translation_world"].shape == (5, 1, 3)
        assert tracked["model_xyz_object"].shape == (1, 3)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        prepare(source, hand_output, object_output, 3, 8)
