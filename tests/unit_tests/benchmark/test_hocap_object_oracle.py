"""Oracle object lift reads HO-Cap depth and masks without staging RGB."""

import io
import json
import tarfile

import cv2
import numpy as np

from scripts.run_hocap_object_oracle import build_oracle_cloud


def test_builds_oracle_cloud_directly_from_source_archive(tmp_path):
    cameras = [f"{index:012d}" for index in range(8)]
    episode = tmp_path / "episode"
    (episode / "raw").mkdir(parents=True)
    (episode / "meta.json").write_text(json.dumps({"cameras": {camera: {} for camera in cameras}}))
    (episode / "raw/intrinsics.json").write_text(json.dumps({
        camera: {"color": {"fx": 100.0, "fy": 100.0, "cx": 2.0, "cy": 2.0,
                            "dist_coeffs": [0, 0, 0, 0, 0]}}
        for camera in cameras
    }))
    (episode / "raw/extrinsics.json").write_text(json.dumps({
        camera: {"rvec": [0, 0, 0], "tvec": [index * 0.1, 0, 0]}
        for index, camera in enumerate(cameras)
    }))
    depth = np.full((8, 8), 500, np.uint16)
    encoded = cv2.imencode(".png", depth)[1].tobytes()
    mask = np.zeros((8, 8), np.uint8)
    mask[:4, :4] = 1
    mask[4:, 4:] = 5
    label_stream = io.BytesIO()
    np.savez(label_stream, seg_mask=mask)
    archive_path = tmp_path / "sequence.tar"
    with tarfile.open(archive_path, "w") as archive:
        for camera in cameras:
            for suffix, payload in (("depth.png", encoded), ("label.npy", label_stream.getvalue())):
                name = f"{camera}-000000.{suffix}"
                member = tarfile.TarInfo(name)
                member.size = len(payload)
                archive.addfile(member, io.BytesIO(payload))
    output = tmp_path / "output"
    mapping = build_oracle_cloud(
        archive_path, episode, output,
        object_names=("cube", "a", "b", "c"), frames=1, stride=2,
    )
    assert mapping[1] == "cube"
    with np.load(output / "semantic_cloud/000000.npz", allow_pickle=False) as cloud:
        assert set(cloud["instance_id"]) == {1, 99}
        assert set(cloud["source_camera"]) == set(range(8))
        assert len(cloud["xyz"]) > 0
