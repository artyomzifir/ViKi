import json

import numpy as np

from viki.contracts import Episode
from viki.episode import read_status
from viki.perception.scene import (
    ScenePerceptionOpts,
    lift_segmented_episode,
    scene_perception_episode,
)


def test_scene_pipeline_is_tracked_and_resumable(tmp_path, monkeypatch):
    root = tmp_path / "episode"
    root.mkdir()
    (root / "status.json").write_text('{"stages": {}}')
    ep = Episode(root)
    calls = []

    def fake_prompts(_ep, output, *, config):
        calls.append(("prompts", config.object_count))
        output.parent.mkdir(parents=True, exist_ok=True)
        output.write_text('{"schema": "viki_sam2_prompts_v1"}')
        return output

    def fake_segment(_ep, prompts, **_kwargs):
        calls.append(("masks", str(prompts)))
        ep.segmentation_dir.mkdir(parents=True, exist_ok=True)
        (ep.segmentation_dir / "meta.json").write_text(json.dumps({
            "schema": "viki_sam2_masks_v1",
            "cameras": {
                "kinect_0": {"frames": 12},
                "kinect_1": {"frames": 12},
            },
        }))
        return ep.segmentation_dir

    def fake_lift(_ep, segmentation, **_kwargs):
        calls.append(("lift", str(segmentation)))
        output = ep.segmentation_dir / "semantic_cloud"
        output.mkdir(parents=True, exist_ok=True)
        (output / "meta.json").write_text(json.dumps({
            "schema": "viki_semantic_cloud_v1",
            "frames": 12,
            "stride": 2,
            "voxel_m": 0.004,
        }))
        return output

    def fake_models(segmentation, **_kwargs):
        calls.append(("models", str(segmentation)))
        output = ep.object_models_npz
        with output.open("wb") as stream:
            np.savez_compressed(
                stream,
                schema=np.asarray("viki_object_models_v1"),
                object_ids=np.asarray([2, 3], np.int32),
                frame_count=np.int32(12),
                metrics_json=np.asarray(json.dumps({"elapsed_sec": 1.25})),
            )
        return output

    monkeypatch.setattr("viki.perception.scene.generate_auto_prompts", fake_prompts)
    monkeypatch.setattr("viki.perception.scene.segment_episode_sam2", fake_segment)
    monkeypatch.setattr("viki.perception.scene.lift_segmentation_to_3d", fake_lift)
    monkeypatch.setattr("viki.perception.scene.build_object_models", fake_models)

    result = scene_perception_episode(ep, ScenePerceptionOpts(object_count=2))
    assert result == ep.object_models_npz
    assert [item[0] for item in calls] == ["prompts", "masks", "lift", "models"]

    stages = read_status(ep)["stages"]
    assert stages["segment"]["done"] is True
    assert stages["segment"]["lifted_3d"] is True
    assert stages["segment"]["frames"] == 12
    assert stages["object_model"]["done"] is True
    assert stages["object_model"]["objects"] == 2
    assert stages["object_model"]["artifact"].endswith("object_models.npz")

    # Completed durable artifacts make the stage idempotent unless forced.
    assert scene_perception_episode(ep, ScenePerceptionOpts(object_count=2)) == result
    assert len(calls) == 4

    # Re-lifting changed masks/depth invalidates the downstream model instead
    # of exporting a stale object geometry under a completed-stage claim.
    lift_segmented_episode(ep)
    stages = read_status(ep)["stages"]
    assert stages["segment"]["lifted_3d"] is True
    assert "object_model" not in stages
    assert not ep.object_models_npz.exists()


def test_scene_pipeline_adopts_valid_pre_stage_artifacts(tmp_path, monkeypatch):
    root = tmp_path / "episode"
    root.mkdir()
    (root / "status.json").write_text('{"stages": {}}')
    ep = Episode(root)
    semantic = ep.segmentation_dir / "semantic_cloud"
    semantic.mkdir(parents=True)
    (ep.segmentation_dir / "meta.json").write_text(json.dumps({
        "schema": "viki_sam2_masks_v1",
        "cameras": {"kinect_0": {"frames": 8}},
    }))
    (semantic / "meta.json").write_text(json.dumps({
        "schema": "viki_semantic_cloud_v1",
        "frames": 8,
        "stride": 2,
        "voxel_m": 0.004,
    }))
    np.savez_compressed(
        ep.object_models_npz,
        schema=np.asarray("viki_object_models_v1"),
        object_ids=np.asarray([2], np.int32),
        frame_count=np.int32(8),
        metrics_json=np.asarray(json.dumps({"elapsed_sec": 2.0})),
    )

    def should_not_run(*_args, **_kwargs):
        raise AssertionError("valid durable artifact should be adopted, not recomputed")

    monkeypatch.setattr("viki.perception.scene.generate_auto_prompts", should_not_run)
    monkeypatch.setattr("viki.perception.scene.segment_episode_sam2", should_not_run)
    monkeypatch.setattr("viki.perception.scene.lift_segmentation_to_3d", should_not_run)
    monkeypatch.setattr("viki.perception.scene.build_object_models", should_not_run)

    assert scene_perception_episode(ep) == ep.object_models_npz
    stages = read_status(ep)["stages"]
    assert stages["segment"]["lifted_3d"] is True
    assert stages["object_model"]["objects"] == 1
