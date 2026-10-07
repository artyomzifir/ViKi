import json

import numpy as np

from viki.contracts import Episode
from viki.episode import mark_stage, read_status
from viki.perception.scene import (
    ScenePerceptionOpts,
    _invalidate_object_model,
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
                schema=np.asarray("viki_object_models_v2"),
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
        schema=np.asarray("viki_object_models_v2"),
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


def test_stage_progress_lets_an_inner_stage_name_itself():
    """``object_model._track_model`` reports its own finer stage. The wrapper
    used to pass ``stage=`` on top of it, which raised TypeError and took the
    whole object-model step down after SAM had already run."""
    from viki.perception.scene import _stage_report

    seen = []
    emit = _stage_report(lambda **fields: seen.append(fields), "object_model")
    emit(frame=1, total=10)
    emit(stage="object_model_core_track", frame=2, total=10, object_id=2)
    assert seen[0]["stage"] == "object_model"
    assert seen[1]["stage"] == "object_model_core_track"


def test_scene_opts_do_not_hand_the_lift_stride_to_the_prompter():
    """Prompting reads one frame, where full resolution is cheap and decides
    whether a 40 mm object is visible at all; the stride bounds the cost of
    lifting every frame."""
    opts = ScenePerceptionOpts(depth_stride=6, prompt_min_object_mm=40.0)
    cfg = opts.prompt_config()
    assert cfg.depth_stride == 1
    assert cfg.min_object_extent_m == 0.040


def test_rebuilt_models_invalidate_cube_plan_and_replay(tmp_path):
    ep = Episode(tmp_path)
    mark_stage(ep, "object_model")
    mark_stage(ep, "retarget", object_grasp=True)
    mark_stage(ep, "object_relative")
    mark_stage(ep, "replay")

    _invalidate_object_model(ep)
    assert read_status(ep)["stages"] == {}


def test_rebuilt_models_keep_independent_hand_only_plan(tmp_path):
    ep = Episode(tmp_path)
    mark_stage(ep, "object_model")
    mark_stage(ep, "retarget", object_grasp=False)
    mark_stage(ep, "object_relative")
    mark_stage(ep, "replay")

    _invalidate_object_model(ep)
    assert set(read_status(ep)["stages"]) == {"retarget", "replay"}
