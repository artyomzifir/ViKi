"""Durable scene-perception stages for the normal offline pipeline.

SAM remains in its dedicated CUDA image, but its masks, calibrated semantic
cloud and compact rigid-object models are ordinary episode artifacts.  This
module owns the resumable orchestration and status bookkeeping so callers do
not need to know the internal directory layout.
"""

from __future__ import annotations

from dataclasses import dataclass, field
import json
from pathlib import Path
from typing import Callable

import numpy as np

from viki.contracts import Episode
from viki.episode import clear_stage, mark_stage, read_status, stage_done
from viki.perception.auto_prompts import AutoPromptConfig, generate_auto_prompts
from viki.perception.object_model import (
    OBJECT_MODEL_SCHEMA,
    ObjectModelConfig,
    build_object_models,
)
from viki.perception.segmentation import (
    MASK_SCHEMA,
    SAM2_MODEL_KEY,
    SEMANTIC_CLOUD_SCHEMA,
    lift_segmentation_to_3d,
    segment_episode_sam2,
)


Report = Callable[..., None]


def _noop(**_fields) -> None:
    return None


def _relative(ep: Episode, path: Path) -> str:
    try:
        return str(path.relative_to(ep.root))
    except ValueError:
        return str(path)


def _stage_report(report: Report, stage: str) -> Report:
    """Label a stage's progress, unless it labels itself.

    ``_track_model`` emits its own finer ``stage`` (bootstrap vs track), so this
    is a default and not an override — passing both raised
    ``TypeError: got multiple values for keyword argument 'stage'`` and took the
    whole object-model stage down after SAM had already done its work.
    """
    def emit(**fields) -> None:
        fields.setdefault("stage", stage)
        report(**fields)

    return emit


def _invalidate_object_dependents(ep: Episode) -> None:
    if stage_done(ep, "object_relative"):
        clear_stage(ep, "object_relative")
    retarget_status = read_status(ep).get("stages", {}).get("retarget", {})
    if retarget_status.get("done") and retarget_status.get("object_grasp") is not False:
        clear_stage(ep, "retarget")
        if stage_done(ep, "replay"):
            clear_stage(ep, "replay")


def _invalidate_object_model(ep: Episode) -> None:
    clear_stage(ep, "object_model")
    _invalidate_object_dependents(ep)
    if ep.object_models_npz.is_file():
        ep.object_models_npz.unlink()


def _mark_segment_masks(ep: Episode, output: Path, prompts: str | Path) -> None:
    meta = json.loads((output / "meta.json").read_text())
    if meta.get("schema") != MASK_SCHEMA:
        raise ValueError(f"unsupported segmentation schema {meta.get('schema')!r}")
    cameras = meta.get("cameras", {})
    frames = min(
        (int(value.get("frames", 0)) for value in cameras.values()),
        default=0,
    )
    mark_stage(
        ep,
        "segment",
        profile=SAM2_MODEL_KEY,
        masks=True,
        lifted_3d=False,
        frames=frames,
        cameras=sorted(cameras),
        prompts=_relative(ep, Path(prompts)),
        artifact=_relative(ep, output),
    )


def _mark_segment_3d(ep: Episode, output: Path) -> None:
    meta = json.loads((output / "meta.json").read_text())
    if meta.get("schema") != SEMANTIC_CLOUD_SCHEMA:
        raise ValueError(f"unsupported semantic-cloud schema {meta.get('schema')!r}")
    mark_stage(
        ep,
        "segment",
        profile=SAM2_MODEL_KEY,
        masks=True,
        lifted_3d=True,
        frames=int(meta.get("frames", 0)),
        semantic_cloud=_relative(ep, output),
        stride=int(meta.get("stride", 0)),
        voxel_m=float(meta.get("voxel_m", 0.0)),
    )


def _mark_object_model(ep: Episode, result: Path) -> None:
    with np.load(result, allow_pickle=False) as archive:
        schema = str(np.asarray(archive["schema"]))
        if schema != OBJECT_MODEL_SCHEMA:
            raise ValueError(f"unsupported object-model schema {schema!r}")
        object_ids = np.asarray(archive["object_ids"], dtype=np.int32)
        frames = int(np.asarray(archive["frame_count"]))
        metrics = json.loads(str(np.asarray(archive["metrics_json"])))
    mark_stage(
        ep,
        "object_model",
        profile=SAM2_MODEL_KEY,
        frames=frames,
        objects=int(len(object_ids)),
        object_ids=object_ids.tolist(),
        elapsed_sec=float(metrics.get("elapsed_sec", 0.0)),
        artifact=_relative(ep, result),
    )


@dataclass(frozen=True)
class ScenePerceptionOpts:
    """Options for masks → semantic RGB-D → compact object models.

    ``prompts=None`` selects calibrated background-derived prompt generation.
    A supplied prompt manifest is kept as provenance and is never rewritten.
    """

    prompts: str | Path | None = None
    object_count: int = 1
    object_label: str = "other_dynamic"
    checkpoint: str | Path = "models/sam2/sam2.1_hiera_small.pt"
    chunk_frames: int = 100
    render_overlay: bool = True
    depth_stride: int = 2
    voxel_m: float = 0.004
    # Prompt clustering is a separate grid from ``voxel_m`` (which is the
    # semantic cloud's). Its resolution sets the smallest object that can be
    # prompted at all: a 40 mm cube is ~70 voxels on the 6 mm default grid,
    # under ``prompt_min_points``, so it is dropped before it can be ranked.
    # On a 3 mm grid the same cube is a clean 291-point two-view component.
    prompt_voxel_m: float = 0.006
    prompt_min_points: int = 100
    prompt_radius_m: float = 0.018
    prompt_min_object_mm: float = 30.0
    prompt_frame_attempts: int = 6
    auto_prompts: AutoPromptConfig | None = None
    object_model: ObjectModelConfig = field(default_factory=ObjectModelConfig)

    def prompt_config(self) -> AutoPromptConfig:
        if self.auto_prompts is not None:
            return self.auto_prompts
        # ``depth_stride`` is deliberately not inherited: it exists to bound the
        # cost of lifting *every* frame, while prompting reads one, where full
        # resolution is both cheap and the difference between seeing a 40 mm
        # object and not seeing it.
        return AutoPromptConfig(
            object_count=self.object_count,
            object_label=self.object_label,
            voxel_m=self.prompt_voxel_m,
            min_component_points=self.prompt_min_points,
            component_radius_m=self.prompt_radius_m,
            min_object_extent_m=self.prompt_min_object_mm / 1000.0,
            frame_attempts=self.prompt_frame_attempts,
        )


def segment_episode(
    ep: Episode,
    prompts: str | Path,
    *,
    checkpoint: str | Path = "models/sam2/sam2.1_hiera_small.pt",
    chunk_frames: int = 100,
    max_frames: int | None = None,
    render_overlay: bool = True,
    report: Report | None = None,
) -> Path:
    """Run SAM masks as a tracked episode stage."""
    clear_stage(ep, "segment")
    _invalidate_object_model(ep)
    output = segment_episode_sam2(
        ep,
        prompts,
        checkpoint=checkpoint,
        chunk_frames=chunk_frames,
        max_frames=max_frames,
        render_overlay=render_overlay,
        report=_stage_report(report or _noop, "segment_masks"),
    )
    _mark_segment_masks(ep, output, prompts)
    return output


def lift_segmented_episode(
    ep: Episode,
    segmentation_dir: str | Path | None = None,
    *,
    stride: int = 2,
    voxel_m: float = 0.004,
    report: Report | None = None,
) -> Path:
    """Lift saved masks into calibrated 3-D and update the segment stage."""
    root = Path(segmentation_dir) if segmentation_dir is not None else ep.segmentation_dir
    _invalidate_object_model(ep)
    output = lift_segmentation_to_3d(
        ep,
        root,
        stride=stride,
        voxel_m=voxel_m,
        report=_stage_report(report or _noop, "segment_lift"),
    )
    _mark_segment_3d(ep, output)
    return output


def build_object_models_episode(
    ep: Episode,
    segmentation_dir: str | Path | None = None,
    *,
    config: ObjectModelConfig | None = None,
    output: str | Path | None = None,
    report: Report | None = None,
) -> Path:
    """Build compact object models as a tracked episode stage."""
    _invalidate_object_dependents(ep)
    root = Path(segmentation_dir) if segmentation_dir is not None else ep.segmentation_dir
    result = build_object_models(
        root,
        config=config,
        output=output,
        report=_stage_report(report or _noop, "object_model"),
    )
    _mark_object_model(ep, result)
    return result


def _segment_has_3d(ep: Episode) -> bool:
    entry = read_status(ep).get("stages", {}).get("segment", {})
    return bool(
        entry.get("done")
        and entry.get("lifted_3d")
        and (ep.segmentation_dir / "meta.json").is_file()
        and (ep.segmentation_dir / "semantic_cloud" / "meta.json").is_file()
    )


def scene_perception_episode(
    ep: Episode,
    opts: ScenePerceptionOpts | None = None,
    *,
    force: bool = False,
    report: Report | None = None,
) -> Path:
    """Run or resume the complete scene-perception branch for an episode.

    The branch is a first-class part of the offline artifact graph, but it is
    explicit because SAM requires the dedicated CUDA image and object count is
    a property of the recording.  Cube-aware retarget explicitly requires one
    labelled manipulated-object track; other tasks may use hand-only IK.
    """
    options = opts or ScenePerceptionOpts()
    progress = report or _noop
    if options.object_count <= 0:
        raise ValueError("object_count must be positive")

    # v0.0.2 was developed against already-generated probe artifacts. Adopt a
    # valid legacy artifact into the formal stage graph instead of spending
    # minutes rerunning SAM solely because old status.json lacks the new keys.
    if not force and not stage_done(ep, "segment"):
        semantic_meta = ep.segmentation_dir / "semantic_cloud" / "meta.json"
        if (ep.segmentation_dir / "meta.json").is_file() and semantic_meta.is_file():
            _mark_segment_masks(
                ep,
                ep.segmentation_dir,
                options.prompts or ep.scene_prompts_path,
            )
            _mark_segment_3d(ep, semantic_meta.parent)
    if not force and not stage_done(ep, "object_model") and ep.object_models_npz.is_file():
        _mark_object_model(ep, ep.object_models_npz)

    if force or not _segment_has_3d(ep):
        masks_ready = (
            not force
            and stage_done(ep, "segment")
            and (ep.segmentation_dir / "meta.json").is_file()
        )
        if not masks_ready:
            prompts = Path(options.prompts) if options.prompts is not None else None
            if prompts is None:
                progress(stage="auto_prompts", frame=0, total=1)
                prompts = generate_auto_prompts(
                    ep,
                    ep.scene_prompts_path,
                    config=options.prompt_config(),
                )
                progress(stage="auto_prompts", frame=1, total=1)
            segment_episode(
                ep,
                prompts,
                checkpoint=options.checkpoint,
                chunk_frames=options.chunk_frames,
                render_overlay=options.render_overlay,
                report=progress,
            )
        lift_segmented_episode(
            ep,
            stride=options.depth_stride,
            voxel_m=options.voxel_m,
            report=progress,
        )

    if force or not (stage_done(ep, "object_model") and ep.object_models_npz.is_file()):
        build_object_models_episode(
            ep,
            config=options.object_model,
            report=progress,
        )
    progress(stage="scene_done", frame=1, total=1)
    return ep.object_models_npz


__all__ = [
    "ScenePerceptionOpts",
    "build_object_models_episode",
    "lift_segmented_episode",
    "scene_perception_episode",
    "segment_episode",
]
