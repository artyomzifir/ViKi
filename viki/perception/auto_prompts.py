"""Background-derived SAM prompt proposals for recorded RGB-D scenes.

The empty-scene depth plate provides a deliberately simple source of truth:
anything sufficiently closer to or different from that plate is foreground.
We lift the foreground from every calibrated camera, fuse it in the rig frame,
select compact multi-view components as object candidates, and project those
same 3-D components back to each colour image as SAM boxes and clicks.

This is an experimental proposal generator, not semantic ground truth. It can
separate a large operator component from compact colourful rigid objects, but it
cannot infer which object is the task's manipulated object from frame zero.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
import json
from pathlib import Path
from typing import Any

import cv2
import numpy as np
from scipy.spatial import cKDTree

from viki import config as viki_config
from viki.calibration import background_depth
from viki.contracts import CalibrationExtrinsics
from viki.perception.cloud import (
    _bbox_mask,
    _bbox_to_frame,
    _camera_samples,
    _color_K,
    _depth_K,
    _read_json,
    _voxel_downsample_indices,
)
from viki.perception.object_model import _radius_components
from viki.perception.segmentation import LABEL_CODES, PROMPT_SCHEMA


AUTO_PROMPT_METHOD = "background_depth_3d_components_v1"


@dataclass(frozen=True)
class AutoPromptConfig:
    """Conservative geometry thresholds for automatic frame-zero prompts."""

    frame: int = 0
    object_count: int = 3
    object_label: str = "other_dynamic"
    depth_stride: int = 2
    background_tolerance_mm: float = 50.0
    voxel_m: float = 0.006
    component_radius_m: float = 0.018
    min_component_points: int = 100
    min_object_saturation: float = 0.25
    max_object_extent_m: float = 0.25
    box_percentile: float = 1.0
    box_margin_px: int = 8


@dataclass(frozen=True)
class _Component:
    rows: np.ndarray
    point_count: int
    camera_count: int
    centroid: np.ndarray
    dimensions: np.ndarray
    saturation: float


@dataclass(frozen=True)
class _CameraSamples:
    camera: str
    width: int
    height: int
    xyz: np.ndarray
    rgb: np.ndarray
    uv: np.ndarray


def _validate_config(cfg: AutoPromptConfig) -> None:
    if cfg.frame < 0:
        raise ValueError("frame must be non-negative")
    if cfg.object_count <= 0:
        raise ValueError("object_count must be positive")
    if cfg.object_label not in LABEL_CODES or cfg.object_label in {
        "operator", "ambiguous_contact",
    }:
        raise ValueError(f"unsupported automatic object label {cfg.object_label!r}")
    if cfg.depth_stride <= 0:
        raise ValueError("depth_stride must be positive")
    if cfg.background_tolerance_mm < 0:
        raise ValueError("background_tolerance_mm must be non-negative")
    if cfg.voxel_m <= 0 or cfg.component_radius_m <= 0:
        raise ValueError("voxel_m and component_radius_m must be positive")
    if cfg.min_component_points <= 0:
        raise ValueError("min_component_points must be positive")
    if not 0 <= cfg.min_object_saturation <= 1:
        raise ValueError("min_object_saturation must be in [0, 1]")
    if cfg.max_object_extent_m <= 0:
        raise ValueError("max_object_extent_m must be positive")
    if not 0 <= cfg.box_percentile < 50:
        raise ValueError("box_percentile must be in [0, 50)")
    if cfg.box_margin_px < 0:
        raise ValueError("box_margin_px must be non-negative")


def _colour_saturation(rgb: np.ndarray) -> np.ndarray:
    values = np.asarray(rgb, dtype=np.float64)
    high = values.max(axis=1)
    low = values.min(axis=1)
    return (high - low) / np.maximum(high, 1.0)


def _describe_components(
    xyz: np.ndarray,
    rgb: np.ndarray,
    source_camera: np.ndarray,
    cfg: AutoPromptConfig,
) -> list[_Component]:
    result: list[_Component] = []
    saturation = _colour_saturation(rgb)
    for rows in _radius_components(xyz, cfg.component_radius_m):
        if len(rows) < cfg.min_component_points:
            continue
        points = xyz[rows]
        low, high = np.percentile(points, [2, 98], axis=0)
        result.append(_Component(
            rows=rows,
            point_count=len(rows),
            camera_count=len(np.unique(source_camera[rows])),
            centroid=np.median(points, axis=0),
            dimensions=high - low,
            saturation=float(np.median(saturation[rows])),
        ))
    return result


def _select_prompt_components(
    xyz: np.ndarray,
    rgb: np.ndarray,
    source_camera: np.ndarray,
    camera_count: int,
    cfg: AutoPromptConfig,
) -> tuple[_Component, list[_Component], list[_Component]]:
    """Choose one large operator and ``object_count`` compact components."""

    components = _describe_components(xyz, rgb, source_camera, cfg)
    required_views = min(2, camera_count)
    candidates = [
        item for item in components
        if item.camera_count >= required_views
        and item.saturation >= cfg.min_object_saturation
        and float(item.dimensions.max()) <= cfg.max_object_extent_m
    ]
    candidates.sort(
        key=lambda item: (
            item.camera_count,
            item.point_count * item.saturation,
            item.point_count,
        ),
        reverse=True,
    )
    if len(candidates) < cfg.object_count:
        summary = [
            {
                "points": item.point_count,
                "views": item.camera_count,
                "max_extent_m": round(float(item.dimensions.max()), 4),
                "saturation": round(item.saturation, 3),
            }
            for item in components[:10]
        ]
        raise ValueError(
            f"found {len(candidates)} compact multi-view object candidates, "
            f"need {cfg.object_count}; leading components: {summary}"
        )

    objects = candidates[:cfg.object_count]
    selected = {id(item) for item in objects}
    remaining = [item for item in components if id(item) not in selected]
    if not remaining:
        raise ValueError("no foreground component remains for the operator")
    operator = max(remaining, key=lambda item: item.point_count)
    if operator.camera_count < required_views:
        raise ValueError("largest operator candidate is not visible in enough cameras")

    # A spatial order is stable across cameras and keeps ids deterministic.
    objects.sort(key=lambda item: tuple(float(value) for value in item.centroid))
    return operator, objects, components


def _interior_point(uv: np.ndarray, width: int, height: int) -> tuple[int, int]:
    pixels = np.rint(uv).astype(np.int32)
    pixels[:, 0] = np.clip(pixels[:, 0], 0, width - 1)
    pixels[:, 1] = np.clip(pixels[:, 1], 0, height - 1)
    # Select an actually observed foreground pixel nearest the robust centre.
    # This keeps the click on the 3-D component even for a concave object, while
    # avoiding image-edge bias when the operator is clipped by the frame.
    median_xy = np.median(pixels, axis=0)
    nearest = int(np.argmin(np.square(pixels - median_xy).sum(axis=1)))
    return int(pixels[nearest, 0]), int(pixels[nearest, 1])


def _box_from_uv(
    uv: np.ndarray,
    width: int,
    height: int,
    cfg: AutoPromptConfig,
) -> list[int]:
    low = np.percentile(uv, cfg.box_percentile, axis=0)
    high = np.percentile(uv, 100.0 - cfg.box_percentile, axis=0)
    margin = cfg.box_margin_px
    x0 = int(np.clip(np.floor(low[0]) - margin, 0, width - 2))
    y0 = int(np.clip(np.floor(low[1]) - margin, 0, height - 2))
    x1 = int(np.clip(np.ceil(high[0]) + margin, x0 + 1, width - 1))
    y1 = int(np.clip(np.ceil(high[1]) + margin, y0 + 1, height - 1))
    return [x0, y0, x1, y1]


def _read_camera_samples(ep, cfg: AutoPromptConfig) -> tuple[list[_CameraSamples], str]:
    from viki.perception.k4a_offline import K4ACalibration
    from viki.perception.rs_offline import RealSenseCalibration

    raw = ep.raw_dir
    intrinsics = _read_json(raw / "intrinsics.json")
    extrinsics = _read_json(raw / "extrinsics.json")
    episode_meta = _read_json(ep.meta_path)
    preset = str(episode_meta.get("calibration_preset") or "")
    if not preset:
        raise ValueError("automatic prompts require an episode calibration_preset")
    bbox = _bbox_to_frame(
        list(getattr(viki_config, "CLOUD_WORKSPACE_BBOX", []) or []),
        _read_json(raw / "world_anchor.json").get("T_world_display"),
    )

    cameras: list[_CameraSamples] = []
    for video_path in sorted(raw.glob("*.mp4")):
        camera = video_path.stem
        entry = extrinsics.get(camera)
        if entry is None:
            continue
        background = background_depth(preset, camera)
        if background is None:
            raise FileNotFoundError(
                f"calibration preset {preset!r} has no background for {camera}"
            )
        depth_path = raw / f"{camera}_depth" / f"{cfg.frame:06d}.npy"
        if not depth_path.is_file():
            raise FileNotFoundError(depth_path)

        capture = cv2.VideoCapture(str(video_path))
        capture.set(cv2.CAP_PROP_POS_FRAMES, cfg.frame)
        ok, bgr = capture.read()
        capture.release()
        if not ok:
            raise RuntimeError(f"could not read frame {cfg.frame} from {video_path}")

        transform = CalibrationExtrinsics(
            rvec=np.asarray(entry["rvec"], np.float64),
            tvec=np.asarray(entry["tvec"], np.float64),
        ).transform_matrix
        calibration = (
            K4ACalibration.from_episode(raw, camera, episode_meta)
            or RealSenseCalibration.from_episode(raw, camera, episode_meta)
        )
        xyz, rgb, uv = _camera_samples(
            bgr,
            np.load(depth_path),
            cfg.depth_stride,
            _color_K(intrinsics.get(camera, {})),
            calibration,
            transform,
            bg_mm=background,
            bg_tol_mm=cfg.background_tolerance_mm,
            K_depth=_depth_K(intrinsics.get(camera, {})),
            edge_filter=bool(getattr(viki_config, "CLOUD_EDGE_FILTER", True)),
            edge_radius_rad=float(
                getattr(viki_config, "CLOUD_EDGE_RADIUS_RAD", 0.004)
            ),
            edge_jump_mm=float(getattr(viki_config, "CLOUD_EDGE_JUMP_MM", 30.0)),
            edge_jump_relative=float(
                getattr(viki_config, "CLOUD_EDGE_JUMP_RELATIVE", 0.02)
            ),
        )
        keep = _bbox_mask(xyz, bbox)
        cameras.append(_CameraSamples(
            camera=camera,
            width=int(bgr.shape[1]),
            height=int(bgr.shape[0]),
            xyz=xyz[keep],
            rgb=rgb[keep],
            uv=uv[keep],
        ))
    if not cameras:
        raise ValueError("automatic prompts found no calibrated RGB-D cameras")
    return cameras, preset


def _component_record(
    component: _Component,
    *,
    object_id: int,
    role: str,
) -> dict[str, Any]:
    return {
        "object_id": object_id,
        "role": role,
        "point_count_voxelized": component.point_count,
        "camera_count": component.camera_count,
        "centroid_rig_m": np.round(component.centroid, 6).tolist(),
        "visible_dimensions_rig_m": np.round(component.dimensions, 6).tolist(),
        "median_rgb_saturation": round(component.saturation, 6),
    }


def generate_auto_prompts(
    ep,
    output_path: str | Path | None = None,
    *,
    config: AutoPromptConfig | None = None,
) -> Path:
    """Generate a validated SAM prompt manifest from calibrated foreground.

    Object candidates receive the same conservative label because a single seed
    frame does not reveal task role. Object-centric tracking can later decide
    which rigid instance is active in each phase.
    """

    cfg = config or AutoPromptConfig()
    _validate_config(cfg)
    cameras, preset = _read_camera_samples(ep, cfg)
    xyz = np.concatenate([item.xyz for item in cameras])
    rgb = np.concatenate([item.rgb for item in cameras])
    source_camera = np.concatenate([
        np.full(len(item.xyz), index, np.int16)
        for index, item in enumerate(cameras)
    ])
    uv = np.concatenate([item.uv for item in cameras])

    voxel_rows = _voxel_downsample_indices(xyz, cfg.voxel_m)
    voxel_xyz = xyz[voxel_rows]
    voxel_rgb = rgb[voxel_rows]
    voxel_camera = source_camera[voxel_rows]
    operator, objects, all_components = _select_prompt_components(
        voxel_xyz,
        voxel_rgb,
        voxel_camera,
        len(cameras),
        cfg,
    )

    selected_components = [operator, *objects]
    object_ids = list(range(1, len(selected_components) + 1))
    voxel_labels = np.full(len(voxel_xyz), -1, np.int16)
    for label, component in enumerate(selected_components):
        voxel_labels[component.rows] = label
    distance, nearest = cKDTree(voxel_xyz).query(xyz, k=1)
    sample_labels = voxel_labels[nearest]
    sample_labels[distance > cfg.voxel_m * np.sqrt(3.0) + 1e-9] = -1

    camera_prompts: dict[str, list[dict[str, Any]]] = {}
    for camera_index, camera in enumerate(cameras):
        instance_uv: list[np.ndarray] = []
        positive_points: list[tuple[int, int]] = []
        for label in range(len(selected_components)):
            rows = (source_camera == camera_index) & (sample_labels == label)
            projected = uv[rows]
            if len(projected) < 10:
                raise ValueError(
                    f"component {label + 1} has only {len(projected)} projected "
                    f"samples in {camera.camera}"
                )
            instance_uv.append(projected)
            positive_points.append(
                _interior_point(projected, camera.width, camera.height)
            )

        prompts: list[dict[str, Any]] = []
        for label, projected in enumerate(instance_uv):
            points = [positive_points[label]] + [
                point for other, point in enumerate(positive_points) if other != label
            ]
            prompts.append({
                "frame": cfg.frame,
                "object_id": object_ids[label],
                "box": _box_from_uv(projected, camera.width, camera.height, cfg),
                "points": [list(point) for point in points],
                "point_labels": [1] + [0] * (len(points) - 1),
            })
        camera_prompts[camera.camera] = prompts

    objects_manifest = [{"id": 1, "label": "operator"}] + [
        {"id": object_id, "label": cfg.object_label}
        for object_id in object_ids[1:]
    ]
    records = [_component_record(operator, object_id=1, role="operator")]
    records.extend(
        _component_record(component, object_id=object_id, role="object_candidate")
        for object_id, component in zip(object_ids[1:], objects)
    )
    manifest = {
        "schema": PROMPT_SCHEMA,
        "objects": objects_manifest,
        "cameras": camera_prompts,
        "auto_annotation": {
            "method": AUTO_PROMPT_METHOD,
            "calibration_preset": preset,
            "parameters": asdict(cfg),
            "foreground_samples": int(len(xyz)),
            "voxel_samples": int(len(voxel_xyz)),
            "retained_components": int(len(all_components)),
            "components": records,
            "limitation": (
                "frame-zero geometry separates operator and compact objects; "
                "task role remains unknown"
            ),
        },
    }
    destination = Path(output_path) if output_path is not None else (
        ep.intermediates_dir / "segmentation" / "auto_prompts.json"
    )
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(manifest, indent=2) + "\n")
    return destination


__all__ = ["AUTO_PROMPT_METHOD", "AutoPromptConfig", "generate_auto_prompts"]
