"""Experimental SAM 2.1 video masks and calibrated RGB-D lift.

This module deliberately writes below ``intermediates/segmentation``.  Its
binary masks are proposals with prompt provenance, not a production semantic
truth source, and no downstream stage consumes them implicitly.
"""

from __future__ import annotations

import gc
import hashlib
import json
import logging
import tempfile
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from viki import config
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

logger = logging.getLogger(__name__)

SAM2_UPSTREAM_COMMIT = "2b90b9f5ceec907a1c18123530e92e794ad901a4"
SAM2_MODEL_KEY = "sam2.1_hiera_small"
SAM2_MODEL_CONFIG = "configs/sam2.1/sam2.1_hiera_s.yaml"
SAM2_CHECKPOINT_SHA256 = "6d1aa6f30de5c92224f8172114de081d104bbd23dd9dc5c58996f0cad5dc4d38"
PROMPT_SCHEMA = "viki_sam2_prompts_v1"
MASK_SCHEMA = "viki_sam2_masks_v1"
SEMANTIC_CLOUD_SCHEMA = "viki_semantic_cloud_v1"

LABEL_CODES = {
    "operator": 1,
    "manipulated_object": 2,
    "other_object": 3,
    "other_dynamic": 4,
    "ambiguous_contact": 255,
}


@dataclass(frozen=True)
class Prompt:
    frame: int
    object_id: int
    box: tuple[float, float, float, float] | None
    points: np.ndarray | None
    point_labels: np.ndarray | None


@dataclass(frozen=True)
class PromptSpec:
    objects: dict[int, str]
    cameras: dict[str, tuple[Prompt, ...]]
    source: dict[str, Any]


def _finite_vector(value: Any, length: int, name: str) -> np.ndarray:
    array = np.asarray(value, dtype=np.float32)
    if array.shape != (length,) or not np.isfinite(array).all():
        raise ValueError(f"{name} must be a finite vector of length {length}")
    return array


def load_prompt_spec(path: str | Path) -> PromptSpec:
    """Read and strictly validate a scene-specific prompt manifest."""
    source = json.loads(Path(path).read_text())
    if source.get("schema") != PROMPT_SCHEMA:
        raise ValueError(f"prompt schema must be {PROMPT_SCHEMA!r}")

    objects: dict[int, str] = {}
    for item in source.get("objects", []):
        object_id = int(item["id"])
        label = str(item["label"])
        if object_id <= 0 or object_id in objects:
            raise ValueError("object ids must be unique positive integers")
        if label not in LABEL_CODES or label == "ambiguous_contact":
            raise ValueError(f"unsupported prompt label {label!r}")
        objects[object_id] = label
    if not objects:
        raise ValueError("prompt manifest contains no objects")

    cameras: dict[str, tuple[Prompt, ...]] = {}
    for camera, items in (source.get("cameras") or {}).items():
        prompts = []
        for item in items:
            frame = int(item["frame"])
            object_id = int(item["object_id"])
            if frame < 0:
                raise ValueError("prompt frame must be non-negative")
            if object_id not in objects:
                raise ValueError(f"prompt references unknown object id {object_id}")
            box = None
            if item.get("box") is not None:
                values = _finite_vector(item["box"], 4, "box")
                if values[0] < 0 or values[1] < 0 or values[2] <= values[0] or values[3] <= values[1]:
                    raise ValueError("box must be non-negative [x0,y0,x1,y1]")
                box = tuple(float(v) for v in values)
            points = point_labels = None
            if item.get("points") is not None:
                points = np.asarray(item["points"], dtype=np.float32)
                point_labels = np.asarray(item.get("point_labels"), dtype=np.int32)
                if (
                    points.ndim != 2
                    or points.shape[1] != 2
                    or not len(points)
                    or not np.isfinite(points).all()
                    or point_labels.shape != (len(points),)
                    or not np.isin(point_labels, [0, 1]).all()
                ):
                    raise ValueError("points must be finite Nx2 with N binary point_labels")
            if box is None and points is None:
                raise ValueError("each prompt needs a box and/or points")
            prompts.append(Prompt(frame, object_id, box, points, point_labels))
        cameras[str(camera)] = tuple(sorted(prompts, key=lambda row: row.frame))
    if not cameras:
        raise ValueError("prompt manifest contains no cameras")
    return PromptSpec(objects, cameras, source)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _process_rss_mb() -> float | None:
    """Current Linux process RSS for experiment provenance."""
    try:
        for line in Path("/proc/self/status").read_text().splitlines():
            if line.startswith("VmRSS:"):
                return round(int(line.split()[1]) / 1024.0, 1)
    except (OSError, ValueError, IndexError):
        pass
    return None


def _trim_process_heap() -> None:
    """Return freed chunk buffers to glibc when it supports ``malloc_trim``."""
    try:
        import ctypes

        trim = ctypes.CDLL(None).malloc_trim
        trim.argtypes = [ctypes.c_size_t]
        trim.restype = ctypes.c_int
        trim(0)
    except (AttributeError, OSError):
        pass


def _verify_checkpoint(path: Path) -> None:
    if not path.is_file():
        raise FileNotFoundError(
            f"SAM 2.1 checkpoint not found: {path}; see viki/perception/README.md"
        )
    actual = _sha256(path)
    if actual != SAM2_CHECKPOINT_SHA256:
        raise ValueError(
            f"SAM 2.1 checkpoint SHA-256 mismatch: expected "
            f"{SAM2_CHECKPOINT_SHA256}, got {actual}"
        )


def _write_chunk_frames(cap: cv2.VideoCapture, directory: Path, limit: int) -> int:
    count = 0
    while count < limit:
        ok, frame = cap.read()
        if not ok:
            break
        path = directory / f"{count:05d}.jpg"
        if not cv2.imwrite(str(path), frame, [cv2.IMWRITE_JPEG_QUALITY, 95]):
            raise RuntimeError(f"could not write temporary SAM frame {path}")
        count += 1
    return count


def _overlay_frame(
    frame_bgr: np.ndarray,
    masks: dict[int, np.ndarray],
    objects: dict[int, str],
) -> np.ndarray:
    colours = {
        "operator": (40, 70, 255),
        "manipulated_object": (60, 220, 60),
        "other_object": (255, 160, 40),
        "other_dynamic": (220, 60, 220),
    }
    canvas = frame_bgr.copy()
    for object_id, mask in masks.items():
        colour = np.asarray(colours[objects[object_id]], dtype=np.float32)
        pixels = canvas[mask].astype(np.float32)
        canvas[mask] = np.clip(0.55 * pixels + 0.45 * colour, 0, 255).astype(np.uint8)
    y = 28
    for object_id, label in objects.items():
        cv2.putText(
            canvas,
            f"{object_id}: {label}",
            (18, y),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.65,
            tuple(int(v) for v in colours[label]),
            2,
            cv2.LINE_AA,
        )
        y += 27
    return canvas


def _save_mask_chunk(
    path: Path,
    *,
    frame_start: int,
    masks: np.ndarray,
    object_ids: list[int],
    object_labels: list[str],
    area_px: np.ndarray,
    mean_positive_logit: np.ndarray,
    overlap_px: np.ndarray,
) -> None:
    packed = np.packbits(masks, axis=-1, bitorder="little")
    with path.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema=np.asarray(MASK_SCHEMA),
            frame_start=np.int32(frame_start),
            frame_count=np.int32(len(masks)),
            height=np.int32(masks.shape[-2]),
            width=np.int32(masks.shape[-1]),
            object_ids=np.asarray(object_ids, np.int32),
            object_labels=np.asarray(object_labels),
            packed_masks=packed,
            area_px=area_px.astype(np.int32),
            mean_positive_logit=mean_positive_logit.astype(np.float32),
            overlap_px=overlap_px.astype(np.int32),
        )


def _load_packed_mask_chunk(path: str | Path) -> dict[str, Any]:
    with np.load(path) as archive:
        if str(archive["schema"]) != MASK_SCHEMA:
            raise ValueError(f"unsupported mask schema in {path}")
        return {
            "frame_start": int(archive["frame_start"]),
            "frame_count": int(archive["frame_count"]),
            "height": int(archive["height"]),
            "width": int(archive["width"]),
            "object_ids": archive["object_ids"].astype(np.int32),
            "object_labels": archive["object_labels"].astype(str),
            "packed_masks": archive["packed_masks"].astype(np.uint8),
            "area_px": archive["area_px"].astype(np.int32),
            "mean_positive_logit": archive["mean_positive_logit"].astype(np.float32),
            "overlap_px": archive["overlap_px"].astype(np.int32),
        }


def unpack_mask_chunk(path: str | Path) -> dict[str, Any]:
    """Load one chunk and expand all bit-packed masks for consumers/tests."""
    data = _load_packed_mask_chunk(path)
    data["masks"] = np.unpackbits(
        data.pop("packed_masks"),
        axis=-1,
        count=data["width"],
        bitorder="little",
    ).astype(bool)
    return data


def segment_episode_sam2(
    ep,
    prompt_path: str | Path,
    *,
    checkpoint: str | Path = "models/sam2/sam2.1_hiera_small.pt",
    chunk_frames: int = 100,
    max_frames: int | None = None,
    render_overlay: bool = True,
    report=None,
) -> Path:
    """Track prompted instances in every listed camera using SAM 2.1."""
    if chunk_frames <= 0:
        raise ValueError("chunk_frames must be positive")
    if max_frames is not None and max_frames <= 0:
        raise ValueError("max_frames must be positive")
    spec = load_prompt_spec(prompt_path)
    checkpoint_path = Path(checkpoint)
    _verify_checkpoint(checkpoint_path)

    try:
        import torch
        from sam2.build_sam import build_sam2_video_predictor
    except ImportError as exc:
        raise RuntimeError(
            "SAM 2 dependencies are unavailable; run this command through "
            "`docker compose run --rm sam2 ...`"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError("SAM 2.1 segmentation requires a visible CUDA GPU")

    started = time.perf_counter()
    predictor = build_sam2_video_predictor(
        SAM2_MODEL_CONFIG,
        str(checkpoint_path),
        device="cuda",
        apply_postprocessing=False,
        vos_optimized=False,
    )
    output = ep.intermediates_dir / "segmentation" / SAM2_MODEL_KEY
    output.mkdir(parents=True, exist_ok=True)
    # A rerun replaces this experiment in place. Remove completion metadata and
    # every artifact derived from the previous masks before writing new chunks,
    # so an interrupted or masks-only run cannot expose a stale 3-D result.
    for stale in (output / "meta.json", output / "object_tracks.npz"):
        if stale.exists():
            stale.unlink()
    semantic_cloud = output / "semantic_cloud"
    if semantic_cloud.is_dir():
        for stale in semantic_cloud.glob("*.npz"):
            stale.unlink()
        semantic_meta = semantic_cloud / "meta.json"
        if semantic_meta.exists():
            semantic_meta.unlink()
    object_ids = sorted(spec.objects)
    object_labels = [spec.objects[value] for value in object_ids]
    camera_meta: dict[str, Any] = {}

    for camera, prompts in spec.cameras.items():
        video_path = ep.raw_dir / f"{camera}.mp4"
        if not video_path.is_file():
            raise FileNotFoundError(f"prompted camera video not found: {video_path}")
        if not all(any(p.frame == 0 and p.object_id == obj for p in prompts) for obj in object_ids):
            raise ValueError(
                f"camera {camera} needs a frame-0 seed for every object; "
                "later correction prompts are optional"
            )
        camera_dir = output / camera
        camera_dir.mkdir(parents=True, exist_ok=True)
        for old in camera_dir.glob("masks_*.npz"):
            old.unlink()
        old_overlay = camera_dir / "overlay.mp4"
        if old_overlay.exists():
            old_overlay.unlink()

        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"could not open {video_path}")
        total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if max_frames is not None:
            total = min(total, int(max_frames))
        fps = float(cap.get(cv2.CAP_PROP_FPS)) or 30.0
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        overlay = None
        if render_overlay:
            overlay = cv2.VideoWriter(
                str(camera_dir / "overlay.mp4"),
                cv2.VideoWriter_fourcc(*"mp4v"),
                fps,
                (width, height),
            )
            if not overlay.isOpened():
                raise RuntimeError(f"could not open overlay writer for {camera}")

        chunks = []
        carry_masks: dict[int, np.ndarray] = {}
        frame_start = 0
        try:
            while frame_start < total:
                requested = min(chunk_frames, total - frame_start)
                with tempfile.TemporaryDirectory(prefix=f"viki-sam2-{camera}-") as tmp:
                    frame_dir = Path(tmp)
                    count = _write_chunk_frames(cap, frame_dir, requested)
                    if count == 0:
                        break
                    torch.cuda.reset_peak_memory_stats()
                    state = predictor.init_state(
                        str(frame_dir),
                        offload_video_to_cpu=True,
                        offload_state_to_cpu=False,
                        async_loading_frames=True,
                    )
                    for object_id, mask in carry_masks.items():
                        predictor.add_new_mask(state, 0, object_id, mask)
                    for prompt in prompts:
                        if not frame_start <= prompt.frame < frame_start + count:
                            continue
                        predictor.add_new_points_or_box(
                            state,
                            frame_idx=prompt.frame - frame_start,
                            obj_id=prompt.object_id,
                            points=prompt.points,
                            labels=prompt.point_labels,
                            box=prompt.box,
                            clear_old_points=True,
                        )

                    masks = np.zeros((count, len(object_ids), height, width), dtype=bool)
                    area = np.zeros((count, len(object_ids)), dtype=np.int32)
                    mean_logit = np.full((count, len(object_ids)), np.nan, dtype=np.float32)
                    overlap = np.zeros(count, dtype=np.int32)
                    with torch.inference_mode(), torch.autocast(
                        device_type="cuda", dtype=torch.bfloat16
                    ):
                        for local_frame, out_ids, logits in predictor.propagate_in_video(
                            state, start_frame_idx=0, max_frame_num_to_track=count
                        ):
                            frame_masks: dict[int, np.ndarray] = {}
                            for row, object_id in enumerate(out_ids):
                                object_id = int(object_id)
                                column = object_ids.index(object_id)
                                values = logits[row].squeeze().float().cpu().numpy()
                                mask = values > 0.0
                                masks[local_frame, column] = mask
                                area[local_frame, column] = int(mask.sum())
                                if mask.any():
                                    mean_logit[local_frame, column] = float(values[mask].mean())
                                frame_masks[object_id] = mask
                            overlap[local_frame] = int(
                                (masks[local_frame].sum(axis=0) > 1).sum()
                            )
                            if overlay is not None:
                                source = cv2.imread(str(frame_dir / f"{local_frame:05d}.jpg"))
                                overlay.write(_overlay_frame(source, frame_masks, spec.objects))

                    carry_masks = {
                        object_id: masks[-1, column].copy()
                        for column, object_id in enumerate(object_ids)
                    }
                    chunk_path = camera_dir / f"masks_{frame_start:06d}_{frame_start + count - 1:06d}.npz"
                    _save_mask_chunk(
                        chunk_path,
                        frame_start=frame_start,
                        masks=masks,
                        object_ids=object_ids,
                        object_labels=object_labels,
                        area_px=area,
                        mean_positive_logit=mean_logit,
                        overlap_px=overlap,
                    )
                    chunk_meta = {
                        "file": chunk_path.name,
                        "frame_start": frame_start,
                        "frame_count": count,
                        "cuda_peak_allocated_mb": round(
                            torch.cuda.max_memory_allocated() / (1024 ** 2), 1
                        ),
                    }
                    chunks.append(chunk_meta)
                    images = state.get("images")
                    thread = getattr(images, "thread", None)
                    if thread is not None:
                        thread.join()
                    predictor.reset_state(state)
                    # ``images`` owns the CPU-offloaded 1024px frame tensor. If
                    # this local reference survives into the next chunk, two
                    # chunks coexist and a three-object run exceeds the 11 GiB
                    # container ceiling.
                    del images, thread, state, masks
                    gc.collect()
                    torch.cuda.empty_cache()
                    _trim_process_heap()
                    chunk_meta["rss_mb_after_cleanup"] = _process_rss_mb()
                    frame_start += count
                    if report is not None:
                        report(camera=camera, frame=frame_start, total=total)
        finally:
            cap.release()
            if overlay is not None:
                overlay.release()

        camera_meta[camera] = {
            "video": video_path.name,
            "frames": frame_start,
            "width": width,
            "height": height,
            "fps": fps,
            "chunks": chunks,
        }

    meta = {
        "schema": MASK_SCHEMA,
        "model": SAM2_MODEL_KEY,
        "model_config": SAM2_MODEL_CONFIG,
        "sam2_commit": SAM2_UPSTREAM_COMMIT,
        "checkpoint": checkpoint_path.name,
        "checkpoint_sha256": SAM2_CHECKPOINT_SHA256,
        "elapsed_sec": time.perf_counter() - started,
        "chunk_frames": chunk_frames,
        "binary_threshold": 0.0,
        "confidence_note": "mean_positive_logit is diagnostic, not calibrated probability",
        "objects": [{"id": key, "label": spec.objects[key]} for key in object_ids],
        "prompts": spec.source,
        "cameras": camera_meta,
    }
    (output / "meta.json").write_text(json.dumps(meta, indent=2))
    del predictor
    gc.collect()
    torch.cuda.empty_cache()
    _trim_process_heap()
    logger.info("SAM 2.1 masks for %s → %s", ep.id, output)
    return output


class MaskArchive:
    """Random frame access over chunked, compressed SAM mask artifacts."""

    def __init__(self, root: str | Path):
        self.root = Path(root)
        self.meta = json.loads((self.root / "meta.json").read_text())
        if self.meta.get("schema") != MASK_SCHEMA:
            raise ValueError(f"unsupported segmentation schema under {root}")
        self._cache: dict[str, tuple[str, dict[str, Any]]] = {}

    def frame(self, camera: str, frame_index: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        camera_meta = self.meta["cameras"].get(camera)
        if camera_meta is None:
            raise KeyError(f"segmentation has no camera {camera!r}")
        for chunk in camera_meta["chunks"]:
            start = int(chunk["frame_start"])
            count = int(chunk["frame_count"])
            if start <= frame_index < start + count:
                key = (camera, chunk["file"])
                cached = self._cache.get(camera)
                if cached is None or cached[0] != chunk["file"]:
                    cached = (
                        chunk["file"],
                        _load_packed_mask_chunk(self.root / camera / chunk["file"]),
                    )
                    self._cache[camera] = cached
                data = cached[1]
                local = frame_index - start
                masks = np.unpackbits(
                    data["packed_masks"][local],
                    axis=-1,
                    count=data["width"],
                    bitorder="little",
                ).astype(bool)
                return (
                    data["object_ids"],
                    data["object_labels"],
                    masks,
                )
        raise IndexError(f"frame {frame_index} is outside {camera!r} segmentation")


def lift_segmentation_to_3d(
    ep,
    segmentation_dir: str | Path,
    *,
    stride: int = 2,
    voxel_m: float = 0.004,
    report=None,
) -> Path:
    """Lift binary instance masks through recorded depth into world XYZ.

    This first artifact keeps only mask-supported points.  Independent camera
    observations are concatenated then voxel-deduplicated per instance; it does
    not yet claim calibrated semantic probabilities or free-space evidence.
    """
    if stride <= 0 or voxel_m < 0:
        raise ValueError("stride must be positive and voxel_m non-negative")
    archive = MaskArchive(segmentation_dir)
    intrinsics = _read_json(ep.raw_dir / "intrinsics.json")
    extrinsics = _read_json(ep.raw_dir / "extrinsics.json")
    episode_meta = _read_json(ep.meta_path)
    bbox = list(getattr(config, "CLOUD_WORKSPACE_BBOX", []) or [])
    bbox = _bbox_to_frame(
        bbox,
        _read_json(ep.raw_dir / "world_anchor.json").get("T_world_display"),
    )

    from viki.perception.k4a_offline import K4ACalibration
    from viki.perception.rs_offline import RealSenseCalibration

    cameras = []
    for camera, camera_meta in archive.meta["cameras"].items():
        entry = extrinsics.get(camera)
        if entry is None:
            raise ValueError(f"camera {camera} has no recorded extrinsics")
        transform = CalibrationExtrinsics(
            rvec=np.asarray(entry["rvec"], np.float64),
            tvec=np.asarray(entry["tvec"], np.float64),
        ).transform_matrix
        calibration = (
            K4ACalibration.from_episode(ep.raw_dir, camera, episode_meta)
            or RealSenseCalibration.from_episode(ep.raw_dir, camera, episode_meta)
        )
        cameras.append({
            "id": camera,
            "cap": cv2.VideoCapture(str(ep.raw_dir / f"{camera}.mp4")),
            "depth": ep.raw_dir / f"{camera}_depth",
            "K": _color_K(intrinsics.get(camera, {})),
            "K_depth": _depth_K(intrinsics.get(camera, {})),
            "cal": calibration,
            "T": transform,
            "frames": int(camera_meta["frames"]),
        })

    total = min((camera["frames"] for camera in cameras), default=0)
    output = Path(segmentation_dir) / "semantic_cloud"
    output.mkdir(parents=True, exist_ok=True)
    for old in output.glob("*.npz"):
        old.unlink()
    output_meta = output / "meta.json"
    if output_meta.exists():
        output_meta.unlink()
    tracks_path = Path(segmentation_dir) / "object_tracks.npz"
    if tracks_path.exists():
        tracks_path.unlink()
    object_ids = np.asarray([item["id"] for item in archive.meta["objects"]], np.int32)
    labels_by_id = {int(item["id"]): item["label"] for item in archive.meta["objects"]}
    track_count = np.zeros((total, len(object_ids)), np.int32)
    track_centroid = np.full((total, len(object_ids), 3), np.nan, np.float32)
    track_dimensions = np.full((total, len(object_ids), 3), np.nan, np.float32)

    try:
        for frame_index in range(total):
            xyz_parts, rgb_parts, id_parts, code_parts, camera_parts = [], [], [], [], []
            for camera_index, camera in enumerate(cameras):
                ok, bgr = camera["cap"].read()
                depth_path = camera["depth"] / f"{frame_index:06d}.npy"
                if not ok or not depth_path.is_file():
                    continue
                depth = np.load(depth_path)
                xyz, rgb, uv = _camera_samples(
                    bgr,
                    depth,
                    stride,
                    camera["K"],
                    camera["cal"],
                    camera["T"],
                    K_depth=camera["K_depth"],
                    edge_filter=bool(getattr(config, "CLOUD_EDGE_FILTER", True)),
                    edge_radius_rad=float(
                        getattr(config, "CLOUD_EDGE_RADIUS_RAD", 0.004)
                    ),
                    edge_jump_mm=float(
                        getattr(config, "CLOUD_EDGE_JUMP_MM", 30.0)
                    ),
                    edge_jump_relative=float(
                        getattr(config, "CLOUD_EDGE_JUMP_RELATIVE", 0.02)
                    ),
                )
                ids, labels, masks = archive.frame(camera["id"], frame_index)
                membership = masks[:, uv[:, 1], uv[:, 0]]
                support = membership.sum(axis=0)
                keep = (support > 0) & _bbox_mask(xyz, bbox)
                if not keep.any():
                    continue
                winner = np.argmax(membership[:, keep], axis=0)
                selected_ids = ids[winner].astype(np.int32)
                selected_labels = labels[winner]
                ambiguous = support[keep] > 1
                selected_ids[ambiguous] = 0
                codes = np.asarray(
                    [LABEL_CODES[str(label)] for label in selected_labels], np.uint8
                )
                codes[ambiguous] = LABEL_CODES["ambiguous_contact"]
                xyz_parts.append(xyz[keep])
                rgb_parts.append(rgb[keep])
                id_parts.append(selected_ids)
                code_parts.append(codes)
                camera_parts.append(np.full(keep.sum(), camera_index, np.uint8))

            if xyz_parts:
                xyz = np.concatenate(xyz_parts)
                rgb = np.concatenate(rgb_parts)
                instance = np.concatenate(id_parts)
                label_code = np.concatenate(code_parts)
                source_camera = np.concatenate(camera_parts)
                keep_indices = []
                for instance_id in np.unique(instance):
                    rows = np.flatnonzero(instance == instance_id)
                    if not len(rows):
                        continue
                    local_indices = _voxel_downsample_indices(xyz[rows], voxel_m)
                    keep_indices.extend(rows[local_indices])
                keep_indices = np.asarray(sorted(keep_indices), np.int64)
                xyz, rgb = xyz[keep_indices], rgb[keep_indices]
                instance = instance[keep_indices]
                label_code = label_code[keep_indices]
                source_camera = source_camera[keep_indices]
            else:
                xyz = np.empty((0, 3), np.float32)
                rgb = np.empty((0, 3), np.uint8)
                instance = np.empty(0, np.int32)
                label_code = np.empty(0, np.uint8)
                source_camera = np.empty(0, np.uint8)

            for column, object_id in enumerate(object_ids):
                points = xyz[instance == object_id]
                track_count[frame_index, column] = len(points)
                if len(points):
                    track_centroid[frame_index, column] = np.median(points, axis=0)
                    low, high = np.percentile(points, [2, 98], axis=0)
                    track_dimensions[frame_index, column] = high - low
            with (output / f"{frame_index:06d}.npz").open("wb") as stream:
                np.savez_compressed(
                    stream,
                    schema=np.asarray(SEMANTIC_CLOUD_SCHEMA),
                    xyz=xyz.astype(np.float32),
                    rgb=rgb.astype(np.uint8),
                    instance_id=instance.astype(np.int32),
                    label_code=label_code.astype(np.uint8),
                    source_camera=source_camera.astype(np.uint8),
                )
            if report is not None and (frame_index + 1) % 15 == 0:
                report(frame=frame_index + 1, total=total)
    finally:
        for camera in cameras:
            camera["cap"].release()

    with tracks_path.open("wb") as stream:
        np.savez_compressed(
            stream,
            schema=np.asarray(SEMANTIC_CLOUD_SCHEMA),
            object_ids=object_ids,
            object_labels=np.asarray([labels_by_id[int(value)] for value in object_ids]),
            point_count=track_count,
            centroid_world=track_centroid,
            visible_dimensions_world=track_dimensions,
            voxel_m=np.float32(voxel_m),
            stride=np.int32(stride),
        )
    (output / "meta.json").write_text(json.dumps({
        "schema": SEMANTIC_CLOUD_SCHEMA,
        "frames": total,
        "stride": stride,
        "voxel_m": voxel_m,
        "edge_filter": bool(getattr(config, "CLOUD_EDGE_FILTER", True)),
        "edge_radius_rad": float(
            getattr(config, "CLOUD_EDGE_RADIUS_RAD", 0.004)
        ),
        "edge_jump_mm": float(getattr(config, "CLOUD_EDGE_JUMP_MM", 30.0)),
        "edge_jump_relative": float(
            getattr(config, "CLOUD_EDGE_JUMP_RELATIVE", 0.02)
        ),
        "workspace_bbox_rig": bbox,
        "cameras": [camera["id"] for camera in cameras],
        "labels": LABEL_CODES,
        "safety_note": "mask-supported points only; missing points are unknown, not free space",
    }, indent=2))
    logger.info("semantic mask lift for %s → %s", ep.id, output)
    return output
