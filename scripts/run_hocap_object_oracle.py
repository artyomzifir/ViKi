"""HO-Cap object tracking with published masks as oracle input."""

from __future__ import annotations

import argparse
import io
import json
import shutil
import tarfile
from pathlib import Path

import cv2
import numpy as np
import yaml

from viki.benchmark.hocap_objects import evaluate_object_models, import_object_ground_truth
from viki.contracts import CalibrationExtrinsics
from viki.perception.cloud import _voxel_downsample_indices
from viki.perception.object_model import build_object_models
from viki.perception.segmentation import LABEL_CODES, SEMANTIC_CLOUD_SCHEMA


def build_oracle_cloud(
    source_archive: Path, episode_dir: Path, output: Path, *,
    object_names: tuple[str, ...], frames: int,
    stride: int = 4, voxel_m: float = 0.004,
) -> dict[int, str]:
    if stride <= 0 or voxel_m <= 0:
        raise ValueError("stride and voxel size must be positive")
    if len(object_names) != 4:
        raise ValueError("HO-Cap oracle masks require four object IDs")
    episode_meta = json.loads((episode_dir / "meta.json").read_text())
    cameras = tuple(episode_meta["cameras"])
    if len(cameras) != 8:
        raise ValueError("object oracle requires all eight RealSense cameras")
    intrinsics = json.loads((episode_dir / "raw/intrinsics.json").read_text())
    extrinsics = json.loads((episode_dir / "raw/extrinsics.json").read_text())
    calibration = {}
    for camera in cameras:
        color = intrinsics[camera]["color"]
        matrix = np.array([
            [color["fx"], 0, color["cx"]],
            [0, color["fy"], color["cy"]],
            [0, 0, 1],
        ], np.float64)
        world = CalibrationExtrinsics(
            rvec=np.asarray(extrinsics[camera]["rvec"], np.float64),
            tvec=np.asarray(extrinsics[camera]["tvec"], np.float64),
        ).transform_matrix
        calibration[camera] = (matrix, np.asarray(color["dist_coeffs"], np.float64), world)
    cloud = output / "semantic_cloud"
    cloud.mkdir(parents=True, exist_ok=True)
    for stale in cloud.glob("*.npz"):
        stale.unlink()
    object_map = {index + 1: name for index, name in enumerate(object_names)}
    objects = [{"id": index, "label": "other_dynamic"} for index in object_map]
    objects.append({"id": 99, "label": "operator"})
    (output / "meta.json").write_text(json.dumps({
        "schema": "viki_sam2_masks_v1", "source": "HO-Cap published segmentation oracle",
        "objects": objects, "object_name_by_id": object_map,
    }, indent=2))
    with tarfile.open(source_archive) as archive:
        for frame_index in range(frames):
            xyz_parts = []
            instance_parts = []
            camera_parts = []
            for camera_index, camera in enumerate(cameras):
                prefix = f"{camera}-{frame_index:06d}"
                depth_member = archive.extractfile(f"{prefix}.depth.png")
                label_member = archive.extractfile(f"{prefix}.label.npy")
                if depth_member is None or label_member is None:
                    raise ValueError(f"missing depth or label: {prefix}")
                depth = cv2.imdecode(
                    np.frombuffer(depth_member.read(), dtype=np.uint8), cv2.IMREAD_UNCHANGED
                )
                if depth is None or depth.dtype != np.uint16:
                    raise ValueError(f"missing uint16 depth: {camera} frame {frame_index}")
                with np.load(io.BytesIO(label_member.read()), allow_pickle=False) as label:
                    mask = np.asarray(label["seg_mask"], dtype=np.uint8)
                if mask.shape != depth.shape:
                    raise ValueError("segmentation and depth image sizes differ")
                sampled = mask[::stride, ::stride]
                selected = ((sampled >= 1) & (sampled <= len(object_names))) | (sampled == 5) | (sampled == 6)
                selected &= depth[::stride, ::stride] > 0
                row, column = np.nonzero(selected)
                if not len(row):
                    continue
                codes = sampled[row, column].astype(np.int32)
                codes[codes >= 5] = 99
                pixels = np.column_stack((column * stride, row * stride)).astype(np.float64)
                matrix, distortion, world = calibration[camera]
                rays = cv2.undistortPoints(pixels[:, None], matrix, distortion)[:, 0]
                z = depth[row * stride, column * stride].astype(np.float64) / 1000.0
                camera_points = np.column_stack((rays[:, 0] * z, rays[:, 1] * z, z))
                world_points = camera_points @ world[:3, :3].T + world[:3, 3]
                xyz_parts.append(world_points.astype(np.float32))
                instance_parts.append(codes)
                camera_parts.append(np.full(len(codes), camera_index, np.uint8))
            if xyz_parts:
                xyz = np.concatenate(xyz_parts)
                instance = np.concatenate(instance_parts)
                source_camera = np.concatenate(camera_parts)
                keep = np.concatenate([
                    rows[_voxel_downsample_indices(xyz[rows], voxel_m)]
                    for instance_id in np.unique(instance)
                    for rows in [np.flatnonzero(instance == instance_id)]
                ])
                xyz, instance, source_camera = xyz[keep], instance[keep], source_camera[keep]
            else:
                xyz = np.empty((0, 3), np.float32)
                instance = np.empty(0, np.int32)
                source_camera = np.empty(0, np.uint8)
            label_code = np.where(instance == 99, LABEL_CODES["operator"],
                                  LABEL_CODES["other_dynamic"]).astype(np.uint8)
            with (cloud / f"{frame_index:06d}.npz").open("wb") as stream:
                np.savez_compressed(
                    stream, schema=np.asarray(SEMANTIC_CLOUD_SCHEMA), xyz=xyz,
                    rgb=np.zeros((len(xyz), 3), np.uint8), instance_id=instance,
                    label_code=label_code, source_camera=source_camera,
                )
            if (frame_index + 1) % 30 == 0:
                print(f"oracle cloud {frame_index + 1}/{frames}", flush=True)
    (cloud / "meta.json").write_text(json.dumps({
        "schema": SEMANTIC_CLOUD_SCHEMA, "frames": frames,
        "source": "HO-Cap published segmentation oracle", "stride": stride,
        "voxel_m": voxel_m, "cameras": list(cameras),
    }, indent=2))
    return object_map


def run_sequence(
    archives: Path, episode: Path, output: Path, *,
    frames: int | None = None, stride: int = 4, keep_cloud: bool = False,
) -> dict:
    episode_meta = json.loads((episode / "meta.json").read_text())
    sequence_id = f"{episode_meta['hocap_subject']}-{episode_meta['hocap_sequence']}"
    source_frames = int(episode_meta["num_frames"])
    if frames is None:
        frames = source_frames
    if frames <= 0 or frames > source_frames:
        raise ValueError("frame count outside imported episode")
    with tarfile.open(archives / "sequence_labels.tar") as source:
        member = source.extractfile(f"{sequence_id}.meta.yaml")
        if member is None:
            raise ValueError(f"missing metadata for {sequence_id}")
        object_names = tuple(str(value) for value in yaml.safe_load(member.read())["object_ids"])
    output.mkdir(parents=True, exist_ok=True)
    (output / "evaluation.json").unlink(missing_ok=True)
    try:
        mapping = build_oracle_cloud(
            archives / f"{sequence_id}.tar", episode, output,
            object_names=object_names, frames=frames, stride=stride,
        )
        model = build_object_models(output)
        gt = import_object_ground_truth(archives, sequence_id, episode)
        if frames == source_frames:
            score = evaluate_object_models(gt, model, mapping)
        else:
            with np.load(gt, allow_pickle=False) as reference:
                shortened = output / "object_gt_subset.npz"
                with shortened.open("wb") as stream:
                    np.savez_compressed(stream, **{
                        key: value[:frames] if value.ndim and len(value) == source_frames else value
                        for key, value in ((name, reference[name]) for name in reference.files)
                    })
            score = evaluate_object_models(shortened, model, mapping)
            shortened.unlink()
        (output / "evaluation.json").write_text(json.dumps(score, indent=2) + "\n")
        print(json.dumps(score, indent=2), flush=True)
        return score
    finally:
        if not keep_cloud:
            shutil.rmtree(output / "semantic_cloud", ignore_errors=True)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", type=Path, required=True)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--frames", type=int)
    parser.add_argument("--stride", type=int, default=4)
    parser.add_argument("--keep-cloud", action="store_true")
    args = parser.parse_args()
    run_sequence(args.archives, args.episode, args.out, frames=args.frames,
                 stride=args.stride, keep_cloud=args.keep_cloud)


if __name__ == "__main__":
    main()
