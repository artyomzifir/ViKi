"""Build a compact, static HO-Cap predicted-vs-reference pose review."""

from __future__ import annotations

import argparse
import io
import json
import tarfile
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw


def pose(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = translation
    return transform


def build_review(
    episode: Path, archive: Path, output: Path, object_name: str,
    cameras: tuple[str, ...],
) -> dict:
    profile = episode / "intermediates/segmentation/hocap-oracle-v1"
    mapping = json.loads((profile / "meta.json").read_text())["object_name_by_id"]
    with np.load(profile / "object_models.npz", allow_pickle=False) as model:
        model_id = next(int(key) for key, name in mapping.items() if name == object_name)
        model_column = list(model["object_ids"]).index(model_id)
        offsets = model["model_offsets"]
        points = np.asarray(model["model_xyz_object"][
            offsets[model_column]:offsets[model_column + 1]
        ], dtype=np.float64)
        rotations = np.asarray(model["rotation_world_object"][:, model_column], dtype=np.float64)
        translations = np.asarray(model["translation_world"][:, model_column], dtype=np.float64)
        confidence = np.asarray(model["track_confidence"][:, model_column], dtype=np.float64)
    with np.load(episode / "gt/object_poses_world.npz", allow_pickle=False) as reference:
        reference_column = list(reference["object_ids"]).index(object_name)
        truth = np.asarray(reference["poses_world"][:, reference_column], dtype=np.float64)
        reference_valid = np.asarray(reference["valid"][:, reference_column], dtype=bool)

    tracked = (
        reference_valid & np.isfinite(rotations).all(axis=(1, 2))
        & np.isfinite(translations).all(axis=1)
        & np.isfinite(confidence) & (confidence >= 0.35)
    )
    if not tracked.any():
        raise ValueError(f"no scored poses for {object_name}")
    anchor = int(np.flatnonzero(tracked)[0])
    alignment = np.linalg.inv(pose(rotations[anchor], translations[anchor])) @ truth[anchor]
    points = points[np.linspace(0, len(points) - 1, min(500, len(points)), dtype=int)]
    points_h = np.column_stack((points, np.ones(len(points))))
    points_gt_local = (np.linalg.inv(alignment) @ points_h.T).T[:, :3]

    predicted = []
    ground_truth = []
    angle_errors = []
    position_errors = []
    for frame in range(len(truth)):
        ground_truth.append(truth[frame].tolist() if reference_valid[frame] else None)
        if not tracked[frame]:
            predicted.append(None)
            angle_errors.append(None)
            position_errors.append(None)
            continue
        aligned = pose(rotations[frame], translations[frame]) @ alignment
        predicted.append(aligned.tolist())
        delta = truth[frame, :3, :3].T @ aligned[:3, :3]
        cosine = np.clip((np.trace(delta) - 1.0) / 2.0, -1.0, 1.0)
        angle_errors.append(float(np.degrees(np.arccos(cosine))))
        position_errors.append(float(np.linalg.norm(
            aligned[:3, 3] - truth[frame, :3, 3]
        ) * 1000))

    peak = int(np.nanargmax([value if value is not None else np.nan for value in angle_errors]))
    frame_ids = sorted(set(range(0, len(truth), 30))
                       | set(range(max(0, peak - 12), min(len(truth), peak + 13)))
                       | {anchor, peak, len(truth) - 1})
    output.mkdir(parents=True, exist_ok=True)
    image_dir = output / "frames"
    image_dir.mkdir(exist_ok=True)
    with tarfile.open(archive) as source:
        for frame in frame_ids:
            for camera in cameras:
                prefix = f"{camera}-{frame:06d}"
                image_file = source.extractfile(f"{prefix}.color.jpg")
                label_file = source.extractfile(f"{prefix}.label.npy")
                if image_file is None or label_file is None:
                    raise ValueError(f"missing RGB or mask for {prefix}")
                with Image.open(io.BytesIO(image_file.read())) as opened:
                    image = opened.convert("RGB")
                with np.load(io.BytesIO(label_file.read()), allow_pickle=False) as label:
                    mask = np.asarray(label["seg_mask"] == model_id)
                rows, columns = np.nonzero(mask)
                if len(rows):
                    ImageDraw.Draw(image).rectangle(
                        (int(columns.min()), int(rows.min()),
                         int(columns.max()), int(rows.max())),
                        outline="#ff4b3a", width=3,
                    )
                image.save(image_dir / f"{camera}-{frame:06d}.jpg", quality=82)

    report = {
        "episode": episode.name,
        "object": object_name,
        "camera_ids": cameras,
        "image_frames": frame_ids,
        "frame_count": len(truth),
        "anchor_frame": anchor,
        "peak_frame": peak,
        "mean_rotation_deg": float(np.mean([v for v in angle_errors if v is not None])),
        "points_gt_local": points_gt_local.tolist(),
        "gt_pose": ground_truth,
        "predicted_pose_aligned": predicted,
        "rotation_error_deg": angle_errors,
        "translation_error_mm": position_errors,
        "confidence": confidence.tolist(),
        "note": "Published masks; one GT-assisted object-local alignment at the first confident frame. This is relative tracking drift, not absolute pose error.",
    }
    (output / "review.json").write_text(json.dumps(report, separators=(",", ":")))
    return {"peak_frame": peak, "mean_rotation_deg": report["mean_rotation_deg"],
            "images": len(frame_ids) * len(cameras), "output": str(output)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--episode", type=Path, required=True)
    parser.add_argument("--archive", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--object", required=True)
    parser.add_argument("--cameras", nargs="+", required=True)
    args = parser.parse_args()
    print(json.dumps(build_review(args.episode, args.archive, args.output,
                                  args.object, tuple(args.cameras)), indent=2))


if __name__ == "__main__":
    main()
