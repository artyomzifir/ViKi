#!/usr/bin/env python3
"""
Build a visual calibration check from a pair snapshot.

The script lifts aligned depth from both Azure Kinect color cameras into 3D,
transforms the two point clouds into the static board/world frame, and writes a
colored fused PLY plus simple projection plots.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import cv2
import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402


DEFAULT_SNAPSHOTS = Path("data/datasets/rig_20260623_static_world_v2/snapshots")
DEFAULT_CALIBRATION = Path("data/calibration/final/rig_20260623_static_world_v2/static_world.json")
DEFAULT_OUT = Path("data/calibration/visual_check/static_world_v2")
DEFAULT_CAMERA0 = "kinect_0"
DEFAULT_CAMERA1 = "kinect_1"
CAMERA_DEBUG_COLORS = (
    np.asarray([31, 119, 180], dtype=np.uint8),
    np.asarray([255, 127, 14], dtype=np.uint8),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fuse two aligned-depth Kinect snapshots into the calibrated board/world frame."
    )
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument(
        "--snapshot-id",
        default="latest",
        help="Pair snapshot id to process, or 'latest' for the newest complete pair.",
    )
    parser.add_argument("--camera0", default=DEFAULT_CAMERA0)
    parser.add_argument("--camera1", default=DEFAULT_CAMERA1)
    parser.add_argument("--stride", type=int, default=4, help="Pixel stride for point cloud sampling.")
    parser.add_argument("--min-depth-m", type=float, default=0.4)
    parser.add_argument("--max-depth-m", type=float, default=4.0)
    parser.add_argument("--max-points-per-camera", type=int, default=120_000)
    parser.add_argument("--plot-max-points-per-camera", type=int, default=30_000)
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, data: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")


def iter_complete_pairs(snapshots_dir: Path, camera_ids: list[str]) -> list[Path]:
    if not snapshots_dir.exists():
        return []

    pairs: list[Path] = []
    for path in sorted(snapshots_dir.iterdir()):
        if not path.is_dir() or not path.name.startswith("pair_"):
            continue
        complete = True
        for camera_id in camera_ids:
            camera_dir = path / camera_id
            if not (
                (camera_dir / "color.jpg").exists()
                and (camera_dir / "aligned_depth.npy").exists()
                and (camera_dir / "metadata.json").exists()
            ):
                complete = False
                break
        if complete:
            pairs.append(path)
    return pairs


def resolve_snapshot_dir(snapshots_dir: Path, snapshot_id: str, camera_ids: list[str]) -> Path:
    if snapshot_id == "latest":
        pairs = iter_complete_pairs(snapshots_dir, camera_ids)
        if not pairs:
            raise FileNotFoundError(f"No complete pair snapshots found in {snapshots_dir}")
        return pairs[-1]

    path = snapshots_dir / snapshot_id
    if not path.exists():
        raise FileNotFoundError(f"Snapshot {snapshot_id!r} not found in {snapshots_dir}")
    return path


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    rotation = transform[:3, :3]
    translation = transform[:3, 3]
    return points @ rotation.T + translation


def load_world_transforms(
    calibration_path: Path,
    camera_ids: list[str],
) -> tuple[dict[str, np.ndarray], dict[str, str], dict[str, Any]]:
    calibration = read_json(calibration_path)
    final = calibration.get("final", calibration)
    transforms: dict[str, np.ndarray] = {}
    transform_keys: dict[str, str] = {}

    for camera_id in camera_ids:
        key = f"T_world_from_{camera_id}_color"
        if key not in final:
            compact_camera_id = camera_id.replace("kinect_", "kinect")
            compact_key = f"T_world_from_{compact_camera_id}_color"
            if compact_key in final:
                key = compact_key
        if key not in final:
            available = ", ".join(sorted(k for k in final if k.startswith("T_world_from_")))
            raise KeyError(f"{calibration_path} does not contain {key}. Available: {available}")
        transform = np.asarray(final[key], dtype=np.float64)
        if transform.shape != (4, 4):
            raise ValueError(f"{key} has shape {transform.shape}, expected (4, 4)")
        transforms[camera_id] = transform
        transform_keys[camera_id] = key

    return transforms, transform_keys, calibration


def load_camera_intrinsics(metadata_path: Path) -> tuple[np.ndarray, np.ndarray, dict[str, Any]]:
    metadata = read_json(metadata_path)
    intrinsics = metadata.get("color_intrinsics")
    if not intrinsics:
        raise ValueError(f"{metadata_path} has no color_intrinsics")

    camera_matrix = np.asarray(intrinsics.get("camera_matrix"), dtype=np.float64)
    dist_coeffs = np.asarray(intrinsics.get("dist_coeffs", []), dtype=np.float64).reshape(-1, 1)
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"{metadata_path} has invalid camera_matrix shape {camera_matrix.shape}")

    return camera_matrix, dist_coeffs, metadata


def load_camera_cloud(
    snapshot_dir: Path,
    camera_id: str,
    world_from_camera: np.ndarray,
    stride: int,
    min_depth_m: float,
    max_depth_m: float,
    max_points: int,
    rng: np.random.Generator,
) -> dict[str, Any]:
    camera_dir = snapshot_dir / camera_id
    color = cv2.imread(str(camera_dir / "color.jpg"), cv2.IMREAD_COLOR)
    if color is None:
        raise ValueError(f"Could not read {camera_dir / 'color.jpg'}")

    depth = np.load(camera_dir / "aligned_depth.npy")
    if depth.ndim != 2:
        raise ValueError(f"{camera_dir / 'aligned_depth.npy'} has shape {depth.shape}, expected HxW")
    if color.shape[:2] != depth.shape[:2]:
        raise ValueError(
            f"{camera_id}: color shape {color.shape[:2]} does not match aligned depth shape {depth.shape[:2]}"
        )

    camera_matrix, dist_coeffs, metadata = load_camera_intrinsics(camera_dir / "metadata.json")

    step = max(1, int(stride))
    ys, xs = np.mgrid[0:depth.shape[0]:step, 0:depth.shape[1]:step]
    xs_flat = xs.reshape(-1)
    ys_flat = ys.reshape(-1)
    depth_mm = depth[ys_flat, xs_flat].astype(np.float64)
    depth_m = depth_mm / 1000.0

    valid = np.isfinite(depth_m) & (depth_m >= min_depth_m) & (depth_m <= max_depth_m)
    xs_flat = xs_flat[valid]
    ys_flat = ys_flat[valid]
    depth_m = depth_m[valid]

    if max_points > 0 and depth_m.size > max_points:
        keep = rng.choice(depth_m.size, size=max_points, replace=False)
        keep.sort()
        xs_flat = xs_flat[keep]
        ys_flat = ys_flat[keep]
        depth_m = depth_m[keep]

    pixels = np.stack([xs_flat, ys_flat], axis=1).astype(np.float64).reshape(-1, 1, 2)
    normalized = cv2.undistortPoints(pixels, camera_matrix, dist_coeffs).reshape(-1, 2)
    points_camera = np.column_stack(
        [
            normalized[:, 0] * depth_m,
            normalized[:, 1] * depth_m,
            depth_m,
        ]
    )
    points_world = transform_points(world_from_camera, points_camera)
    colors_rgb = color[ys_flat, xs_flat, ::-1].copy()

    return {
        "camera_id": camera_id,
        "metadata": metadata,
        "depth_file": "aligned_depth.npy",
        "depth_frame": "color_camera_frame",
        "intrinsics_source": "metadata.color_intrinsics",
        "color_shape": list(color.shape),
        "aligned_depth_shape": list(depth.shape),
        "points_world": points_world,
        "colors_rgb": colors_rgb,
        "valid_points": int(points_world.shape[0]),
    }


def write_ply(path: Path, points: np.ndarray, colors_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if points.shape[0] != colors_rgb.shape[0]:
        raise ValueError("points/colors length mismatch")

    with path.open("w", encoding="ascii") as file:
        file.write("ply\n")
        file.write("format ascii 1.0\n")
        file.write(f"element vertex {points.shape[0]}\n")
        file.write("property float x\n")
        file.write("property float y\n")
        file.write("property float z\n")
        file.write("property uchar red\n")
        file.write("property uchar green\n")
        file.write("property uchar blue\n")
        file.write("end_header\n")

        colors = np.clip(colors_rgb, 0, 255).astype(np.uint8)
        for point, color in zip(points, colors):
            file.write(
                f"{point[0]:.6f} {point[1]:.6f} {point[2]:.6f} "
                f"{int(color[0])} {int(color[1])} {int(color[2])}\n"
            )


def make_projection_plot(
    path: Path,
    clouds: list[dict[str, Any]],
    calibration: dict[str, Any],
    max_points_per_camera: int,
    rng: np.random.Generator,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    plot_colors = {
        clouds[0]["camera_id"]: "#1f77b4",
        clouds[1]["camera_id"]: "#ff7f0e" if len(clouds) > 1 else "#1f77b4",
    }

    fig, axes = plt.subplots(1, 3, figsize=(16, 5), constrained_layout=True)
    views = [
        ("world XY / board plane", 0, 1, "X, m", "Y, m"),
        ("world XZ / side", 0, 2, "X, m", "Z, m"),
        ("world YZ / side", 1, 2, "Y, m", "Z, m"),
    ]
    board = calibration.get("board", {})
    board_width = float(board.get("squares_x", 0) or 0) * float(board.get("square_length_m", 0) or 0)
    board_height = float(board.get("squares_y", 0) or 0) * float(board.get("square_length_m", 0) or 0)
    board_outline = None
    if board_width > 0 and board_height > 0:
        board_outline = np.asarray(
            [
                [0.0, 0.0, 0.0],
                [board_width, 0.0, 0.0],
                [board_width, board_height, 0.0],
                [0.0, board_height, 0.0],
                [0.0, 0.0, 0.0],
            ],
            dtype=np.float64,
        )

    for ax, (title, x_idx, y_idx, x_label, y_label) in zip(axes, views):
        for cloud in clouds:
            points = cloud["points_world"]
            if max_points_per_camera > 0 and points.shape[0] > max_points_per_camera:
                keep = rng.choice(points.shape[0], size=max_points_per_camera, replace=False)
                points = points[keep]
            ax.scatter(
                points[:, x_idx],
                points[:, y_idx],
                s=0.4,
                alpha=0.45,
                c=plot_colors[cloud["camera_id"]],
                label=cloud["camera_id"],
                rasterized=True,
            )
        if board_outline is not None:
            ax.plot(
                board_outline[:, x_idx],
                board_outline[:, y_idx],
                color="black",
                linewidth=1.2,
                alpha=0.85,
                label="board",
            )
        ax.axhline(0.0, color="black", linewidth=0.6, alpha=0.35)
        ax.axvline(0.0, color="black", linewidth=0.6, alpha=0.35)
        ax.set_title(title)
        ax.set_xlabel(x_label)
        ax.set_ylabel(y_label)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, linewidth=0.4, alpha=0.25)
        ax.legend(markerscale=8)

    baseline = calibration.get("final", {}).get("baseline_m")
    if baseline is not None:
        fig.suptitle(f"Fused calibrated point clouds, baseline={baseline:.3f} m")
    else:
        fig.suptitle("Fused calibrated point clouds")
    fig.savefig(path, dpi=180)
    plt.close(fig)


def main() -> int:
    args = parse_args()
    camera_ids = [args.camera0, args.camera1]
    snapshot_dir = resolve_snapshot_dir(args.snapshots, args.snapshot_id, camera_ids)
    output_dir = args.out / snapshot_dir.name
    rng = np.random.default_rng(42)

    world_transforms, transform_keys, calibration = load_world_transforms(args.calibration, camera_ids)
    clouds: list[dict[str, Any]] = []

    for camera_id in camera_ids:
        cloud = load_camera_cloud(
            snapshot_dir=snapshot_dir,
            camera_id=camera_id,
            world_from_camera=world_transforms[camera_id],
            stride=args.stride,
            min_depth_m=args.min_depth_m,
            max_depth_m=args.max_depth_m,
            max_points=args.max_points_per_camera,
            rng=rng,
        )
        clouds.append(cloud)

        write_ply(
            output_dir / f"{camera_id}_world.ply",
            cloud["points_world"],
            cloud["colors_rgb"],
        )

    fused_points = np.concatenate([cloud["points_world"] for cloud in clouds], axis=0)
    fused_colors = np.concatenate([cloud["colors_rgb"] for cloud in clouds], axis=0)
    fused_ply = output_dir / "fused_world.ply"
    write_ply(fused_ply, fused_points, fused_colors)

    debug_colors = []
    for idx, cloud in enumerate(clouds):
        color = CAMERA_DEBUG_COLORS[idx % len(CAMERA_DEBUG_COLORS)]
        debug_colors.append(np.tile(color, (cloud["points_world"].shape[0], 1)))
    fused_debug_ply = output_dir / "fused_world_by_camera.ply"
    write_ply(fused_debug_ply, fused_points, np.concatenate(debug_colors, axis=0))

    projection_plot = output_dir / "world_projections.png"
    make_projection_plot(
        projection_plot,
        clouds,
        calibration,
        max_points_per_camera=args.plot_max_points_per_camera,
        rng=rng,
    )

    summary = {
        "snapshot_id": snapshot_dir.name,
        "snapshot_dir": str(snapshot_dir),
        "calibration": str(args.calibration),
        "output_dir": str(output_dir),
        "fused_ply": str(fused_ply),
        "fused_by_camera_ply": str(fused_debug_ply),
        "projection_plot": str(projection_plot),
        "stride": args.stride,
        "min_depth_m": args.min_depth_m,
        "max_depth_m": args.max_depth_m,
        "baseline_m": calibration.get("final", {}).get("baseline_m"),
        "coordinate_chain": {
            "depth_source": "aligned_depth.npy",
            "depth_frame": "color camera frame",
            "backprojection": "aligned depth pixels + metadata.color_intrinsics -> color camera 3D",
            "world_transform": "T_world_from_*_color -> static board/world frame",
            "raw_depth_note": (
                "raw_depth.npy is not used here. Raw depth would require depth intrinsics "
                "and Azure factory T_color_from_depth before applying T_world_from_color."
            ),
        },
        "cameras": [
            {
                "camera_id": cloud["camera_id"],
                "serial_number": cloud["metadata"].get("serial_number"),
                "depth_file": cloud["depth_file"],
                "depth_frame": cloud["depth_frame"],
                "intrinsics_source": cloud["intrinsics_source"],
                "transform_key": transform_keys[cloud["camera_id"]],
                "color_shape": cloud["color_shape"],
                "aligned_depth_shape": cloud["aligned_depth_shape"],
                "points": cloud["valid_points"],
                "ply": str(output_dir / f"{cloud['camera_id']}_world.ply"),
            }
            for cloud in clouds
        ],
    }
    write_json(output_dir / "summary.json", summary)

    print(f"snapshot: {snapshot_dir.name}")
    print(f"calibration: {args.calibration}")
    for camera in summary["cameras"]:
        print(
            f"{camera['camera_id']} serial={camera['serial_number']} "
            f"points={camera['points']} color={camera['color_shape']} "
            f"aligned_depth={camera['aligned_depth_shape']}"
        )
    print(f"fused ply: {fused_ply}")
    print(f"fused by-camera ply: {fused_debug_ply}")
    print(f"projection plot: {projection_plot}")
    print(f"summary: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
