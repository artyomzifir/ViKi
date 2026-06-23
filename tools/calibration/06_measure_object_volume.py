#!/usr/bin/env python3
"""
Measure an object volume above the calibrated static ChArUco board plane.

This is intended as a practical smoke test for the static-board/world
calibration: aligned depth from both cameras is fused into the board frame,
points above the board are segmented, and a volume estimate is computed.
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


DEFAULT_CALIBRATION = Path("data/calibration/final/rig_20260623_static_world_v2/static_world.json")
DEFAULT_SNAPSHOTS = Path("data/snapshots")
DEFAULT_OUT = Path("data/calibration/volume_check")
DEFAULT_CAMERA0 = "kinect_0"
DEFAULT_CAMERA1 = "kinect_1"
CAMERA_DEBUG_COLORS = (
    np.asarray([31, 119, 180], dtype=np.uint8),
    np.asarray([255, 127, 14], dtype=np.uint8),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate volume of an object resting on the calibrated board plane."
    )
    parser.add_argument("--snapshot", type=Path, default=None, help="Direct path to one pair_* snapshot.")
    parser.add_argument("--snapshots", type=Path, default=DEFAULT_SNAPSHOTS)
    parser.add_argument("--snapshot-id", default="latest")
    parser.add_argument("--calibration", type=Path, default=DEFAULT_CALIBRATION)
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--camera0", default=DEFAULT_CAMERA0)
    parser.add_argument("--camera1", default=DEFAULT_CAMERA1)
    parser.add_argument("--stride", type=int, default=2, help="Pixel stride for point sampling.")
    parser.add_argument("--cell-mm", type=float, default=5.0, help="Height-map cell size.")
    parser.add_argument("--min-depth-m", type=float, default=0.4)
    parser.add_argument("--max-depth-m", type=float, default=4.0)
    parser.add_argument("--min-height-mm", type=float, default=10.0)
    parser.add_argument("--max-height-mm", type=float, default=300.0)
    parser.add_argument(
        "--height-sign",
        choices=("auto", "positive", "negative", "absolute"),
        default="auto",
        help="Which side of board Z=0 contains the object.",
    )
    parser.add_argument("--board-margin-mm", type=float, default=15.0)
    parser.add_argument("--roi-x-min-m", type=float, default=None)
    parser.add_argument("--roi-x-max-m", type=float, default=None)
    parser.add_argument("--roi-y-min-m", type=float, default=None)
    parser.add_argument("--roi-y-max-m", type=float, default=None)
    parser.add_argument(
        "--expected-size-cm",
        nargs=3,
        type=float,
        default=None,
        metavar=("A", "B", "C"),
        help="Known object dimensions, for example: --expected-size-cm 4.8 4.8 12.5",
    )
    parser.add_argument("--max-points-per-camera", type=int, default=250_000)
    parser.add_argument("--plot-max-points", type=int, default=60_000)
    parser.add_argument("--no-largest-component", action="store_true")
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


def resolve_snapshot_dir(args: argparse.Namespace, camera_ids: list[str]) -> Path:
    if args.snapshot is not None:
        if not args.snapshot.exists():
            raise FileNotFoundError(args.snapshot)
        return args.snapshot

    if args.snapshot_id == "latest":
        pairs = iter_complete_pairs(args.snapshots, camera_ids)
        if not pairs:
            raise FileNotFoundError(f"No complete pair snapshots found in {args.snapshots}")
        return pairs[-1]

    path = args.snapshots / args.snapshot_id
    if not path.exists():
        raise FileNotFoundError(f"Snapshot {args.snapshot_id!r} not found in {args.snapshots}")
    return path


def transform_points(transform: np.ndarray, points: np.ndarray) -> np.ndarray:
    return points @ transform[:3, :3].T + transform[:3, 3]


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
            compact_key = f"T_world_from_{camera_id.replace('kinect_', 'kinect')}_color"
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
    depth_m = depth[ys_flat, xs_flat].astype(np.float64) / 1000.0

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
        "points_world": points_world,
        "colors_rgb": colors_rgb,
        "color_shape": list(color.shape),
        "aligned_depth_shape": list(depth.shape),
    }


def board_bounds(calibration: dict[str, Any], args: argparse.Namespace) -> dict[str, float]:
    board = calibration.get("board", {})
    square = float(board.get("square_length_m", 0.0) or 0.0)
    width = float(board.get("squares_x", 0.0) or 0.0) * square
    height = float(board.get("squares_y", 0.0) or 0.0) * square
    if width <= 0.0 or height <= 0.0:
        raise ValueError("Calibration JSON has no valid board dimensions")

    margin = args.board_margin_mm / 1000.0
    return {
        "x_min": args.roi_x_min_m if args.roi_x_min_m is not None else -margin,
        "x_max": args.roi_x_max_m if args.roi_x_max_m is not None else width + margin,
        "y_min": args.roi_y_min_m if args.roi_y_min_m is not None else -margin,
        "y_max": args.roi_y_max_m if args.roi_y_max_m is not None else height + margin,
        "board_width_m": width,
        "board_height_m": height,
    }


def choose_height_sign(points: np.ndarray, inside_roi: np.ndarray, args: argparse.Namespace) -> tuple[str, np.ndarray]:
    z = points[:, 2]
    min_h = args.min_height_mm / 1000.0
    max_h = args.max_height_mm / 1000.0

    if args.height_sign == "positive":
        return "positive", z
    if args.height_sign == "negative":
        return "negative", -z
    if args.height_sign == "absolute":
        return "absolute", np.abs(z)

    positive = inside_roi & (z >= min_h) & (z <= max_h)
    negative = inside_roi & (-z >= min_h) & (-z <= max_h)
    if int(np.count_nonzero(negative)) > int(np.count_nonzero(positive)):
        return "negative", -z
    return "positive", z


def build_height_grid(
    points: np.ndarray,
    colors: np.ndarray,
    camera_labels: np.ndarray,
    args: argparse.Namespace,
    calibration: dict[str, Any],
) -> dict[str, Any]:
    bounds = board_bounds(calibration, args)
    x = points[:, 0]
    y = points[:, 1]
    inside_roi = (
        (x >= bounds["x_min"])
        & (x <= bounds["x_max"])
        & (y >= bounds["y_min"])
        & (y <= bounds["y_max"])
    )

    sign_name, signed_height = choose_height_sign(points, inside_roi, args)
    min_h = args.min_height_mm / 1000.0
    max_h = args.max_height_mm / 1000.0
    object_mask = inside_roi & (signed_height >= min_h) & (signed_height <= max_h)

    object_points = points[object_mask]
    object_colors = colors[object_mask]
    object_camera_labels = camera_labels[object_mask]
    object_heights = signed_height[object_mask]
    if object_points.shape[0] < 20:
        raise RuntimeError(
            f"Only {object_points.shape[0]} object points found. "
            "Check that the object is on the calibrated board and adjust --min-height-mm/ROI."
        )

    cell = args.cell_mm / 1000.0
    nx = int(np.ceil((bounds["x_max"] - bounds["x_min"]) / cell))
    ny = int(np.ceil((bounds["y_max"] - bounds["y_min"]) / cell))
    ix = np.floor((object_points[:, 0] - bounds["x_min"]) / cell).astype(np.int32)
    iy = np.floor((object_points[:, 1] - bounds["y_min"]) / cell).astype(np.int32)
    valid_bins = (ix >= 0) & (ix < nx) & (iy >= 0) & (iy < ny)
    ix = ix[valid_bins]
    iy = iy[valid_bins]
    object_points = object_points[valid_bins]
    object_colors = object_colors[valid_bins]
    object_camera_labels = object_camera_labels[valid_bins]
    object_heights = object_heights[valid_bins]

    height_grid = np.zeros((ny, nx), dtype=np.float32)
    flat = iy * nx + ix
    np.maximum.at(height_grid.ravel(), flat, object_heights.astype(np.float32))

    component_mask = height_grid >= min_h
    selected_component = None
    if not args.no_largest_component:
        count, labels, stats, _centroids = cv2.connectedComponentsWithStats(
            component_mask.astype(np.uint8),
            connectivity=8,
        )
        if count > 1:
            areas = stats[1:, cv2.CC_STAT_AREA]
            best_label = int(np.argmax(areas) + 1)
            selected_component = {
                "label": best_label,
                "cells": int(stats[best_label, cv2.CC_STAT_AREA]),
            }
            component_mask = labels == best_label

    point_component_mask = component_mask[iy, ix]
    object_points = object_points[point_component_mask]
    object_colors = object_colors[point_component_mask]
    object_camera_labels = object_camera_labels[point_component_mask]
    object_heights = object_heights[point_component_mask]

    filtered_grid = np.where(component_mask, height_grid, 0.0)
    volume_heightmap_m3 = float(np.sum(filtered_grid, dtype=np.float64) * cell * cell)

    if object_points.shape[0] < 20:
        raise RuntimeError("Largest component filtering removed too many points.")

    return {
        "bounds": bounds,
        "cell_m": cell,
        "height_sign": sign_name,
        "height_grid": filtered_grid,
        "object_points": object_points,
        "object_colors": object_colors,
        "object_camera_labels": object_camera_labels,
        "object_heights": object_heights,
        "selected_component": selected_component,
        "volume_heightmap_m3": volume_heightmap_m3,
    }


def robust_box_estimate(object_points: np.ndarray, object_heights: np.ndarray) -> dict[str, Any]:
    xy = object_points[:, :2].astype(np.float32)
    rect = cv2.minAreaRect(xy)
    box = cv2.boxPoints(rect).astype(np.float64)
    width_m = float(rect[1][0])
    depth_m = float(rect[1][1])
    footprint_area_m2 = float(max(width_m, 0.0) * max(depth_m, 0.0))
    height_p95_m = float(np.percentile(object_heights, 95))
    height_p99_m = float(np.percentile(object_heights, 99))
    height_max_m = float(np.max(object_heights))
    volume_p95_m3 = footprint_area_m2 * height_p95_m
    volume_p99_m3 = footprint_area_m2 * height_p99_m

    return {
        "footprint_rect_xy_m": box.tolist(),
        "footprint_width_m": width_m,
        "footprint_depth_m": depth_m,
        "footprint_area_m2": footprint_area_m2,
        "height_p95_m": height_p95_m,
        "height_p99_m": height_p99_m,
        "height_max_m": height_max_m,
        "volume_p95_m3": volume_p95_m3,
        "volume_p99_m3": volume_p99_m3,
    }


def write_ply(path: Path, points: np.ndarray, colors_rgb: np.ndarray) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
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


def save_height_map(
    path: Path,
    result: dict[str, Any],
    box: dict[str, Any],
    expected_size_cm: list[float] | None,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    grid_mm = result["height_grid"] * 1000.0
    bounds = result["bounds"]
    extent = [
        bounds["x_min"] * 100.0,
        bounds["x_max"] * 100.0,
        bounds["y_min"] * 100.0,
        bounds["y_max"] * 100.0,
    ]

    fig, ax = plt.subplots(figsize=(8, 7), constrained_layout=True)
    image = ax.imshow(
        grid_mm,
        origin="lower",
        extent=extent,
        cmap="turbo",
        interpolation="nearest",
    )
    cbar = fig.colorbar(image, ax=ax)
    cbar.set_label("height above board, mm")

    rect = np.asarray(box["footprint_rect_xy_m"]) * 100.0
    rect_closed = np.vstack([rect, rect[0]])
    ax.plot(rect_closed[:, 0], rect_closed[:, 1], color="white", linewidth=2.0)
    ax.plot(rect_closed[:, 0], rect_closed[:, 1], color="black", linewidth=0.8)

    title = (
        f"height-map volume={result['volume_heightmap_m3'] * 1000.0:.3f} L, "
        f"box-fit volume={box['volume_p95_m3'] * 1000.0:.3f} L"
    )
    if expected_size_cm:
        expected_l = np.prod(np.asarray(expected_size_cm, dtype=np.float64) / 100.0) * 1000.0
        title += f", expected={expected_l:.3f} L"
    ax.set_title(title)
    ax.set_xlabel("world X, cm")
    ax.set_ylabel("world Y, cm")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.3)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def save_top_view(
    path: Path,
    result: dict[str, Any],
    box: dict[str, Any],
    camera_ids: list[str],
    rng: np.random.Generator,
    max_points: int,
) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    points = result["object_points"]
    labels = result["object_camera_labels"]
    if max_points > 0 and points.shape[0] > max_points:
        keep = rng.choice(points.shape[0], size=max_points, replace=False)
        points = points[keep]
        labels = labels[keep]

    fig, ax = plt.subplots(figsize=(7, 7), constrained_layout=True)
    for idx, camera_id in enumerate(camera_ids):
        mask = labels == idx
        if np.any(mask):
            color = CAMERA_DEBUG_COLORS[idx % len(CAMERA_DEBUG_COLORS)] / 255.0
            ax.scatter(
                points[mask, 0] * 100.0,
                points[mask, 1] * 100.0,
                s=1.0,
                alpha=0.55,
                c=[color],
                label=camera_id,
                rasterized=True,
            )

    rect = np.asarray(box["footprint_rect_xy_m"]) * 100.0
    rect_closed = np.vstack([rect, rect[0]])
    ax.plot(rect_closed[:, 0], rect_closed[:, 1], color="black", linewidth=1.2, label="min-area rect")
    ax.set_title("segmented object top view")
    ax.set_xlabel("world X, cm")
    ax.set_ylabel("world Y, cm")
    ax.set_aspect("equal", adjustable="box")
    ax.grid(True, linewidth=0.4, alpha=0.3)
    ax.legend(markerscale=5)
    fig.savefig(path, dpi=180)
    plt.close(fig)


def expected_summary(expected_size_cm: list[float] | None, box: dict[str, Any]) -> dict[str, Any] | None:
    if expected_size_cm is None:
        return None

    expected_m = np.asarray(expected_size_cm, dtype=np.float64) / 100.0
    expected_volume_l = float(np.prod(expected_m) * 1000.0)
    measured_dims_cm = np.asarray(
        [
            box["footprint_width_m"] * 100.0,
            box["footprint_depth_m"] * 100.0,
            box["height_p95_m"] * 100.0,
        ],
        dtype=np.float64,
    )
    measured_volume_l = float(box["volume_p95_m3"] * 1000.0)
    volume_error_pct = None
    if expected_volume_l > 0.0:
        volume_error_pct = float((measured_volume_l - expected_volume_l) / expected_volume_l * 100.0)

    return {
        "expected_size_cm": expected_size_cm,
        "expected_size_sorted_cm": sorted(float(v) for v in expected_size_cm),
        "expected_volume_l": expected_volume_l,
        "measured_dims_cm": measured_dims_cm.tolist(),
        "measured_dims_sorted_cm": sorted(float(v) for v in measured_dims_cm),
        "measured_volume_l": measured_volume_l,
        "volume_error_pct": volume_error_pct,
    }


def main() -> int:
    args = parse_args()
    camera_ids = [args.camera0, args.camera1]
    snapshot_dir = resolve_snapshot_dir(args, camera_ids)
    output_dir = args.out / snapshot_dir.name
    rng = np.random.default_rng(42)

    world_transforms, transform_keys, calibration = load_world_transforms(args.calibration, camera_ids)

    clouds = []
    for camera_id in camera_ids:
        clouds.append(
            load_camera_cloud(
                snapshot_dir=snapshot_dir,
                camera_id=camera_id,
                world_from_camera=world_transforms[camera_id],
                stride=args.stride,
                min_depth_m=args.min_depth_m,
                max_depth_m=args.max_depth_m,
                max_points=args.max_points_per_camera,
                rng=rng,
            )
        )

    points = np.concatenate([cloud["points_world"] for cloud in clouds], axis=0)
    colors = np.concatenate([cloud["colors_rgb"] for cloud in clouds], axis=0)
    labels = np.concatenate(
        [
            np.full(cloud["points_world"].shape[0], idx, dtype=np.int32)
            for idx, cloud in enumerate(clouds)
        ],
        axis=0,
    )

    result = build_height_grid(points, colors, labels, args, calibration)
    box = robust_box_estimate(result["object_points"], result["object_heights"])

    object_ply = output_dir / "object_points_rgb.ply"
    write_ply(object_ply, result["object_points"], result["object_colors"])

    debug_palette = np.vstack(CAMERA_DEBUG_COLORS)
    camera_debug_colors = debug_palette[result["object_camera_labels"] % len(CAMERA_DEBUG_COLORS)]
    object_by_camera_ply = output_dir / "object_points_by_camera.ply"
    write_ply(object_by_camera_ply, result["object_points"], camera_debug_colors)

    height_map = output_dir / "height_map.png"
    top_view = output_dir / "object_top_view.png"
    save_height_map(height_map, result, box, args.expected_size_cm)
    save_top_view(top_view, result, box, camera_ids, rng, args.plot_max_points)

    expected = expected_summary(args.expected_size_cm, box)
    summary = {
        "snapshot_id": snapshot_dir.name,
        "snapshot_dir": str(snapshot_dir),
        "calibration": str(args.calibration),
        "output_dir": str(output_dir),
        "coordinate_chain": {
            "depth_source": "aligned_depth.npy",
            "depth_frame": "color camera frame",
            "backprojection": "aligned depth pixels + metadata.color_intrinsics -> color camera 3D",
            "world_transform": "T_world_from_*_color -> static board/world frame",
            "board_plane": "world Z=0",
        },
        "parameters": {
            "stride": args.stride,
            "cell_mm": args.cell_mm,
            "min_height_mm": args.min_height_mm,
            "max_height_mm": args.max_height_mm,
            "height_sign": result["height_sign"],
            "roi": result["bounds"],
            "largest_component": not args.no_largest_component,
        },
        "cameras": [
            {
                "camera_id": cloud["camera_id"],
                "serial_number": cloud["metadata"].get("serial_number"),
                "transform_key": transform_keys[cloud["camera_id"]],
                "color_shape": cloud["color_shape"],
                "aligned_depth_shape": cloud["aligned_depth_shape"],
                "sampled_points": int(cloud["points_world"].shape[0]),
            }
            for cloud in clouds
        ],
        "segmentation": {
            "object_points": int(result["object_points"].shape[0]),
            "selected_component": result["selected_component"],
            "height_sign": result["height_sign"],
        },
        "measurement": {
            "footprint_width_cm": box["footprint_width_m"] * 100.0,
            "footprint_depth_cm": box["footprint_depth_m"] * 100.0,
            "height_p95_cm": box["height_p95_m"] * 100.0,
            "height_p99_cm": box["height_p99_m"] * 100.0,
            "height_max_cm": box["height_max_m"] * 100.0,
            "box_fit_volume_l": box["volume_p95_m3"] * 1000.0,
            "box_fit_volume_p99_l": box["volume_p99_m3"] * 1000.0,
            "heightmap_volume_l": result["volume_heightmap_m3"] * 1000.0,
            "footprint_rect_xy_m": box["footprint_rect_xy_m"],
        },
        "expected": expected,
        "files": {
            "object_points_rgb_ply": str(object_ply),
            "object_points_by_camera_ply": str(object_by_camera_ply),
            "height_map": str(height_map),
            "object_top_view": str(top_view),
            "summary": str(output_dir / "summary.json"),
        },
    }
    write_json(output_dir / "summary.json", summary)

    print(f"snapshot: {snapshot_dir.name}")
    print(f"object points: {summary['segmentation']['object_points']}")
    print(
        "measured dims cm: "
        f"{summary['measurement']['footprint_width_cm']:.2f} x "
        f"{summary['measurement']['footprint_depth_cm']:.2f} x "
        f"{summary['measurement']['height_p95_cm']:.2f}"
    )
    print(f"box-fit volume: {summary['measurement']['box_fit_volume_l']:.3f} L")
    print(f"height-map volume: {summary['measurement']['heightmap_volume_l']:.3f} L")
    if expected:
        print(
            f"expected volume: {expected['expected_volume_l']:.3f} L, "
            f"error: {expected['volume_error_pct']:.1f}%"
        )
    print(f"height map: {height_map}")
    print(f"top view: {top_view}")
    print(f"object ply: {object_ply}")
    print(f"summary: {output_dir / 'summary.json'}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
