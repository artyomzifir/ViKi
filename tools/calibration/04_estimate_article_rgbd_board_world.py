#!/usr/bin/env python3
"""
Estimate RGB-D board/world calibration from ChArUco RGB corners + aligned depth.

This is the article-style path: RGB gives board corner IDs/pixels, aligned depth
turns those pixels into 3D points in each color camera frame, and a rigid
Kabsch fit estimates the board pose from measured 3D points.
"""

from __future__ import annotations

import argparse
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

try:
    from scipy.spatial.transform import Rotation
except Exception:  # pragma: no cover - only used on minimal installs
    Rotation = None


DEFAULT_SQUARES_X = 8
DEFAULT_SQUARES_Y = 10
# Measured board: 484 mm x 384 mm -> (484/10 + 384/8)/2 = 48.2 mm.
DEFAULT_SQUARE_LENGTH_M = 0.0482
DEFAULT_MARKER_RATIO = 0.75
DEFAULT_CAMERA0 = "kinect_0"
DEFAULT_CAMERA1 = "kinect_1"


@dataclass
class CameraFit:
    camera_id: str
    serial_number: str | None
    status: str
    reason: str | None
    detected_corners: int
    valid_depth_points: int
    used_points: int
    rejected_depth_points: int
    rejected_fit_points: int
    transform_camera_from_board: np.ndarray | None
    transform_board_from_camera: np.ndarray | None
    residuals_mm: np.ndarray
    used_charuco_ids: list[int]


@dataclass
class PairEstimate:
    snapshot_id: str
    status: str
    reason: str | None
    camera_fits: dict[str, CameraFit]
    transform_cam1_from_cam0: np.ndarray | None
    baseline_m: float | None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Article-style RGB-D board/world calibration from ChArUco + aligned depth."
    )
    parser.add_argument("--snapshots", type=Path, default=Path("data/snapshots"))
    parser.add_argument(
        "--detections",
        type=Path,
        default=Path("data/calibration/charuco_detect"),
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/calibration/article_rgbd_board_world.json"),
    )
    parser.add_argument(
        "--mode",
        choices=("moving-board-relative", "static-board-world"),
        default="moving-board-relative",
    )
    parser.add_argument("--camera0", default=DEFAULT_CAMERA0)
    parser.add_argument("--camera1", default=DEFAULT_CAMERA1)
    parser.add_argument("--squares-x", type=int, default=DEFAULT_SQUARES_X)
    parser.add_argument("--squares-y", type=int, default=DEFAULT_SQUARES_Y)
    parser.add_argument("--square-length-m", type=float, default=DEFAULT_SQUARE_LENGTH_M)
    parser.add_argument(
        "--marker-length-m",
        type=float,
        default=None,
        help="Defaults to square-length-m * 0.75.",
    )
    parser.add_argument("--dictionary", default="DICT_5X5_1000")
    parser.add_argument("--depth-window", type=int, default=5)
    parser.add_argument("--min-depth-m", type=float, default=0.4)
    parser.add_argument("--max-depth-m", type=float, default=4.0)
    parser.add_argument("--min-valid-points", type=int, default=20)
    parser.add_argument("--max-fit-residual-mm", type=float, default=30.0)
    return parser.parse_args()


def dictionary_id(name: str) -> int:
    if not name.startswith("DICT_"):
        name = f"DICT_{name}"
    if not hasattr(cv2.aruco, name):
        raise ValueError(f"Unknown cv2.aruco dictionary {name}")
    return int(getattr(cv2.aruco, name))


def get_aruco_dictionary(name: str):
    dict_id = dictionary_id(name)
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(dict_id)
    return cv2.aruco.Dictionary_get(dict_id)


def make_charuco_board(args: argparse.Namespace):
    dictionary = get_aruco_dictionary(args.dictionary)
    marker_length_m = marker_length(args)
    if hasattr(cv2.aruco, "CharucoBoard"):
        try:
            return cv2.aruco.CharucoBoard(
                (args.squares_x, args.squares_y),
                args.square_length_m,
                marker_length_m,
                dictionary,
            )
        except TypeError:
            pass
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        return cv2.aruco.CharucoBoard_create(
            args.squares_x,
            args.squares_y,
            args.square_length_m,
            marker_length_m,
            dictionary,
        )
    raise RuntimeError("This OpenCV build does not expose ChArUco board creation.")


def marker_length(args: argparse.Namespace) -> float:
    return args.marker_length_m if args.marker_length_m is not None else args.square_length_m * DEFAULT_MARKER_RATIO


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def iter_pair_snapshots(snapshots_dir: Path, camera0: str, camera1: str) -> list[Path]:
    if not snapshots_dir.exists():
        return []
    pair_dirs = []
    for path in sorted(snapshots_dir.iterdir()):
        if not path.is_dir() or not path.name.startswith("pair_"):
            continue
        if (
            (path / camera0 / "metadata.json").exists()
            and (path / camera1 / "metadata.json").exists()
            and (path / camera0 / "aligned_depth.npy").exists()
            and (path / camera1 / "aligned_depth.npy").exists()
        ):
            pair_dirs.append(path)
    return pair_dirs


def load_intrinsics_and_serial(snapshot_dir: Path, camera_id: str) -> tuple[np.ndarray, np.ndarray, str | None]:
    metadata = read_json(snapshot_dir / camera_id / "metadata.json")
    intrinsics = metadata.get("color_intrinsics")
    if not intrinsics:
        raise ValueError(f"{snapshot_dir / camera_id / 'metadata.json'} has no color_intrinsics")
    camera_matrix = np.asarray(intrinsics["camera_matrix"], dtype=np.float64)
    dist_coeffs = np.asarray(intrinsics.get("dist_coeffs", []), dtype=np.float64).reshape(-1, 1)
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"invalid camera_matrix shape {camera_matrix.shape}")
    return camera_matrix, dist_coeffs, metadata.get("serial_number")


def load_detection(detections_dir: Path, snapshot_id: str, camera_id: str) -> dict[str, Any]:
    path = detections_dir / snapshot_id / f"{camera_id}_detections.json"
    if not path.exists():
        raise ValueError(f"missing detection file {path}")
    return read_json(path)


def detection_map(detection: dict[str, Any]) -> dict[int, tuple[float, float]]:
    return {
        int(charuco_id): (float(corner[0]), float(corner[1]))
        for charuco_id, corner in zip(
            detection.get("charuco_ids", []),
            detection.get("charuco_corners", []),
        )
    }


def sample_depth_m(
    depth: np.ndarray,
    pixel: tuple[float, float],
    min_depth_m: float,
    max_depth_m: float,
    window: int,
) -> float | None:
    u, v = pixel
    x = int(round(u))
    y = int(round(v))
    h, w = depth.shape[:2]
    if x < 0 or y < 0 or x >= w or y >= h:
        return None

    if window <= 1:
        patch = np.asarray([depth[y, x]], dtype=np.float64)
    else:
        radius = window // 2
        patch = depth[
            max(0, y - radius):min(h, y + radius + 1),
            max(0, x - radius):min(w, x + radius + 1),
        ].astype(np.float64)

    values = patch[np.isfinite(patch) & (patch > 0)]
    if values.size == 0:
        return None
    depths_m = values / 1000.0
    depths_m = depths_m[(depths_m >= min_depth_m) & (depths_m <= max_depth_m)]
    if depths_m.size == 0:
        return None
    return float(np.median(depths_m))


def backproject_color_pixel(
    pixel: tuple[float, float],
    depth_m: float,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> np.ndarray:
    pixels = np.asarray([[pixel]], dtype=np.float64)
    normalized = cv2.undistortPoints(pixels, camera_matrix, dist_coeffs).reshape(2)
    x, y = normalized
    return np.asarray([x * depth_m, y * depth_m, depth_m], dtype=np.float64)


def rigid_fit_no_scale(source_points: np.ndarray, target_points: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Return R, t such that target ~= R @ source + t."""
    source = np.asarray(source_points, dtype=np.float64).reshape(-1, 3)
    target = np.asarray(target_points, dtype=np.float64).reshape(-1, 3)
    if len(source) != len(target) or len(source) < 3:
        raise ValueError("rigid fit needs matching source/target arrays with at least 3 points")

    source_center = np.mean(source, axis=0)
    target_center = np.mean(target, axis=0)
    source_zero = source - source_center
    target_zero = target - target_center
    covariance = source_zero.T @ target_zero
    u, _s, vt = np.linalg.svd(covariance)
    rotation = vt.T @ u.T
    if np.linalg.det(rotation) < 0:
        vt[-1, :] *= -1.0
        rotation = vt.T @ u.T
    translation = target_center - rotation @ source_center
    return rotation, translation


def transform_from_rotation_translation(rotation: np.ndarray, translation: np.ndarray) -> np.ndarray:
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(translation, dtype=np.float64).reshape(3)
    return transform


def residuals_mm(transform: np.ndarray, board_points: np.ndarray, camera_points: np.ndarray) -> np.ndarray:
    predicted = (transform[:3, :3] @ board_points.T).T + transform[:3, 3]
    return np.linalg.norm(predicted - camera_points, axis=1) * 1000.0


def fit_board_to_camera(
    camera_id: str,
    serial_number: str | None,
    board_points: np.ndarray,
    camera_points: np.ndarray,
    charuco_ids: list[int],
    detected_corners: int,
    rejected_depth_points: int,
    args: argparse.Namespace,
) -> CameraFit:
    if len(board_points) < args.min_valid_points:
        return CameraFit(
            camera_id=camera_id,
            serial_number=serial_number,
            status="fail",
            reason=f"valid depth points {len(board_points)} < min_valid_points {args.min_valid_points}",
            detected_corners=detected_corners,
            valid_depth_points=len(board_points),
            used_points=0,
            rejected_depth_points=rejected_depth_points,
            rejected_fit_points=0,
            transform_camera_from_board=None,
            transform_board_from_camera=None,
            residuals_mm=np.asarray([], dtype=np.float64),
            used_charuco_ids=[],
        )

    rotation, translation = rigid_fit_no_scale(board_points, camera_points)
    transform = transform_from_rotation_translation(rotation, translation)
    initial_residuals = residuals_mm(transform, board_points, camera_points)
    keep = initial_residuals <= args.max_fit_residual_mm
    rejected_fit_points = int(np.count_nonzero(~keep))

    if np.count_nonzero(keep) >= args.min_valid_points:
        board_used = board_points[keep]
        camera_used = camera_points[keep]
        ids_used = [charuco_id for charuco_id, keep_it in zip(charuco_ids, keep) if keep_it]
        rotation, translation = rigid_fit_no_scale(board_used, camera_used)
        transform = transform_from_rotation_translation(rotation, translation)
        final_residuals = residuals_mm(transform, board_used, camera_used)
    else:
        board_used = board_points
        camera_used = camera_points
        ids_used = charuco_ids
        final_residuals = initial_residuals
        rejected_fit_points = 0

    status = "ok" if len(board_used) >= args.min_valid_points else "fail"
    reason = None if status == "ok" else "not enough points after robust fit"
    return CameraFit(
        camera_id=camera_id,
        serial_number=serial_number,
        status=status,
        reason=reason,
        detected_corners=detected_corners,
        valid_depth_points=len(board_points),
        used_points=len(board_used),
        rejected_depth_points=rejected_depth_points,
        rejected_fit_points=rejected_fit_points,
        transform_camera_from_board=transform,
        transform_board_from_camera=np.linalg.inv(transform),
        residuals_mm=final_residuals,
        used_charuco_ids=ids_used,
    )


def estimate_camera_fit(
    snapshot_dir: Path,
    detections_dir: Path,
    board_corners: np.ndarray,
    camera_id: str,
    args: argparse.Namespace,
) -> CameraFit:
    snapshot_id = snapshot_dir.name
    camera_matrix, dist_coeffs, serial_number = load_intrinsics_and_serial(snapshot_dir, camera_id)
    detection = load_detection(detections_dir, snapshot_id, camera_id)
    pixels_by_id = detection_map(detection)
    depth = np.load(snapshot_dir / camera_id / "aligned_depth.npy")

    board_points = []
    camera_points = []
    used_ids = []
    rejected_depth = 0
    for charuco_id, pixel in sorted(pixels_by_id.items()):
        if charuco_id < 0 or charuco_id >= len(board_corners):
            continue
        depth_m = sample_depth_m(
            depth,
            pixel,
            args.min_depth_m,
            args.max_depth_m,
            args.depth_window,
        )
        if depth_m is None:
            rejected_depth += 1
            continue
        board_points.append(board_corners[charuco_id])
        camera_points.append(backproject_color_pixel(pixel, depth_m, camera_matrix, dist_coeffs))
        used_ids.append(charuco_id)

    return fit_board_to_camera(
        camera_id=camera_id,
        serial_number=serial_number,
        board_points=np.asarray(board_points, dtype=np.float64).reshape(-1, 3),
        camera_points=np.asarray(camera_points, dtype=np.float64).reshape(-1, 3),
        charuco_ids=used_ids,
        detected_corners=len(pixels_by_id),
        rejected_depth_points=rejected_depth,
        args=args,
    )


def estimate_pair(
    snapshot_dir: Path,
    detections_dir: Path,
    board_corners: np.ndarray,
    args: argparse.Namespace,
) -> PairEstimate:
    camera_fits: dict[str, CameraFit] = {}
    for camera_id in (args.camera0, args.camera1):
        try:
            camera_fits[camera_id] = estimate_camera_fit(
                snapshot_dir,
                detections_dir,
                board_corners,
                camera_id,
                args,
            )
        except Exception as exc:
            camera_fits[camera_id] = CameraFit(
                camera_id=camera_id,
                serial_number=None,
                status="fail",
                reason=str(exc),
                detected_corners=0,
                valid_depth_points=0,
                used_points=0,
                rejected_depth_points=0,
                rejected_fit_points=0,
                transform_camera_from_board=None,
                transform_board_from_camera=None,
                residuals_mm=np.asarray([], dtype=np.float64),
                used_charuco_ids=[],
            )

    failed = [fit for fit in camera_fits.values() if fit.status != "ok"]
    if failed:
        reason = "; ".join(f"{fit.camera_id}: {fit.reason}" for fit in failed)
        return PairEstimate(
            snapshot_id=snapshot_dir.name,
            status="fail",
            reason=reason,
            camera_fits=camera_fits,
            transform_cam1_from_cam0=None,
            baseline_m=None,
        )

    fit0 = camera_fits[args.camera0]
    fit1 = camera_fits[args.camera1]
    assert fit0.transform_camera_from_board is not None
    assert fit1.transform_camera_from_board is not None
    transform = fit1.transform_camera_from_board @ np.linalg.inv(fit0.transform_camera_from_board)
    return PairEstimate(
        snapshot_id=snapshot_dir.name,
        status="ok",
        reason=None,
        camera_fits=camera_fits,
        transform_cam1_from_cam0=transform,
        baseline_m=float(np.linalg.norm(transform[:3, 3])),
    )


def aggregate_transforms(transforms: list[np.ndarray], scores: list[float]) -> tuple[np.ndarray, str]:
    if not transforms:
        raise RuntimeError("No transforms to aggregate")
    translations = np.asarray([transform[:3, 3] for transform in transforms], dtype=np.float64)
    final_translation = np.median(translations, axis=0)

    if Rotation is not None:
        final_rotation = Rotation.from_matrix([transform[:3, :3] for transform in transforms]).mean().as_matrix()
        method = "median_translation_scipy_rotation_mean"
    else:
        best_index = int(np.argmin(scores))
        final_rotation = transforms[best_index][:3, :3]
        method = "median_translation_best_residual_rotation"

    final = np.eye(4, dtype=np.float64)
    final[:3, :3] = final_rotation
    final[:3, 3] = final_translation
    return final, method


def rotation_delta_deg(rotation: np.ndarray, reference_rotation: np.ndarray) -> float:
    delta = rotation @ reference_rotation.T
    rvec, _ = cv2.Rodrigues(delta)
    return float(np.linalg.norm(rvec) * 180.0 / math.pi)


def transform_spread(transforms: list[np.ndarray], reference: np.ndarray) -> dict[str, Any]:
    if not transforms:
        return {
            "translation_std_mm": None,
            "translation_std_xyz_mm": None,
            "rotation_std_deg": None,
            "rotation_mean_delta_deg": None,
        }
    translations = np.asarray([transform[:3, 3] for transform in transforms], dtype=np.float64)
    std_xyz_mm = np.std(translations, axis=0) * 1000.0
    rotation_deltas = [
        rotation_delta_deg(transform[:3, :3], reference[:3, :3])
        for transform in transforms
    ]
    return {
        "translation_std_mm": float(np.linalg.norm(std_xyz_mm)),
        "translation_std_xyz_mm": [float(v) for v in std_xyz_mm],
        "rotation_std_deg": float(np.std(rotation_deltas)),
        "rotation_mean_delta_deg": float(np.mean(rotation_deltas)),
    }


def residual_metrics(values: np.ndarray) -> dict[str, float | None]:
    values = np.asarray(values, dtype=np.float64).reshape(-1)
    if values.size == 0:
        return {"mean_mm": None, "median_mm": None, "rmse_mm": None, "p95_mm": None}
    return {
        "mean_mm": float(np.mean(values)),
        "median_mm": float(np.median(values)),
        "rmse_mm": float(np.sqrt(np.mean(values * values))),
        "p95_mm": float(np.percentile(values, 95)),
    }


def fit_score(pair: PairEstimate) -> float:
    residuals = []
    for fit in pair.camera_fits.values():
        residuals.extend(fit.residuals_mm.tolist())
    return float(np.mean(residuals)) if residuals else float("inf")


def build_final(args: argparse.Namespace, pairs: list[PairEstimate]) -> dict[str, Any]:
    used = [pair for pair in pairs if pair.status == "ok"]
    rejected = [pair for pair in pairs if pair.status != "ok"]
    if not used:
        raise RuntimeError("No usable snapshots")

    scores = [fit_score(pair) for pair in used]
    relative_transforms = [pair.transform_cam1_from_cam0 for pair in used]
    assert all(transform is not None for transform in relative_transforms)

    final: dict[str, Any] = {}
    if args.mode == "moving-board-relative":
        final_relative, method = aggregate_transforms(relative_transforms, scores)
        final["T_kinect1_color_from_kinect0_color"] = final_relative.tolist()
        final["T_kinect0_color_from_kinect1_color"] = np.linalg.inv(final_relative).tolist()
        final["baseline_m"] = float(np.linalg.norm(final_relative[:3, 3]))
        final["aggregation_method"] = method
        final["relative_spread"] = transform_spread(relative_transforms, final_relative)
    else:
        camera_world: dict[str, np.ndarray] = {}
        camera_methods: dict[str, str] = {}
        for camera_id in (args.camera0, args.camera1):
            transforms = []
            camera_scores = []
            for pair in used:
                fit = pair.camera_fits[camera_id]
                assert fit.transform_board_from_camera is not None
                transforms.append(fit.transform_board_from_camera)
                camera_scores.append(float(np.mean(fit.residuals_mm)))
            camera_world[camera_id], camera_methods[camera_id] = aggregate_transforms(transforms, camera_scores)

        transform_cam1_from_cam0 = np.linalg.inv(camera_world[args.camera1]) @ camera_world[args.camera0]
        final["T_world_from_kinect0_color"] = camera_world[args.camera0].tolist()
        final["T_world_from_kinect1_color"] = camera_world[args.camera1].tolist()
        final["T_kinect1_color_from_kinect0_color"] = transform_cam1_from_cam0.tolist()
        final["T_kinect0_color_from_kinect1_color"] = np.linalg.inv(transform_cam1_from_cam0).tolist()
        final["baseline_m"] = float(np.linalg.norm(transform_cam1_from_cam0[:3, 3]))
        final["aggregation_method"] = camera_methods
        final["relative_spread"] = transform_spread(relative_transforms, transform_cam1_from_cam0)

    all_residuals = np.asarray(
        [
            residual
            for pair in used
            for fit in pair.camera_fits.values()
            for residual in fit.residuals_mm
        ],
        dtype=np.float64,
    )
    final["used_snapshots"] = [pair.snapshot_id for pair in used]
    final["rejected_snapshots"] = [
        {"snapshot_id": pair.snapshot_id, "reason": pair.reason}
        for pair in rejected
    ]
    final["point_counts"] = {
        "used_points": int(sum(fit.used_points for pair in used for fit in pair.camera_fits.values())),
        "rejected_depth_points": int(sum(fit.rejected_depth_points for pair in used for fit in pair.camera_fits.values())),
        "rejected_fit_points": int(sum(fit.rejected_fit_points for pair in used for fit in pair.camera_fits.values())),
    }
    final["board_fit_residuals"] = residual_metrics(all_residuals)
    return final


def camera_fit_to_json(fit: CameraFit) -> dict[str, Any]:
    return {
        "camera_id": fit.camera_id,
        "serial_number": fit.serial_number,
        "status": fit.status,
        "reason": fit.reason,
        "detected_corners": fit.detected_corners,
        "valid_depth_points": fit.valid_depth_points,
        "used_points": fit.used_points,
        "rejected_depth_points": fit.rejected_depth_points,
        "rejected_fit_points": fit.rejected_fit_points,
        "used_charuco_ids": fit.used_charuco_ids,
        "fit_residuals": residual_metrics(fit.residuals_mm),
        "T_camera_from_board": None if fit.transform_camera_from_board is None else fit.transform_camera_from_board.tolist(),
        "T_board_from_camera": None if fit.transform_board_from_camera is None else fit.transform_board_from_camera.tolist(),
    }


def pair_to_json(pair: PairEstimate) -> dict[str, Any]:
    return {
        "snapshot_id": pair.snapshot_id,
        "status": pair.status,
        "reason": pair.reason,
        "baseline_m": pair.baseline_m,
        "cameras": {
            camera_id: camera_fit_to_json(fit)
            for camera_id, fit in pair.camera_fits.items()
        },
        "T_kinect1_color_from_kinect0_color": (
            None if pair.transform_cam1_from_cam0 is None else pair.transform_cam1_from_cam0.tolist()
        ),
    }


def print_table(pairs: list[PairEstimate]) -> None:
    headers = ["snapshot_id", "camera_id", "valid_points", "fit_median_mm", "fit_p95_mm", "status"]
    rows = []
    for pair in pairs:
        for camera_id, fit in pair.camera_fits.items():
            metrics = residual_metrics(fit.residuals_mm)
            status = fit.status if fit.reason is None else f"{fit.status}: {fit.reason}"
            rows.append(
                [
                    pair.snapshot_id,
                    camera_id,
                    str(fit.used_points),
                    fmt(metrics["median_mm"]),
                    fmt(metrics["p95_mm"]),
                    status,
                ]
            )
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows)) if rows else len(header)
        for index, header in enumerate(headers)
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def fmt(value: float | None) -> str:
    return "" if value is None else f"{value:.2f}"


def main() -> int:
    args = parse_args()
    board = make_charuco_board(args)
    board_corners = np.asarray(board.getChessboardCorners(), dtype=np.float64)

    pair_dirs = iter_pair_snapshots(args.snapshots, args.camera0, args.camera1)
    if not pair_dirs:
        print(f"No pair snapshots found in {args.snapshots}")
        return 1

    pairs = [
        estimate_pair(pair_dir, args.detections, board_corners, args)
        for pair_dir in pair_dirs
    ]
    print_table(pairs)
    final = build_final(args, pairs)

    output = {
        "board": {
            "squares_x": args.squares_x,
            "squares_y": args.squares_y,
            "square_length_m": args.square_length_m,
            "marker_length_m": marker_length(args),
            "dictionary": args.dictionary,
        },
        "mode": args.mode,
        "devices": {
            "camera0": args.camera0,
            "camera1": args.camera1,
            "serial_numbers": serial_numbers(pairs),
        },
        "final": final,
        "per_snapshot": [pair_to_json(pair) for pair in pairs],
    }

    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print_summary(output)
    print(f"Saved: {args.out}")
    return 0


def serial_numbers(pairs: list[PairEstimate]) -> dict[str, str | None]:
    result: dict[str, str | None] = {}
    for pair in pairs:
        for camera_id, fit in pair.camera_fits.items():
            if fit.serial_number and camera_id not in result:
                result[camera_id] = fit.serial_number
    return result


def print_summary(output: dict[str, Any]) -> None:
    final = output["final"]
    print()
    print(f"mode: {output['mode']}")
    print(f"used_snapshots: {len(final['used_snapshots'])}")
    print(f"rejected_snapshots: {len(final['rejected_snapshots'])}")
    print(f"baseline_m: {final['baseline_m']:.6f}")
    residuals = final["board_fit_residuals"]
    print(
        "board_fit_residuals_mm: "
        f"mean={fmt(residuals['mean_mm'])} "
        f"median={fmt(residuals['median_mm'])} "
        f"rmse={fmt(residuals['rmse_mm'])} "
        f"p95={fmt(residuals['p95_mm'])}"
    )
    spread = final["relative_spread"]
    print(f"relative_translation_std_mm: {fmt(spread['translation_std_mm'])}")
    print(f"relative_rotation_std_deg: {fmt(spread['rotation_std_deg'])}")
    matrix = np.asarray(final["T_kinect1_color_from_kinect0_color"])
    print("T_kinect1_color_from_kinect0_color:")
    with np.printoptions(precision=6, suppress=True):
        print(matrix)


if __name__ == "__main__":
    raise SystemExit(main())
