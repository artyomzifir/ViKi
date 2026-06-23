#!/usr/bin/env python3
"""
Validate RGB camera extrinsics with aligned depth points at ChArUco corners.

This script does not estimate extrinsics. It uses the existing
T_kinect1_color_from_kinect0_color from 02_estimate_rgb_extrinsics.py and
checks whether depth-backed ChArUco corner points agree in millimetres.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


CAMERA0_ID = "kinect_0"
CAMERA1_ID = "kinect_1"


@dataclass
class DepthPoint:
    snapshot_id: str
    charuco_id: int
    p0_cam0: np.ndarray
    p0_cam1: np.ndarray
    p1_cam1: np.ndarray
    pixel0: tuple[float, float]
    pixel1: tuple[float, float]
    depth0_m: float
    depth1_m: float
    error_xyz_mm: np.ndarray
    error_mm: float


@dataclass
class SnapshotValidation:
    snapshot_id: str
    status: str
    reason: str | None
    common_corners: int
    valid_points: int
    rejected_depth_points: int
    rejected_outlier_points: int
    mean_mm: float | None
    median_mm: float | None
    rmse_mm: float | None
    p95_mm: float | None
    mean_abs_xyz_mm: list[float] | None
    points: list[DepthPoint]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Validate RGB extrinsics using aligned depth at ChArUco corners."
    )
    parser.add_argument("--snapshots", type=Path, default=Path("data/snapshots"))
    parser.add_argument(
        "--detections",
        type=Path,
        default=Path("data/calibration/charuco_detect"),
    )
    parser.add_argument(
        "--extrinsics",
        type=Path,
        default=Path("data/calibration/extrinsics_rgb.json"),
    )
    parser.add_argument(
        "--out-json",
        type=Path,
        default=Path("data/calibration/depth_validation.json"),
    )
    parser.add_argument(
        "--out-csv",
        type=Path,
        default=Path("data/calibration/depth_validation.csv"),
    )
    parser.add_argument("--camera0", default=CAMERA0_ID)
    parser.add_argument("--camera1", default=CAMERA1_ID)
    parser.add_argument("--min-valid-points", type=int, default=20)
    parser.add_argument("--min-depth-m", type=float, default=0.4)
    parser.add_argument("--max-depth-m", type=float, default=4.0)
    parser.add_argument("--max-error-mm", type=float, default=100.0)
    parser.add_argument(
        "--depth-window",
        type=int,
        default=1,
        help="Odd local window size for nonzero median depth sampling. 1 = nearest pixel.",
    )
    return parser.parse_args()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_intrinsics(snapshot_dir: Path, camera_id: str) -> tuple[np.ndarray, np.ndarray]:
    metadata = read_json(snapshot_dir / camera_id / "metadata.json")
    intrinsics = metadata.get("color_intrinsics")
    if not intrinsics:
        raise ValueError(f"{snapshot_dir / camera_id / 'metadata.json'} has no color_intrinsics")

    camera_matrix = np.asarray(intrinsics.get("camera_matrix"), dtype=np.float64)
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"invalid camera_matrix shape for {camera_id}: {camera_matrix.shape}")
    dist_coeffs = np.asarray(intrinsics.get("dist_coeffs", []), dtype=np.float64).reshape(-1, 1)
    return camera_matrix, dist_coeffs


def load_detection(detections_dir: Path, snapshot_id: str, camera_id: str) -> dict[str, Any]:
    return read_json(detections_dir / snapshot_id / f"{camera_id}_detections.json")


def detection_map(detection: dict[str, Any]) -> dict[int, tuple[float, float]]:
    ids = detection.get("charuco_ids", [])
    corners = detection.get("charuco_corners", [])
    return {
        int(charuco_id): (float(corner[0]), float(corner[1]))
        for charuco_id, corner in zip(ids, corners)
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
        value = float(depth[y, x])
        if not math.isfinite(value) or value <= 0:
            return None
        depth_m = value / 1000.0
        if depth_m < min_depth_m or depth_m > max_depth_m:
            return None
        return depth_m

    radius = window // 2
    x0 = max(0, x - radius)
    x1 = min(w, x + radius + 1)
    y0 = max(0, y - radius)
    y1 = min(h, y + radius + 1)
    patch = depth[y0:y1, x0:x1].astype(np.float64)
    valid = patch[np.isfinite(patch) & (patch > 0)]
    if valid.size == 0:
        return None
    depths_m = valid / 1000.0
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


def transform_point(transform: np.ndarray, point: np.ndarray) -> np.ndarray:
    homogeneous = np.ones(4, dtype=np.float64)
    homogeneous[:3] = point
    return (transform @ homogeneous)[:3]


def validate_snapshot(
    snapshot_id: str,
    args: argparse.Namespace,
    transform_cam1_from_cam0: np.ndarray,
) -> SnapshotValidation:
    snapshot_dir = args.snapshots / snapshot_id
    try:
        det0 = detection_map(load_detection(args.detections, snapshot_id, args.camera0))
        det1 = detection_map(load_detection(args.detections, snapshot_id, args.camera1))
        depth0 = np.load(snapshot_dir / args.camera0 / "aligned_depth.npy")
        depth1 = np.load(snapshot_dir / args.camera1 / "aligned_depth.npy")
        k0, d0 = load_intrinsics(snapshot_dir, args.camera0)
        k1, d1 = load_intrinsics(snapshot_dir, args.camera1)
    except Exception as exc:
        return SnapshotValidation(
            snapshot_id=snapshot_id,
            status="fail",
            reason=str(exc),
            common_corners=0,
            valid_points=0,
            rejected_depth_points=0,
            rejected_outlier_points=0,
            mean_mm=None,
            median_mm=None,
            rmse_mm=None,
            p95_mm=None,
            mean_abs_xyz_mm=None,
            points=[],
        )

    common_ids = sorted(set(det0) & set(det1))
    points: list[DepthPoint] = []
    rejected_depth = 0
    rejected_outlier = 0

    for charuco_id in common_ids:
        pixel0 = det0[charuco_id]
        pixel1 = det1[charuco_id]
        depth0_m = sample_depth_m(
            depth0,
            pixel0,
            args.min_depth_m,
            args.max_depth_m,
            args.depth_window,
        )
        depth1_m = sample_depth_m(
            depth1,
            pixel1,
            args.min_depth_m,
            args.max_depth_m,
            args.depth_window,
        )
        if depth0_m is None or depth1_m is None:
            rejected_depth += 1
            continue

        p0_cam0 = backproject_color_pixel(pixel0, depth0_m, k0, d0)
        p1_cam1 = backproject_color_pixel(pixel1, depth1_m, k1, d1)
        p0_cam1 = transform_point(transform_cam1_from_cam0, p0_cam0)
        error_xyz_mm = (p0_cam1 - p1_cam1) * 1000.0
        error_mm = float(np.linalg.norm(error_xyz_mm))
        if error_mm > args.max_error_mm:
            rejected_outlier += 1
            continue

        points.append(
            DepthPoint(
                snapshot_id=snapshot_id,
                charuco_id=charuco_id,
                p0_cam0=p0_cam0,
                p0_cam1=p0_cam1,
                p1_cam1=p1_cam1,
                pixel0=pixel0,
                pixel1=pixel1,
                depth0_m=depth0_m,
                depth1_m=depth1_m,
                error_xyz_mm=error_xyz_mm,
                error_mm=error_mm,
            )
        )

    metrics = compute_metrics(points)
    status = "ok" if len(points) >= args.min_valid_points else "fail"
    reason = None
    if status != "ok":
        reason = f"valid_points {len(points)} < min_valid_points {args.min_valid_points}"

    return SnapshotValidation(
        snapshot_id=snapshot_id,
        status=status,
        reason=reason,
        common_corners=len(common_ids),
        valid_points=len(points),
        rejected_depth_points=rejected_depth,
        rejected_outlier_points=rejected_outlier,
        mean_mm=metrics["mean_mm"],
        median_mm=metrics["median_mm"],
        rmse_mm=metrics["rmse_mm"],
        p95_mm=metrics["p95_mm"],
        mean_abs_xyz_mm=metrics["mean_abs_xyz_mm"],
        points=points,
    )


def compute_metrics(points: list[DepthPoint]) -> dict[str, Any]:
    if not points:
        return {
            "mean_mm": None,
            "median_mm": None,
            "rmse_mm": None,
            "p95_mm": None,
            "mean_abs_xyz_mm": None,
        }

    errors = np.asarray([point.error_mm for point in points], dtype=np.float64)
    error_xyz = np.asarray([point.error_xyz_mm for point in points], dtype=np.float64)
    return {
        "mean_mm": float(np.mean(errors)),
        "median_mm": float(np.median(errors)),
        "rmse_mm": float(np.sqrt(np.mean(errors * errors))),
        "p95_mm": float(np.percentile(errors, 95)),
        "mean_abs_xyz_mm": [float(v) for v in np.mean(np.abs(error_xyz), axis=0)],
    }


def aggregate_global(validations: list[SnapshotValidation]) -> dict[str, Any]:
    points = [
        point
        for validation in validations
        if validation.status == "ok"
        for point in validation.points
    ]
    metrics = compute_metrics(points)
    if points:
        error_xyz = np.asarray([point.error_xyz_mm for point in points], dtype=np.float64)
        metrics["mean_xyz_mm"] = [float(v) for v in np.mean(error_xyz, axis=0)]
        metrics["std_xyz_mm"] = [float(v) for v in np.std(error_xyz, axis=0)]
    else:
        metrics["mean_xyz_mm"] = None
        metrics["std_xyz_mm"] = None
    metrics["valid_points"] = len(points)
    metrics["ok_snapshots"] = sum(1 for validation in validations if validation.status == "ok")
    metrics["failed_snapshots"] = sum(1 for validation in validations if validation.status != "ok")
    return metrics


def validation_to_json(validation: SnapshotValidation, include_points: bool = True) -> dict[str, Any]:
    data = {
        "snapshot_id": validation.snapshot_id,
        "status": validation.status,
        "reason": validation.reason,
        "common_corners": validation.common_corners,
        "valid_points": validation.valid_points,
        "rejected_depth_points": validation.rejected_depth_points,
        "rejected_outlier_points": validation.rejected_outlier_points,
        "mean_mm": validation.mean_mm,
        "median_mm": validation.median_mm,
        "rmse_mm": validation.rmse_mm,
        "p95_mm": validation.p95_mm,
        "mean_abs_xyz_mm": validation.mean_abs_xyz_mm,
    }
    if include_points:
        data["points"] = [point_to_json(point) for point in validation.points]
    return data


def point_to_json(point: DepthPoint) -> dict[str, Any]:
    return {
        "snapshot_id": point.snapshot_id,
        "charuco_id": point.charuco_id,
        "pixel0": list(point.pixel0),
        "pixel1": list(point.pixel1),
        "depth0_m": point.depth0_m,
        "depth1_m": point.depth1_m,
        "p0_cam0_m": point.p0_cam0.tolist(),
        "p0_cam1_m": point.p0_cam1.tolist(),
        "p1_cam1_m": point.p1_cam1.tolist(),
        "error_x_mm": float(point.error_xyz_mm[0]),
        "error_y_mm": float(point.error_xyz_mm[1]),
        "error_z_mm": float(point.error_xyz_mm[2]),
        "error_mm": point.error_mm,
    }


def write_csv(path: Path, validations: list[SnapshotValidation]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.DictWriter(
            file,
            fieldnames=[
                "snapshot_id",
                "status",
                "charuco_id",
                "pixel0_u",
                "pixel0_v",
                "pixel1_u",
                "pixel1_v",
                "depth0_m",
                "depth1_m",
                "error_x_mm",
                "error_y_mm",
                "error_z_mm",
                "error_mm",
            ],
        )
        writer.writeheader()
        for validation in validations:
            for point in validation.points:
                writer.writerow(
                    {
                        "snapshot_id": validation.snapshot_id,
                        "status": validation.status,
                        "charuco_id": point.charuco_id,
                        "pixel0_u": point.pixel0[0],
                        "pixel0_v": point.pixel0[1],
                        "pixel1_u": point.pixel1[0],
                        "pixel1_v": point.pixel1[1],
                        "depth0_m": point.depth0_m,
                        "depth1_m": point.depth1_m,
                        "error_x_mm": float(point.error_xyz_mm[0]),
                        "error_y_mm": float(point.error_xyz_mm[1]),
                        "error_z_mm": float(point.error_xyz_mm[2]),
                        "error_mm": point.error_mm,
                    }
                )


def print_table(validations: list[SnapshotValidation]) -> None:
    headers = ["snapshot_id", "valid_points", "mean_mm", "median_mm", "rmse_mm", "p95_mm", "status"]
    rows = []
    for validation in validations:
        rows.append(
            [
                validation.snapshot_id,
                str(validation.valid_points),
                fmt(validation.mean_mm),
                fmt(validation.median_mm),
                fmt(validation.rmse_mm),
                fmt(validation.p95_mm),
                validation.status if validation.reason is None else f"{validation.status}: {validation.reason}",
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
    extrinsics = read_json(args.extrinsics)
    transform = np.asarray(
        extrinsics["T_kinect1_color_from_kinect0_color"],
        dtype=np.float64,
    )
    if transform.shape != (4, 4):
        raise ValueError(f"Expected 4x4 extrinsic transform, got {transform.shape}")

    used_snapshots = extrinsics.get("used_snapshots", [])
    validations = [
        validate_snapshot(snapshot_id, args, transform)
        for snapshot_id in used_snapshots
    ]
    global_metrics = aggregate_global(validations)

    output = {
        "extrinsics_file": str(args.extrinsics),
        "T_kinect1_color_from_kinect0_color": transform.tolist(),
        "settings": {
            "camera0": args.camera0,
            "camera1": args.camera1,
            "min_valid_points": args.min_valid_points,
            "min_depth_m": args.min_depth_m,
            "max_depth_m": args.max_depth_m,
            "max_error_mm": args.max_error_mm,
            "depth_window": args.depth_window,
        },
        "global": global_metrics,
        "snapshots": [validation_to_json(validation) for validation in validations],
    }

    args.out_json.parent.mkdir(parents=True, exist_ok=True)
    args.out_json.write_text(json.dumps(output, indent=2), encoding="utf-8")
    write_csv(args.out_csv, validations)

    print_table(validations)
    print()
    print("Global depth validation:")
    print(f"valid_points: {global_metrics['valid_points']}")
    print(f"ok_snapshots: {global_metrics['ok_snapshots']}")
    print(f"failed_snapshots: {global_metrics['failed_snapshots']}")
    print(f"mean_mm: {fmt(global_metrics['mean_mm'])}")
    print(f"median_mm: {fmt(global_metrics['median_mm'])}")
    print(f"rmse_mm: {fmt(global_metrics['rmse_mm'])}")
    print(f"p95_mm: {fmt(global_metrics['p95_mm'])}")
    print(f"mean_abs_xyz_mm: {global_metrics['mean_abs_xyz_mm']}")
    print(f"Saved JSON: {args.out_json}")
    print(f"Saved CSV: {args.out_csv}")

    return 0 if global_metrics["ok_snapshots"] > 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
