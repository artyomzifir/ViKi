#!/usr/bin/env python3
"""
Estimate RGB camera-to-camera extrinsics from paired ChArUco detections.

Input:
  - ViKi pair snapshots with per-camera metadata.json containing color_intrinsics.
  - Output of tools/calibration/01_detect_charuco.py.

Output:
  - A JSON file containing T_kinect1_color_from_kinect0_color and per-snapshot
    solvePnP/reprojection metrics.
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
except Exception:  # pragma: no cover - exercised only on minimal installs
    Rotation = None


# ChArUco board defaults. Measure your printed board and override via CLI if
# needed; translation scale depends directly on square size.
SQUARES_X = 8
SQUARES_Y = 10
SQUARE_LENGTH_M = 0.050
MARKER_LENGTH_M = 0.0375
ARUCO_DICTIONARY = cv2.aruco.DICT_5X5_1000

CAMERA0_ID = "kinect_0"
CAMERA1_ID = "kinect_1"


@dataclass
class CameraPose:
    camera_id: str
    serial_number: str | None
    rvec: np.ndarray
    tvec: np.ndarray
    transform: np.ndarray
    reprojection_error_px: float
    corners_count: int
    inliers_count: int | None


@dataclass
class SnapshotEstimate:
    snapshot_id: str
    status: str
    reason: str | None
    cam0: CameraPose | None
    cam1: CameraPose | None
    transform_cam1_from_cam0: np.ndarray | None
    baseline_m: float | None
    consistency_translation_delta_mm: float | None = None
    consistency_rotation_delta_deg: float | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Estimate kinect_1 color-from-kinect_0 color extrinsics using ChArUco solvePnP."
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
        default=Path("data/calibration/extrinsics_rgb.json"),
    )
    parser.add_argument("--square-length-m", type=float, default=SQUARE_LENGTH_M)
    parser.add_argument("--marker-length-m", type=float, default=MARKER_LENGTH_M)
    parser.add_argument("--min-corners", type=int, default=30)
    parser.add_argument("--max-reproj-px", type=float, default=2.0)
    parser.add_argument("--ransac-reproj-px", type=float, default=3.0)
    parser.add_argument("--max-translation-deviation-mm", type=float, default=50.0)
    parser.add_argument("--max-rotation-deviation-deg", type=float, default=5.0)
    parser.add_argument("--camera0", default=CAMERA0_ID)
    parser.add_argument("--camera1", default=CAMERA1_ID)
    return parser.parse_args()


def get_aruco_dictionary():
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    return cv2.aruco.Dictionary_get(ARUCO_DICTIONARY)


def make_charuco_board(square_length_m: float, marker_length_m: float):
    dictionary = get_aruco_dictionary()
    if hasattr(cv2.aruco, "CharucoBoard"):
        try:
            return cv2.aruco.CharucoBoard(
                (SQUARES_X, SQUARES_Y),
                square_length_m,
                marker_length_m,
                dictionary,
            )
        except TypeError:
            pass
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        return cv2.aruco.CharucoBoard_create(
            SQUARES_X,
            SQUARES_Y,
            square_length_m,
            marker_length_m,
            dictionary,
        )
    raise RuntimeError("This OpenCV build does not expose ChArUco board creation.")


def iter_pair_snapshots(snapshots_dir: Path, camera0: str, camera1: str) -> list[Path]:
    if not snapshots_dir.exists():
        return []

    pair_dirs = []
    for path in sorted(snapshots_dir.iterdir()):
        if not path.is_dir() or not path.name.startswith("pair_"):
            continue
        if (path / camera0 / "metadata.json").exists() and (path / camera1 / "metadata.json").exists():
            pair_dirs.append(path)
    return pair_dirs


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_intrinsics(snapshot_dir: Path, camera_id: str) -> tuple[np.ndarray, np.ndarray, str | None]:
    metadata = read_json(snapshot_dir / camera_id / "metadata.json")
    intrinsics = metadata.get("color_intrinsics")
    if not intrinsics:
        raise ValueError("metadata.json has no color_intrinsics")

    camera_matrix = np.asarray(intrinsics.get("camera_matrix"), dtype=np.float64)
    if camera_matrix.shape != (3, 3):
        raise ValueError(f"invalid camera_matrix shape {camera_matrix.shape}")

    dist_coeffs = np.asarray(intrinsics.get("dist_coeffs", []), dtype=np.float64).reshape(-1, 1)
    return camera_matrix, dist_coeffs, metadata.get("serial_number")


def load_detection(detections_dir: Path, snapshot_id: str, camera_id: str) -> dict[str, Any]:
    path = detections_dir / snapshot_id / f"{camera_id}_detections.json"
    if not path.exists():
        raise ValueError(f"missing detection file {path}")
    return read_json(path)


def charuco_points_from_detection(
    board_corners: np.ndarray,
    detection: dict[str, Any],
) -> tuple[np.ndarray, np.ndarray]:
    ids = np.asarray(detection.get("charuco_ids", []), dtype=np.int32).reshape(-1)
    image_points = np.asarray(detection.get("charuco_corners", []), dtype=np.float64)
    if image_points.size == 0:
        image_points = image_points.reshape(0, 2)
    else:
        image_points = image_points.reshape(-1, 2)

    if len(ids) != len(image_points):
        raise ValueError(
            f"charuco_ids length {len(ids)} != charuco_corners length {len(image_points)}"
        )
    if np.any(ids < 0) or np.any(ids >= len(board_corners)):
        raise ValueError("charuco_ids contain values outside board chessboard corner range")

    object_points = np.asarray(board_corners[ids], dtype=np.float64).reshape(-1, 3)
    return object_points, image_points


def solve_camera_pose(
    camera_id: str,
    serial_number: str | None,
    object_points: np.ndarray,
    image_points: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
    ransac_reproj_px: float,
) -> CameraPose:
    if len(object_points) < 4:
        raise ValueError(f"need at least 4 ChArUco corners, got {len(object_points)}")

    obj = np.asarray(object_points, dtype=np.float64).reshape(-1, 3)
    img = np.asarray(image_points, dtype=np.float64).reshape(-1, 2)
    inliers = None

    if len(obj) >= 6:
        ok, rvec, tvec, inliers = cv2.solvePnPRansac(
            obj,
            img,
            camera_matrix,
            dist_coeffs,
            iterationsCount=100,
            reprojectionError=float(ransac_reproj_px),
            confidence=0.999,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise ValueError("solvePnPRansac failed")
        if inliers is not None and len(inliers) >= 4:
            idx = inliers.reshape(-1)
            ok, rvec, tvec = cv2.solvePnP(
                obj[idx],
                img[idx],
                camera_matrix,
                dist_coeffs,
                rvec,
                tvec,
                useExtrinsicGuess=True,
                flags=cv2.SOLVEPNP_ITERATIVE,
            )
            if not ok:
                raise ValueError("solvePnP refinement failed")
    else:
        ok, rvec, tvec = cv2.solvePnP(
            obj,
            img,
            camera_matrix,
            dist_coeffs,
            flags=cv2.SOLVEPNP_ITERATIVE,
        )
        if not ok:
            raise ValueError("solvePnP failed")

    transform = transform_from_rvec_tvec(rvec, tvec)
    reproj_error = reprojection_error_px(
        obj,
        img,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    return CameraPose(
        camera_id=camera_id,
        serial_number=serial_number,
        rvec=rvec.reshape(3),
        tvec=tvec.reshape(3),
        transform=transform,
        reprojection_error_px=reproj_error,
        corners_count=len(obj),
        inliers_count=None if inliers is None else int(len(inliers)),
    )


def transform_from_rvec_tvec(rvec: np.ndarray, tvec: np.ndarray) -> np.ndarray:
    rotation, _ = cv2.Rodrigues(np.asarray(rvec, dtype=np.float64).reshape(3, 1))
    transform = np.eye(4, dtype=np.float64)
    transform[:3, :3] = rotation
    transform[:3, 3] = np.asarray(tvec, dtype=np.float64).reshape(3)
    return transform


def reprojection_error_px(
    object_points: np.ndarray,
    image_points: np.ndarray,
    rvec: np.ndarray,
    tvec: np.ndarray,
    camera_matrix: np.ndarray,
    dist_coeffs: np.ndarray,
) -> float:
    projected, _ = cv2.projectPoints(
        object_points,
        rvec,
        tvec,
        camera_matrix,
        dist_coeffs,
    )
    projected = projected.reshape(-1, 2)
    errors = np.linalg.norm(projected - image_points.reshape(-1, 2), axis=1)
    return float(np.mean(errors))


def estimate_snapshot(
    snapshot_dir: Path,
    detections_dir: Path,
    board_corners: np.ndarray,
    args: argparse.Namespace,
) -> SnapshotEstimate:
    snapshot_id = snapshot_dir.name
    try:
        cam0_pose = estimate_camera_for_snapshot(
            snapshot_dir,
            detections_dir,
            board_corners,
            snapshot_id,
            args.camera0,
            args,
        )
        cam1_pose = estimate_camera_for_snapshot(
            snapshot_dir,
            detections_dir,
            board_corners,
            snapshot_id,
            args.camera1,
            args,
        )
    except Exception as exc:
        return SnapshotEstimate(
            snapshot_id=snapshot_id,
            status="fail",
            reason=str(exc),
            cam0=None,
            cam1=None,
            transform_cam1_from_cam0=None,
            baseline_m=None,
        )

    reproj_too_high = [
        pose.camera_id
        for pose in (cam0_pose, cam1_pose)
        if pose.reprojection_error_px > args.max_reproj_px
    ]
    if reproj_too_high:
        return SnapshotEstimate(
            snapshot_id=snapshot_id,
            status="reject",
            reason=f"reprojection error > {args.max_reproj_px}px for {', '.join(reproj_too_high)}",
            cam0=cam0_pose,
            cam1=cam1_pose,
            transform_cam1_from_cam0=None,
            baseline_m=None,
        )

    transform = cam1_pose.transform @ np.linalg.inv(cam0_pose.transform)
    baseline_m = float(np.linalg.norm(transform[:3, 3]))
    return SnapshotEstimate(
        snapshot_id=snapshot_id,
        status="ok",
        reason=None,
        cam0=cam0_pose,
        cam1=cam1_pose,
        transform_cam1_from_cam0=transform,
        baseline_m=baseline_m,
    )


def estimate_camera_for_snapshot(
    snapshot_dir: Path,
    detections_dir: Path,
    board_corners: np.ndarray,
    snapshot_id: str,
    camera_id: str,
    args: argparse.Namespace,
) -> CameraPose:
    detection = load_detection(detections_dir, snapshot_id, camera_id)
    object_points, image_points = charuco_points_from_detection(board_corners, detection)
    if len(object_points) < args.min_corners:
        raise ValueError(
            f"{camera_id} has {len(object_points)} ChArUco corners, "
            f"min_corners={args.min_corners}"
        )
    camera_matrix, dist_coeffs, serial_number = load_intrinsics(snapshot_dir, camera_id)
    return solve_camera_pose(
        camera_id=camera_id,
        serial_number=serial_number,
        object_points=object_points,
        image_points=image_points,
        camera_matrix=camera_matrix,
        dist_coeffs=dist_coeffs,
        ransac_reproj_px=args.ransac_reproj_px,
    )


def aggregate_transforms(estimates: list[SnapshotEstimate]) -> tuple[np.ndarray, str]:
    used = [estimate for estimate in estimates if estimate.status == "ok"]
    if not used:
        raise RuntimeError("No usable snapshots after filtering")

    transforms = [estimate.transform_cam1_from_cam0 for estimate in used]
    assert all(transform is not None for transform in transforms)
    translations = np.asarray([transform[:3, 3] for transform in transforms], dtype=np.float64)
    final_translation = np.median(translations, axis=0)

    if Rotation is not None:
        rotations = Rotation.from_matrix([transform[:3, :3] for transform in transforms])
        final_rotation = rotations.mean().as_matrix()
        method = "median_translation_scipy_rotation_mean"
    else:
        best = min(
            used,
            key=lambda estimate: estimate.cam0.reprojection_error_px + estimate.cam1.reprojection_error_px,
        )
        assert best.transform_cam1_from_cam0 is not None
        final_rotation = best.transform_cam1_from_cam0[:3, :3]
        method = "median_translation_best_reprojection_rotation"

    final_transform = np.eye(4, dtype=np.float64)
    final_transform[:3, :3] = final_rotation
    final_transform[:3, 3] = final_translation
    return final_transform, method


def rotation_delta_deg(rotation: np.ndarray, reference_rotation: np.ndarray) -> float:
    delta = rotation @ reference_rotation.T
    rvec, _ = cv2.Rodrigues(delta)
    return float(np.linalg.norm(rvec) * 180.0 / math.pi)


def apply_consistency_filter(estimates: list[SnapshotEstimate], args: argparse.Namespace) -> None:
    used = [estimate for estimate in estimates if estimate.status == "ok"]
    if len(used) < 3:
        return

    reference_transform, _method = aggregate_transforms(estimates)
    reference_translation = reference_transform[:3, 3]
    reference_rotation = reference_transform[:3, :3]

    for estimate in used:
        assert estimate.transform_cam1_from_cam0 is not None
        transform = estimate.transform_cam1_from_cam0
        translation_delta_mm = float(
            np.linalg.norm(transform[:3, 3] - reference_translation) * 1000.0
        )
        rotation_delta = rotation_delta_deg(transform[:3, :3], reference_rotation)
        estimate.consistency_translation_delta_mm = translation_delta_mm
        estimate.consistency_rotation_delta_deg = rotation_delta

        if translation_delta_mm > args.max_translation_deviation_mm:
            estimate.status = "reject"
            estimate.reason = (
                "translation consistency delta "
                f"{translation_delta_mm:.1f}mm > {args.max_translation_deviation_mm:.1f}mm"
            )
            continue
        if rotation_delta > args.max_rotation_deviation_deg:
            estimate.status = "reject"
            estimate.reason = (
                "rotation consistency delta "
                f"{rotation_delta:.2f}deg > {args.max_rotation_deviation_deg:.2f}deg"
            )


def rotation_spread_deg(estimates: list[SnapshotEstimate], final_transform: np.ndarray) -> dict[str, float | list[float]]:
    used = [estimate for estimate in estimates if estimate.status == "ok"]
    if not used:
        return {"mean_deg": 0.0, "std_deg": 0.0, "per_snapshot_deg": []}

    final_rotation = final_transform[:3, :3]
    angles = []
    for estimate in used:
        assert estimate.transform_cam1_from_cam0 is not None
        angles.append(rotation_delta_deg(estimate.transform_cam1_from_cam0[:3, :3], final_rotation))

    return {
        "mean_deg": float(np.mean(angles)),
        "std_deg": float(np.std(angles)),
        "per_snapshot_deg": angles,
    }


def translation_spread(estimates: list[SnapshotEstimate], final_transform: np.ndarray) -> dict[str, float | list[float]]:
    used = [estimate for estimate in estimates if estimate.status == "ok"]
    if not used:
        return {"std_mm": 0.0, "std_xyz_mm": [0.0, 0.0, 0.0], "per_snapshot_delta_mm": []}

    final_t = final_transform[:3, 3]
    translations = np.asarray(
        [estimate.transform_cam1_from_cam0[:3, 3] for estimate in used],
        dtype=np.float64,
    )
    std_xyz_mm = np.std(translations, axis=0) * 1000.0
    deltas_mm = np.linalg.norm(translations - final_t, axis=1) * 1000.0
    return {
        "std_mm": float(np.linalg.norm(std_xyz_mm)),
        "std_xyz_mm": [float(v) for v in std_xyz_mm],
        "per_snapshot_delta_mm": [float(v) for v in deltas_mm],
    }


def pose_to_json(pose: CameraPose | None) -> dict[str, Any] | None:
    if pose is None:
        return None
    return {
        "camera_id": pose.camera_id,
        "serial_number": pose.serial_number,
        "corners_count": pose.corners_count,
        "inliers_count": pose.inliers_count,
        "reprojection_error_px": pose.reprojection_error_px,
        "rvec": pose.rvec.tolist(),
        "tvec_m": pose.tvec.tolist(),
        "T_cam_from_board": pose.transform.tolist(),
    }


def estimate_to_json(estimate: SnapshotEstimate) -> dict[str, Any]:
    return {
        "snapshot_id": estimate.snapshot_id,
        "status": estimate.status,
        "reason": estimate.reason,
        "cam0": pose_to_json(estimate.cam0),
        "cam1": pose_to_json(estimate.cam1),
        "baseline_m": estimate.baseline_m,
        "consistency_translation_delta_mm": estimate.consistency_translation_delta_mm,
        "consistency_rotation_delta_deg": estimate.consistency_rotation_delta_deg,
        "T_kinect1_color_from_kinect0_color": (
            None
            if estimate.transform_cam1_from_cam0 is None
            else estimate.transform_cam1_from_cam0.tolist()
        ),
    }


def build_output_json(
    args: argparse.Namespace,
    estimates: list[SnapshotEstimate],
    final_transform: np.ndarray,
    aggregation_method: str,
) -> dict[str, Any]:
    used = [estimate for estimate in estimates if estimate.status == "ok"]
    rejected = [estimate for estimate in estimates if estimate.status != "ok"]
    reproj_errors = [
        pose.reprojection_error_px
        for estimate in used
        for pose in (estimate.cam0, estimate.cam1)
        if pose is not None
    ]
    baselines = [estimate.baseline_m for estimate in used if estimate.baseline_m is not None]
    t_spread = translation_spread(estimates, final_transform)
    r_spread = rotation_spread_deg(estimates, final_transform)

    serials = {}
    for estimate in used:
        if estimate.cam0 is not None:
            serials[estimate.cam0.camera_id] = estimate.cam0.serial_number
        if estimate.cam1 is not None:
            serials[estimate.cam1.camera_id] = estimate.cam1.serial_number

    inverse_transform = np.linalg.inv(final_transform)
    return {
        "board": {
            "squares_x": SQUARES_X,
            "squares_y": SQUARES_Y,
            "square_length_m": args.square_length_m,
            "marker_length_m": args.marker_length_m,
            "dictionary": "DICT_5X5_1000",
        },
        "cameras": {
            "camera0_id": args.camera0,
            "camera1_id": args.camera1,
            "serial_numbers": serials,
        },
        "aggregation_method": aggregation_method,
        "T_kinect1_color_from_kinect0_color": final_transform.tolist(),
        "T_kinect0_color_from_kinect1_color": inverse_transform.tolist(),
        "translation_m": final_transform[:3, 3].tolist(),
        "baseline_m": float(np.linalg.norm(final_transform[:3, 3])),
        "mean_reprojection_error_px": float(np.mean(reproj_errors)) if reproj_errors else None,
        "median_reprojection_error_px": float(np.median(reproj_errors)) if reproj_errors else None,
        "baseline_per_snapshot_m": [float(v) for v in baselines],
        "translation_std_mm": t_spread["std_mm"],
        "translation_std_xyz_mm": t_spread["std_xyz_mm"],
        "rotation_std_deg": r_spread["std_deg"],
        "rotation_mean_delta_deg": r_spread["mean_deg"],
        "used_snapshots": [estimate.snapshot_id for estimate in used],
        "rejected_snapshots": [
            {"snapshot_id": estimate.snapshot_id, "status": estimate.status, "reason": estimate.reason}
            for estimate in rejected
        ],
        "per_snapshot_metrics": [estimate_to_json(estimate) for estimate in estimates],
    }


def print_table(estimates: list[SnapshotEstimate]) -> None:
    headers = ["snapshot_id", "cam0_reproj_px", "cam1_reproj_px", "baseline_m", "status"]
    rows = []
    for estimate in estimates:
        cam0_reproj = "" if estimate.cam0 is None else f"{estimate.cam0.reprojection_error_px:.3f}"
        cam1_reproj = "" if estimate.cam1 is None else f"{estimate.cam1.reprojection_error_px:.3f}"
        baseline = "" if estimate.baseline_m is None else f"{estimate.baseline_m:.4f}"
        status = estimate.status if estimate.reason is None else f"{estimate.status}: {estimate.reason}"
        rows.append([estimate.snapshot_id, cam0_reproj, cam1_reproj, baseline, status])

    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows)) if rows else len(header)
        for index, header in enumerate(headers)
    ]
    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def print_final_summary(output: dict[str, Any]) -> None:
    print()
    print("Estimated T_kinect1_color_from_kinect0_color:")
    matrix = np.asarray(output["T_kinect1_color_from_kinect0_color"])
    with np.printoptions(precision=6, suppress=True):
        print(matrix)
    print(f"translation_m: {output['translation_m']}")
    print(f"baseline_m: {output['baseline_m']:.6f}")
    print(f"mean_reprojection_error_px: {output['mean_reprojection_error_px']:.4f}")
    print(f"median_reprojection_error_px: {output['median_reprojection_error_px']:.4f}")
    print(f"translation_std_mm: {output['translation_std_mm']:.3f}")
    print(f"rotation_std_deg: {output['rotation_std_deg']:.4f}")
    print(f"used_snapshots: {len(output['used_snapshots'])}")
    print(f"rejected_snapshots: {len(output['rejected_snapshots'])}")


def main() -> int:
    args = parse_args()

    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco is not available. Install opencv-contrib-python or a "
            "headless OpenCV build that includes aruco."
        )

    board = make_charuco_board(args.square_length_m, args.marker_length_m)
    board_corners = np.asarray(board.getChessboardCorners(), dtype=np.float64)

    pair_dirs = iter_pair_snapshots(args.snapshots, args.camera0, args.camera1)
    if not pair_dirs:
        print(f"No pair snapshots found in {args.snapshots}")
        return 1

    estimates = [
        estimate_snapshot(snapshot_dir, args.detections, board_corners, args)
        for snapshot_dir in pair_dirs
    ]
    print_table(estimates)

    apply_consistency_filter(estimates, args)
    if any(
        estimate.status == "reject" and estimate.consistency_translation_delta_mm is not None
        for estimate in estimates
    ):
        print()
        print("After transform consistency filtering:")
        print_table(estimates)

    final_transform, aggregation_method = aggregate_transforms(estimates)
    output = build_output_json(args, estimates, final_transform, aggregation_method)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(output, indent=2), encoding="utf-8")
    print_final_summary(output)
    print(f"Saved: {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
