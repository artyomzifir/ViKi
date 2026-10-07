"""Measure per-frame Kinect depth range bias against stereo ChArUco colour rays.

This is an offline diagnostic, not a camera calibration. It does not modify raw
recordings or the canonical Viewer cloud. Even corner IDs fit a scalar range
offset for each depth camera; odd IDs independently check that offset.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import cv2
import numpy as np

from viki.contracts import CalibrationExtrinsics, Episode
from viki.perception.k4a_offline import K4ACalibration


def _percentiles(values: list[float]) -> list[float] | None:
    if not values:
        return None
    return np.percentile(values, [50, 90]).round(3).tolist()


def _corners(frame: np.ndarray, detector) -> dict[int, tuple[float, float]]:
    points, ids, _, _ = detector.detectBoard(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY))
    if ids is None:
        return {}
    return {int(idx): tuple(map(float, point)) for idx, point in zip(ids.ravel(), points.reshape(-1, 2))}


def _triangulate(camera_a: dict, camera_b: dict, uv_a, uv_b):
    origins = []
    directions = []
    for camera, uv in ((camera_a, uv_a), (camera_b, uv_b)):
        ray = camera["cal"].color_pixel_to_ray(*uv)
        if ray is None:
            return None
        transform = camera["T"]
        origins.append(transform[:3, 3])
        direction = transform[:3, :3] @ ray
        directions.append(direction / np.linalg.norm(direction))
    matrix = np.column_stack((directions[0], -directions[1]))
    scales = np.linalg.lstsq(matrix, origins[1] - origins[0], rcond=None)[0]
    point_a = origins[0] + scales[0] * directions[0]
    point_b = origins[1] + scales[1] * directions[1]
    return (point_a + point_b) / 2, float(np.linalg.norm(point_a - point_b)) * 1000


def _depth_observation(camera: dict, uv, target_world: np.ndarray, depth: np.ndarray):
    transform = camera["T"]
    target_cam = transform[:3, :3].T @ (target_world - transform[:3, 3])
    mapped = camera["cal"].project_color_to_depth(*uv, float(target_cam[2]))
    if mapped is None:
        return None
    depth_u, depth_v = (int(round(value)) for value in mapped)
    if not (0 <= depth_v < depth.shape[0] and 0 <= depth_u < depth.shape[1]):
        return None
    raw_range = float(depth[depth_v, depth_u])
    if raw_range <= 0:
        return None
    point_a = camera["cal"].deproject_depth_px_to_color3d(depth_u, depth_v, 1000.0)
    point_b = camera["cal"].deproject_depth_px_to_color3d(depth_u, depth_v, 2000.0)
    if point_a is None or point_b is None:
        return None
    direction = (point_b - point_a) / 1000.0
    offset = point_a - 1000.0 * direction
    expected_range = np.dot(target_cam * 1000 - offset, direction) / np.dot(direction, direction)
    if not (200 <= expected_range <= 6000):
        return None
    return raw_range - expected_range, raw_range, direction, offset, transform


def measure_frame(cameras: dict[str, dict], frame_index: int, detector) -> dict:
    names = sorted(cameras)
    if len(names) != 2:
        raise ValueError("exactly two Kinect cameras are required")
    images = {}
    corners = {}
    depths = {}
    for name in names:
        camera = cameras[name]
        camera["cap"].set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        ok, image = camera["cap"].read()
        path = camera["depth_dir"] / f"{frame_index:06d}.npy"
        if not ok or not path.is_file():
            return {"frame": frame_index, "error": f"missing {name} RGB-D frame"}
        images[name] = image
        corners[name] = _corners(image, detector)
        depths[name] = np.load(path)

    observations = {name: {} for name in names}
    ray_gaps = []
    for corner_id in sorted(corners[names[0]].keys() & corners[names[1]].keys()):
        matched = _triangulate(
            cameras[names[0]], cameras[names[1]],
            corners[names[0]][corner_id], corners[names[1]][corner_id],
        )
        if matched is None:
            continue
        target, ray_gap = matched
        if ray_gap > 5:
            continue
        ray_gaps.append(ray_gap)
        for name in names:
            measured = _depth_observation(cameras[name], corners[name][corner_id], target, depths[name])
            if measured is not None:
                observations[name][corner_id] = measured

    biases = {}
    heldout_errors = {}
    for name in names:
        training = [item[0] for corner_id, item in observations[name].items() if corner_id % 2 == 0]
        testing = [item[0] for corner_id, item in observations[name].items() if corner_id % 2]
        bias = float(np.median(training)) if len(training) >= 6 else None
        biases[name] = {"range_bias_mm": round(bias, 3) if bias is not None else None,
                        "train_corners": len(training), "test_corners": len(testing),
                        "heldout_abs_error_mm": _percentiles([abs(value - bias) for value in testing]) if bias is not None else None}
        heldout_errors[name] = {corner_id: value for corner_id, value in observations[name].items() if corner_id % 2}

    pair_raw = []
    pair_corrected = []
    if all(biases[name]["range_bias_mm"] is not None for name in names):
        for corner_id in heldout_errors[names[0]].keys() & heldout_errors[names[1]].keys():
            points = []
            corrected = []
            for name in names:
                bias_mm = biases[name]["range_bias_mm"]
                _, raw_range, direction, offset, transform = heldout_errors[name][corner_id]
                world_raw = transform[:3, :3] @ ((raw_range * direction + offset) / 1000) + transform[:3, 3]
                world_corrected = transform[:3, :3] @ (((raw_range - bias_mm) * direction + offset) / 1000) + transform[:3, 3]
                points.append(world_raw)
                corrected.append(world_corrected)
            pair_raw.append(float(np.linalg.norm(points[0] - points[1])) * 1000)
            pair_corrected.append(float(np.linalg.norm(corrected[0] - corrected[1])) * 1000)

    return {"frame": frame_index, "shared_rgb_corners": len(ray_gaps),
            "stereo_color_ray_gap_mm": _percentiles(ray_gaps), "cameras": biases,
            "heldout_pair_raw_mm": _percentiles(pair_raw),
            "heldout_pair_corrected_mm": _percentiles(pair_corrected)}


def diagnose(episode: Episode, frames: list[int]) -> dict:
    raw = episode.raw_dir
    intr = json.loads((raw / "intrinsics.json").read_text())
    extr = json.loads((raw / "extrinsics.json").read_text())
    meta = json.loads(episode.meta_path.read_text())
    preset = meta.get("calibration_preset")
    if not preset:
        raise ValueError("episode has no calibration preset")
    board_config = json.loads((Path("data/calibrations") / preset / "extrinsics.json").read_text())["board"]
    board = cv2.aruco.CharucoBoard(
        tuple(board_config["board_size"]), board_config["square_size"],
        board_config["marker_size"], cv2.aruco.getPredefinedDictionary(board_config["aruco_dict"]),
    )
    detector = cv2.aruco.CharucoDetector(board)
    cameras = {}
    try:
        for video in sorted(raw.glob("*.mp4")):
            name = video.stem
            if name not in intr or name not in extr:
                continue
            calibration = K4ACalibration.from_episode(raw, name, meta)
            if calibration is None:
                raise ValueError(f"missing Kinect SDK calibration for {name}")
            cameras[name] = {
                "cal": calibration, "cap": cv2.VideoCapture(str(video)),
                "depth_dir": raw / f"{name}_depth",
                "T": CalibrationExtrinsics(
                    rvec=np.asarray(extr[name]["rvec"], np.float64),
                    tvec=np.asarray(extr[name]["tvec"], np.float64),
                ).transform_matrix,
            }
        return {"episode": episode.id, "method": "even ChArUco IDs fit; odd IDs test",
                "frames": [measure_frame(cameras, index, detector) for index in frames]}
    finally:
        for camera in cameras.values():
            camera["cap"].release()
            camera["cal"].close()


def validated_bias_series(report: dict, n_frames: int) -> dict[str, list[float]]:
    """Interpolate only densely sampled, held-out-validated board measurements."""
    rows = report["frames"]
    frame_ids = np.array([row["frame"] for row in rows], dtype=int)
    if (len(frame_ids) < 2 or frame_ids[0] != 0 or frame_ids[-1] < n_frames - 20
            or np.any(np.diff(frame_ids) <= 0) or np.max(np.diff(frame_ids)) > 20):
        raise ValueError("corrected cloud needs ordered board samples every 20 frames or less")
    camera_names = sorted(rows[0]["cameras"])
    for row in rows:
        raw_pair = row["heldout_pair_raw_mm"]
        corrected_pair = row["heldout_pair_corrected_mm"]
        if (row["shared_rgb_corners"] < 12 or raw_pair is None or corrected_pair is None
                or corrected_pair[0] > 5 or corrected_pair[0] >= raw_pair[0]):
            raise ValueError(f"board correction fails independent check at frame {row['frame']}")
        for name in camera_names:
            reading = row["cameras"][name]
            bias = reading["range_bias_mm"]
            if (bias is None or not 0 <= bias <= 30 or reading["test_corners"] < 5
                    or reading["heldout_abs_error_mm"][0] > 5):
                raise ValueError(f"unreliable depth estimate for {name} at frame {row['frame']}")
    return {
        name: np.interp(np.arange(n_frames), frame_ids,
                        [row["cameras"][name]["range_bias_mm"] for row in rows]).round(3).tolist()
        for name in camera_names
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episode", type=Path)
    parser.add_argument("--frames", type=int, nargs="+", required=True)
    parser.add_argument("--out", type=Path)
    parser.add_argument("--corrected-cloud", action="store_true",
                        help="write an experimental, separate cloud after validation")
    args = parser.parse_args()
    result = diagnose(Episode(args.episode), args.frames)
    output = json.dumps(result, indent=2)
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(output + "\n")
    else:
        print(output)
    if args.corrected_cloud:
        from viki.perception.cloud import build_cloud

        episode = Episode(args.episode)
        n_frames = len(list((episode.raw_dir / "kinect_0_depth").glob("*.npy")))
        biases = validated_bias_series(result, n_frames)
        target = episode.root / "cloud_board_corrected"
        build_cloud(episode, depth_biases_by_frame=biases, output_dir=target)
        print(f"Experimental corrected cloud: {target}")


if __name__ == "__main__":
    main()
