"""Screen all HO-Cap camera pairs against an imported hand track."""

from __future__ import annotations

import argparse
import json
from itertools import combinations
from pathlib import Path

import cv2
import numpy as np

from viki.benchmark.hocap import (
    K_and_dist,
    convergence_angle_deg,
    load_hocap_extrinsics,
    load_hocap_intrinsics,
    world_camera_pose_to_rvec_tvec,
)
from viki.contracts import PreparedFrame
from viki.perception.backends.mediapipe import MediaPipeHandBackend


def screen_sequence(root: Path, episode: Path) -> dict:
    metadata = json.loads((episode / "meta.json").read_text())
    sequence = metadata["hocap_sequence"]
    source = root / metadata["hocap_subject"] / sequence
    calibration = root / "calibration"
    extrinsics = load_hocap_extrinsics(
        calibration / "extrinsics" / metadata["hocap_extrinsics_file"]
    )
    with np.load(episode / "gt/joints3d_world.npz", allow_pickle=False) as labels:
        world = np.asarray(labels["joints_world"], dtype=np.float64)
        valid = np.asarray(labels["valid"], dtype=bool)
    total = int(valid.sum())
    wrist = world[:, 0]
    centre = np.nanmedian(wrist[valid[:, 0]], axis=0)
    depth_masks: dict[str, np.ndarray] = {}
    detection_masks: dict[str, np.ndarray] = {}
    camera_rows: dict[str, dict] = {}

    for serial in sorted(extrinsics.cam_to_master):
        intrinsics = load_hocap_intrinsics(calibration, serial)
        matrix, distortion = K_and_dist(intrinsics)
        rvec, tvec = world_camera_pose_to_rvec_tvec(
            extrinsics.world_from_camera(serial)
        )
        rotation = cv2.Rodrigues(rvec)[0]
        good_depth = np.zeros(valid.shape, dtype=bool)
        good_detection = np.zeros(len(world), dtype=bool)
        in_bounds = 0
        depth_present = 0
        detector_frames = 0
        detector = MediaPipeHandBackend(mode="video", models_dir="models")
        try:
            for frame in range(len(world)):
                depth = cv2.imread(
                    str(source / serial / f"depth_{frame:06d}.png"),
                    cv2.IMREAD_ANYDEPTH,
                )
                image = cv2.imread(str(source / serial / f"color_{frame:06d}.jpg"))
                if depth is None or image is None:
                    raise FileNotFoundError(f"missing frame {frame} from {serial}")
                projected, _ = cv2.projectPoints(
                    world[frame], rvec, tvec, matrix, distortion
                )
                projected = projected[:, 0, :]
                camera_z = (world[frame] @ rotation.T + tvec)[:, 2]
                for joint in np.flatnonzero(valid[frame]):
                    pixel_x, pixel_y = np.rint(projected[joint]).astype(int)
                    if not (
                        4 <= pixel_x < depth.shape[1] - 4
                        and 4 <= pixel_y < depth.shape[0] - 4
                        and camera_z[joint] > 0
                    ):
                        continue
                    in_bounds += 1
                    patch = depth[
                        pixel_y - 4:pixel_y + 5, pixel_x - 4:pixel_x + 5
                    ]
                    samples = patch[patch > 0]
                    if len(samples) == 0:
                        continue
                    depth_present += 1
                    difference_mm = float(np.median(samples) - camera_z[joint] * 1000)
                    good_depth[frame, joint] = abs(difference_mm) <= 50

                prepared = PreparedFrame(
                    rgb=cv2.cvtColor(image, cv2.COLOR_BGR2RGB),
                    depth_m=np.empty((0, 0), dtype=np.float32),
                    depth_K=None,
                    device_id=serial,
                    timestamp_us=frame * 33_333,
                )
                detection = detector.detect(prepared, "right")
                if detection is None:
                    continue
                detector_frames += 1
                observed = np.stack([detection.points[index] for index in range(21)])
                error_px = np.linalg.norm(observed - projected, axis=1)
                good_detection[frame] = np.median(error_px[valid[frame]]) < 25
        finally:
            detector.close()

        depth_masks[serial] = good_depth
        detection_masks[serial] = good_detection
        camera_rows[serial] = {
            "in_bounds_joints": in_bounds,
            "depth_present_joints": depth_present,
            "depth_within_50mm_joints": int(good_depth.sum()),
            "detector_frames": detector_frames,
            "detector_correct_frames": int(good_detection.sum()),
        }

    pairs = []
    for first, second in combinations(sorted(camera_rows), 2):
        both_depth = valid & depth_masks[first] & depth_masks[second]
        both_detected = detection_masks[first] & detection_masks[second]
        viable = both_depth & both_detected[:, None]
        angle = convergence_angle_deg(
            extrinsics.camera_centre_world(first),
            extrinsics.camera_centre_world(second),
            centre,
        )
        pairs.append({
            "cameras": [first, second],
            "baseline_mm": round(float(np.linalg.norm(
                extrinsics.camera_centre_world(first)
                - extrinsics.camera_centre_world(second)
            ) * 1000), 1),
            "convergence_deg": round(angle, 1),
            "both_depth_within_50mm_joints": int(both_depth.sum()),
            "both_detector_correct_frames": int(both_detected.sum()),
            "viable_joints": int(viable.sum()),
            "viable_fraction": round(float(viable.sum() / total), 4),
        })
    pairs.sort(key=lambda row: (-row["viable_joints"], -row["both_depth_within_50mm_joints"]))
    return {
        "sequence": sequence,
        "frames": len(world),
        "annotated_joints": total,
        "cameras": camera_rows,
        "pairs": pairs,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("episodes", nargs="+", type=Path)
    parser.add_argument("--root", type=Path, default=Path("data/raw_datasets/hocap"))
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    report = {
        "method": "GT colour reprojection; 9x9 median depth within 50 mm of camera-Z; MediaPipe JPEG video detection with median 2D error <25 px",
        "sequences": [screen_sequence(args.root, episode) for episode in args.episodes],
    }
    serialized = json.dumps(report, indent=2)
    if args.out is not None:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(serialized + "\n")
    for sequence in report["sequences"]:
        print(sequence["sequence"])
        for pair in sequence["pairs"][:8]:
            print(pair)


if __name__ == "__main__":
    main()
