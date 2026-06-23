#!/usr/bin/env python3
"""
Detect ChArUco board corners in paired ViKi snapshots.

This script only performs detection and writes visual overlays/JSON summaries.
It does not run solvePnP or estimate inter-camera extrinsics.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np


# ChArUco board parameters.
SQUARES_X = 8
SQUARES_Y = 10
SQUARE_LENGTH_M = 0.050
MARKER_LENGTH_M = 0.0375
ARUCO_DICTIONARY = cv2.aruco.DICT_5X5_1000

CAMERA_IDS = ("kinect_0", "kinect_1")
MIN_OK_CHARUCO_CORNERS = 4


@dataclass
class DetectionResult:
    snapshot_id: str
    camera_id: str
    serial_number: str | None
    markers_count: int
    charuco_corners_count: int
    charuco_ids: list[int]
    charuco_corners: list[list[float]]
    image_shape: list[int]
    ok: bool
    error: str | None = None

    def to_json(self) -> dict[str, Any]:
        return {
            "snapshot_id": self.snapshot_id,
            "camera_id": self.camera_id,
            "serial_number": self.serial_number,
            "markers_count": self.markers_count,
            "charuco_corners_count": self.charuco_corners_count,
            "charuco_ids": self.charuco_ids,
            "charuco_corners": self.charuco_corners,
            "image_shape": self.image_shape,
            "ok": self.ok,
            "error": self.error,
        }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Detect ArUco markers and ChArUco corners in ViKi pair snapshots."
    )
    parser.add_argument(
        "--snapshots",
        type=Path,
        default=Path("data/snapshots"),
        help="Directory containing pair snapshot folders.",
    )
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("data/calibration/charuco_detect"),
        help="Output directory for overlays and detection JSON files.",
    )
    return parser.parse_args()


def make_charuco_board():
    dictionary = get_aruco_dictionary()
    if hasattr(cv2.aruco, "CharucoBoard"):
        try:
            return cv2.aruco.CharucoBoard(
                (SQUARES_X, SQUARES_Y),
                SQUARE_LENGTH_M,
                MARKER_LENGTH_M,
                dictionary,
            )
        except TypeError:
            pass
    if hasattr(cv2.aruco, "CharucoBoard_create"):
        return cv2.aruco.CharucoBoard_create(
            SQUARES_X,
            SQUARES_Y,
            SQUARE_LENGTH_M,
            MARKER_LENGTH_M,
            dictionary,
        )
    raise RuntimeError("This OpenCV build does not expose ChArUco board creation.")


def get_aruco_dictionary():
    if hasattr(cv2.aruco, "getPredefinedDictionary"):
        return cv2.aruco.getPredefinedDictionary(ARUCO_DICTIONARY)
    return cv2.aruco.Dictionary_get(ARUCO_DICTIONARY)


def make_detector_parameters():
    if hasattr(cv2.aruco, "DetectorParameters"):
        return cv2.aruco.DetectorParameters()
    return cv2.aruco.DetectorParameters_create()


def make_aruco_detector(dictionary):
    params = make_detector_parameters()
    if hasattr(cv2.aruco, "ArucoDetector"):
        return cv2.aruco.ArucoDetector(dictionary, params)
    return None


def iter_pair_snapshots(snapshots_dir: Path) -> list[Path]:
    if not snapshots_dir.exists():
        return []

    pair_dirs = []
    for path in sorted(snapshots_dir.iterdir()):
        if not path.is_dir() or not path.name.startswith("pair_"):
            continue
        if all((path / camera_id / "color.jpg").exists() for camera_id in CAMERA_IDS):
            pair_dirs.append(path)
    return pair_dirs


def read_metadata(camera_dir: Path) -> dict[str, Any]:
    metadata_path = camera_dir / "metadata.json"
    if not metadata_path.exists():
        return {}
    return json.loads(metadata_path.read_text(encoding="utf-8"))


def detect_markers(gray: np.ndarray, dictionary, detector, detector_params):
    if detector is not None:
        marker_corners, marker_ids, rejected = detector.detectMarkers(gray)
        return marker_corners, marker_ids, rejected

    return cv2.aruco.detectMarkers(
        gray,
        dictionary,
        parameters=detector_params,
    )


def interpolate_charuco(gray: np.ndarray, board, marker_corners, marker_ids):
    if marker_ids is None or len(marker_ids) == 0:
        return None, None

    _retval, charuco_corners, charuco_ids = cv2.aruco.interpolateCornersCharuco(
        marker_corners,
        marker_ids,
        gray,
        board,
    )
    return charuco_corners, charuco_ids


def corners_to_list(charuco_corners) -> list[list[float]]:
    if charuco_corners is None:
        return []
    corners = np.asarray(charuco_corners, dtype=np.float64).reshape(-1, 2)
    return [[float(x), float(y)] for x, y in corners]


def ids_to_list(ids) -> list[int]:
    if ids is None:
        return []
    return [int(value) for value in np.asarray(ids).reshape(-1)]


def draw_overlay(
    image: np.ndarray,
    marker_corners,
    marker_ids,
    charuco_corners,
    charuco_ids,
    result: DetectionResult,
) -> np.ndarray:
    overlay = image.copy()

    if marker_ids is not None and len(marker_ids) > 0:
        cv2.aruco.drawDetectedMarkers(overlay, marker_corners, marker_ids)

    if charuco_ids is not None and len(charuco_ids) > 0:
        cv2.aruco.drawDetectedCornersCharuco(
            overlay,
            charuco_corners,
            charuco_ids,
            (0, 255, 0),
        )

    status = "OK" if result.ok else "FAIL"
    label = (
        f"{result.snapshot_id} {result.camera_id} {status} "
        f"markers={result.markers_count} charuco={result.charuco_corners_count}"
    )
    cv2.rectangle(overlay, (12, 12), (900, 54), (0, 0, 0), thickness=-1)
    cv2.putText(
        overlay,
        label,
        (24, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (255, 255, 255),
        2,
        cv2.LINE_AA,
    )
    return overlay


def detect_camera(
    snapshot_dir: Path,
    camera_id: str,
    out_dir: Path,
    board,
    dictionary,
    detector,
    detector_params,
) -> DetectionResult:
    snapshot_id = snapshot_dir.name
    camera_dir = snapshot_dir / camera_id
    metadata = read_metadata(camera_dir)
    serial_number = metadata.get("serial_number")
    image_path = camera_dir / "color.jpg"

    image = cv2.imread(str(image_path), cv2.IMREAD_COLOR)
    if image is None:
        result = DetectionResult(
            snapshot_id=snapshot_id,
            camera_id=camera_id,
            serial_number=serial_number,
            markers_count=0,
            charuco_corners_count=0,
            charuco_ids=[],
            charuco_corners=[],
            image_shape=[],
            ok=False,
            error=f"failed to read {image_path}",
        )
        write_detection_json(out_dir, snapshot_id, camera_id, result)
        return result

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    marker_corners, marker_ids, _rejected = detect_markers(
        gray,
        dictionary,
        detector,
        detector_params,
    )
    charuco_corners, charuco_ids = interpolate_charuco(
        gray,
        board,
        marker_corners,
        marker_ids,
    )

    markers_count = 0 if marker_ids is None else int(len(marker_ids))
    charuco_ids_list = ids_to_list(charuco_ids)
    charuco_corners_list = corners_to_list(charuco_corners)
    charuco_count = len(charuco_ids_list)
    result = DetectionResult(
        snapshot_id=snapshot_id,
        camera_id=camera_id,
        serial_number=serial_number,
        markers_count=markers_count,
        charuco_corners_count=charuco_count,
        charuco_ids=charuco_ids_list,
        charuco_corners=charuco_corners_list,
        image_shape=list(image.shape),
        ok=charuco_count >= MIN_OK_CHARUCO_CORNERS,
    )

    snapshot_out_dir = out_dir / snapshot_id
    snapshot_out_dir.mkdir(parents=True, exist_ok=True)
    overlay = draw_overlay(
        image,
        marker_corners,
        marker_ids,
        charuco_corners,
        charuco_ids,
        result,
    )
    cv2.imwrite(str(snapshot_out_dir / f"{camera_id}_overlay.jpg"), overlay)
    write_detection_json(out_dir, snapshot_id, camera_id, result)
    return result


def write_detection_json(
    out_dir: Path,
    snapshot_id: str,
    camera_id: str,
    result: DetectionResult,
) -> None:
    snapshot_out_dir = out_dir / snapshot_id
    snapshot_out_dir.mkdir(parents=True, exist_ok=True)
    path = snapshot_out_dir / f"{camera_id}_detections.json"
    path.write_text(json.dumps(result.to_json(), indent=2), encoding="utf-8")


def print_summary(results: list[DetectionResult]) -> None:
    headers = ["snapshot_id", "camera_id", "markers", "charuco", "status"]
    rows = [
        [
            result.snapshot_id,
            result.camera_id,
            str(result.markers_count),
            str(result.charuco_corners_count),
            "ok" if result.ok else "fail",
        ]
        for result in results
    ]
    widths = [
        max(len(headers[index]), *(len(row[index]) for row in rows)) if rows else len(header)
        for index, header in enumerate(headers)
    ]

    print("  ".join(header.ljust(widths[index]) for index, header in enumerate(headers)))
    print("  ".join("-" * width for width in widths))
    for row in rows:
        print("  ".join(value.ljust(widths[index]) for index, value in enumerate(row)))


def main() -> int:
    args = parse_args()

    if not hasattr(cv2, "aruco"):
        raise RuntimeError(
            "cv2.aruco is not available. Install opencv-contrib-python or a "
            "headless OpenCV build that includes aruco."
        )

    dictionary = get_aruco_dictionary()
    board = make_charuco_board()
    detector_params = make_detector_parameters()
    detector = make_aruco_detector(dictionary)

    pair_dirs = iter_pair_snapshots(args.snapshots)
    if not pair_dirs:
        print(f"No pair snapshots found in {args.snapshots}")
        return 1

    results = []
    for snapshot_dir in pair_dirs:
        for camera_id in CAMERA_IDS:
            result = detect_camera(
                snapshot_dir=snapshot_dir,
                camera_id=camera_id,
                out_dir=args.out,
                board=board,
                dictionary=dictionary,
                detector=detector,
                detector_params=detector_params,
            )
            results.append(result)

    print_summary(results)
    failed = sum(1 for result in results if not result.ok)
    print()
    print(
        f"Processed {len(pair_dirs)} pair snapshots, "
        f"{len(results) - failed}/{len(results)} camera images ok. "
        f"Output: {args.out}"
    )
    return 0 if failed == 0 else 2


if __name__ == "__main__":
    raise SystemExit(main())
