#!/usr/bin/env python3
"""
Guided paired snapshot capture with audible countdown.

Run this on the host while the ViKi FastAPI server is running. The script gives
you time to move the ChArUco board, counts down, captures one pair snapshot via
the API, and repeats until the requested dataset size is reached.
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path
from typing import Any


DEFAULT_SERVER = "http://localhost:8000"
DEFAULT_ROOT_DIR = "data/datasets/rig_20260623_moving_board/snapshots"
DEFAULT_DEVICE_IDS = ("kinect_0", "kinect_1")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Capture a moving-board pair dataset with voice/beep countdown."
    )
    parser.add_argument("--server", default=DEFAULT_SERVER)
    parser.add_argument("--root-dir", default=DEFAULT_ROOT_DIR)
    parser.add_argument("--count", type=int, default=50)
    parser.add_argument(
        "--move-sec",
        type=float,
        default=3.0,
        help="Time to move the board before each countdown.",
    )
    parser.add_argument(
        "--countdown-sec",
        type=int,
        default=3,
        help="Audible countdown length before capture.",
    )
    parser.add_argument("--device-ids", nargs="+", default=list(DEFAULT_DEVICE_IDS))
    parser.add_argument("--retry-attempts", type=int, default=5)
    parser.add_argument("--retry-delay-sec", type=float, default=1.0)
    parser.add_argument("--aligned-depth", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument("--save", action=argparse.BooleanOptionalAction, default=True)
    parser.add_argument(
        "--sound",
        choices=("auto", "voice", "bell", "off"),
        default="auto",
        help="auto uses spd-say/espeak when available, else terminal bell.",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run countdowns without calling the API.",
    )
    return parser.parse_args()


def find_voice_command() -> list[str] | None:
    if shutil.which("spd-say"):
        return ["spd-say", "-w"]
    if shutil.which("espeak"):
        return ["espeak"]
    return None


def speak(message: str, mode: str) -> None:
    if mode == "off":
        return

    voice_cmd = find_voice_command() if mode in ("auto", "voice") else None
    if voice_cmd:
        try:
            subprocess.run(
                [*voice_cmd, message],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
            return
        except OSError:
            if mode == "voice":
                return

    if mode in ("auto", "bell"):
        print("\a", end="", flush=True)


def post_json(url: str, payload: dict[str, Any], timeout: float = 120.0) -> dict[str, Any]:
    body = json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        url,
        data=body,
        headers={"Content-Type": "application/json"},
        method="POST",
    )
    try:
        with urllib.request.urlopen(request, timeout=timeout) as response:
            return json.loads(response.read().decode("utf-8"))
    except urllib.error.HTTPError as exc:
        detail = exc.read().decode("utf-8", errors="replace")
        raise RuntimeError(f"HTTP {exc.code}: {detail}") from exc
    except urllib.error.URLError as exc:
        raise RuntimeError(f"Could not reach {url}: {exc}") from exc


def capture_pair(args: argparse.Namespace) -> dict[str, Any]:
    payload = {
        "device_ids": args.device_ids,
        "aligned_depth": args.aligned_depth,
        "save": args.save,
        "root_dir": args.root_dir,
    }
    if args.dry_run:
        return {
            "snapshot_id": "dry_run",
            "root": str(Path(args.root_dir) / "dry_run"),
            "device_ids": args.device_ids,
        }
    return post_json(f"{args.server.rstrip('/')}/api/capture/pair_snapshot", payload)


def countdown(args: argparse.Namespace, index: int) -> None:
    remaining = args.count - index
    print()
    print(f"[{index + 1}/{args.count}] Move board now. Remaining after this: {remaining - 1}")
    speak(f"Move board. Shot {index + 1} of {args.count}", args.sound)
    if args.move_sec > 0:
        time.sleep(args.move_sec)

    for value in range(args.countdown_sec, 0, -1):
        print(f"  {value}...", flush=True)
        speak(str(value), args.sound)
        time.sleep(1.0)
    print("  capture!", flush=True)
    speak("capture", args.sound)


def count_complete_pairs(root_dir: str, device_ids: list[str]) -> int:
    root = Path(root_dir)
    count = 0
    for pair in root.glob("pair_*"):
        if all((pair / device_id / "metadata.json").exists() for device_id in device_ids):
            count += 1
    return count


def print_next_steps(args: argparse.Namespace) -> None:
    dataset_root = Path(args.root_dir).parent
    detect_dir = dataset_root / "charuco_detect"
    extrinsics = dataset_root / "extrinsics_rgb.json"
    depth_validation = dataset_root / "depth_validation.json"
    depth_validation_csv = dataset_root / "depth_validation.csv"
    article = dataset_root / "article_rgbd_relative.json"

    print()
    print("Next commands:")
    print(
        "python3 tools/calibration/01_detect_charuco.py "
        f"--snapshots {args.root_dir} "
        f"--out {detect_dir}"
    )
    print(
        "python3 tools/calibration/02_estimate_rgb_extrinsics.py "
        f"--snapshots {args.root_dir} "
        f"--detections {detect_dir} "
        "--square-length-m 0.0482 "
        "--marker-length-m 0.03615 "
        f"--out {extrinsics}"
    )
    print(
        "python3 tools/calibration/03_validate_depth_points.py "
        f"--snapshots {args.root_dir} "
        f"--detections {detect_dir} "
        f"--extrinsics {extrinsics} "
        f"--out-json {depth_validation} "
        f"--out-csv {depth_validation_csv}"
    )
    print(
        "python3 tools/calibration/04_estimate_article_rgbd_board_world.py "
        f"--snapshots {args.root_dir} "
        f"--detections {detect_dir} "
        "--mode moving-board-relative "
        "--square-length-m 0.0482 "
        "--marker-length-m 0.03615 "
        f"--out {article}"
    )


def main() -> int:
    args = parse_args()
    if args.count < 1:
        raise SystemExit("--count must be >= 1")
    if args.retry_attempts < 1:
        raise SystemExit("--retry-attempts must be >= 1")

    print("Guided ViKi pair capture")
    print(f"server: {args.server}")
    print(f"root_dir: {args.root_dir}")
    print(f"device_ids: {', '.join(args.device_ids)}")
    print(f"count: {args.count}")
    print(f"sound: {args.sound}")
    print("Press Ctrl+C to stop.")
    speak("Starting guided capture", args.sound)

    saved = []
    errors = []
    try:
        for index in range(args.count):
            countdown(args, index)
            result = None
            last_error = None
            for attempt in range(1, args.retry_attempts + 1):
                try:
                    result = capture_pair(args)
                    break
                except Exception as exc:
                    last_error = str(exc)
                    print(
                        f"  capture failed attempt {attempt}/{args.retry_attempts}: "
                        f"{last_error}"
                    )
                    speak("capture failed", args.sound)
                    if attempt < args.retry_attempts:
                        time.sleep(args.retry_delay_sec)

            if result is None:
                errors.append({"index": index, "error": last_error})
                print("Stopping because capture failed after retries.")
                break

            saved.append(result)
            print(f"  saved: {result.get('root')}")
            speak("saved", args.sound)
    except KeyboardInterrupt:
        print()
        print("Stopped by user.")
        speak("stopped", args.sound)

    complete_pairs = count_complete_pairs(args.root_dir, args.device_ids)
    print()
    print(f"Saved this run: {len(saved)}/{args.count}")
    print(f"Complete pairs in root_dir: {complete_pairs}")
    if errors:
        print(f"Errors: {errors}")
    print_next_steps(args)
    return 0 if saved else 2


if __name__ == "__main__":
    raise SystemExit(main())
