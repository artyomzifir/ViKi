"""Resume the compact, eight-view HO-Cap oracle-mask object benchmark."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
import time
from pathlib import Path


def write_status(path: Path, status: dict) -> None:
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(json.dumps(status, indent=2) + "\n")
    temporary.replace(path)


def run_batch(archives: Path, plan_path: Path, dataset: Path, *,
              poll_seconds: float = 30.0, idle_timeout_minutes: float = 60.0) -> None:
    plan = json.loads(plan_path.read_text())
    hand_status_path = dataset / "batch_status.json"
    object_status_path = dataset / "object_batch_status.json"
    if object_status_path.exists():
        status = json.loads(object_status_path.read_text())
        if status.get("revision") != plan["revision"]:
            raise ValueError("object batch status uses a different source revision")
    else:
        status = {
            "revision": plan["revision"],
            "profile": "hocap-oracle-v1",
            "mask_source": "published HO-Cap labels, not an independent segmenter",
            "sequences": {},
        }
    pending = {item["id"] for item in plan["sequences"]}
    last_progress = time.monotonic()
    while pending:
        if not hand_status_path.is_file():
            raise FileNotFoundError(hand_status_path)
        hand_status = json.loads(hand_status_path.read_text())["sequences"]
        progress = False
        for item in plan["sequences"]:
            sequence_id = item["id"]
            if sequence_id not in pending:
                continue
            source = hand_status.get(sequence_id, {}).get("eight", {})
            if source.get("status") == "failed":
                status["sequences"][sequence_id] = {
                    "status": "blocked", "reason": "eight-view hand benchmark failed",
                }
                pending.remove(sequence_id)
                progress = True
                write_status(object_status_path, status)
                continue
            if source.get("status") != "complete":
                continue
            episode = dataset / source["episode_name"]
            output = episode / "intermediates/segmentation/hocap-oracle-v1"
            model = output / "object_models.npz"
            evaluation = output / "evaluation.json"
            reference = episode / "gt/object_poses_world.npz"
            if not (model.is_file() and evaluation.is_file() and reference.is_file()):
                if shutil.disk_usage(dataset).free < 10_000_000_000:
                    print("less than 10 GB free; waiting before object tracking", flush=True)
                    break
                started = time.monotonic()
                print(f"object tracking {sequence_id}", flush=True)
                command = [
                    sys.executable, "/app/scripts/run_hocap_object_oracle.py",
                    "--archives", str(archives), "--episode", str(episode),
                    "--out", str(output),
                ]
                result = subprocess.run(command, check=False)
                if result.returncode != 0:
                    status["sequences"][sequence_id] = {
                        "status": "failed", "returncode": result.returncode,
                    }
                    pending.remove(sequence_id)
                    progress = True
                    write_status(object_status_path, status)
                    continue
                elapsed = round(time.monotonic() - started, 3)
            else:
                elapsed = 0.0
            if not (model.is_file() and evaluation.is_file() and reference.is_file()):
                raise RuntimeError(f"incomplete compact artifacts for {sequence_id}")
            score = json.loads(evaluation.read_text())
            status["sequences"][sequence_id] = {
                "status": "complete", "episode_name": episode.name,
                "elapsed_seconds": elapsed,
                "objects": {
                    name: {
                        "coverage": values["coverage"],
                        "translation_mean_mm": values["translation_mean_mm"],
                        "rotation_mean_deg": values["rotation_mean_deg"],
                    }
                    for name, values in score["objects"].items()
                },
            }
            pending.remove(sequence_id)
            progress = True
            write_status(object_status_path, status)
            print(f"object complete {sequence_id} ({len(pending)} remaining)", flush=True)
        if not pending:
            break
        if progress:
            last_progress = time.monotonic()
        elif time.monotonic() - last_progress > idle_timeout_minutes * 60:
            raise RuntimeError(f"object batch idle for {idle_timeout_minutes} minutes; pending: {sorted(pending)}")
        time.sleep(poll_seconds)
    failed = [name for name, result in status["sequences"].items()
              if result["status"] != "complete"]
    print(f"object batch finished: {len(status['sequences'])} scenes, failures={failed}", flush=True)
    if failed:
        raise RuntimeError(f"object benchmark failed for {len(failed)} scenes")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--archives", required=True, type=Path)
    parser.add_argument("--plan", required=True, type=Path)
    parser.add_argument("--dataset", required=True, type=Path)
    parser.add_argument("--poll-seconds", type=float, default=30.0)
    parser.add_argument("--idle-timeout-minutes", type=float, default=60.0)
    args = parser.parse_args()
    run_batch(args.archives, args.plan, args.dataset,
              poll_seconds=args.poll_seconds,
              idle_timeout_minutes=args.idle_timeout_minutes)


if __name__ == "__main__":
    main()
