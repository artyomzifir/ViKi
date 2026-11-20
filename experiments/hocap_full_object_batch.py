"""Full-length HO-Cap object-aware retarget and official-mesh replay batch."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import shutil
import time
from pathlib import Path

import h5py
import numpy as np

from experiments.hocap_collision_batch import _options
from experiments.hocap_four_object_assisted_replay import replay
from viki.benchmark.hocap_provenance import require_hand_fit_clean, require_hand_fit_plan
from viki.contracts import Episode
from viki.retarget.run import retarget_episode


TRACK_RELATIVE_PATH = Path("intermediates/segmentation/hocap-oracle-v1/object_models.npz")


def _write_status(path: Path, status: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(status, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _plan_result(path: Path, robot: str, object_id: int, frame_count: int) -> dict:
    require_hand_fit_plan(path)
    with h5py.File(path) as plan:
        key = plan["robot_key"][()].decode()
        joints = np.asarray(plan["q"])
        metrics = json.loads(plan["metrics_json"][()].decode())
        timestamps = np.asarray(plan["timestamps"])
        configuration = json.loads(plan["config_json"][()].decode())
    grasp = metrics.get("object_grasp")
    if (key != robot or grasp is None or grasp.get("object_id") != object_id
            or joints.shape[0] != frame_count or len(timestamps) != frame_count
            or not np.isfinite(joints).all()
            or not configuration.get("object_track_path")):
        raise ValueError("plan is not a full-length object-conditioned trajectory")
    return {"status": "complete", "plan": str(path), "frames": frame_count,
            "metrics": metrics, "object_grasp": grasp}


def _run_scene(source: Path, destination: Path, robot: str, status: dict,
               status_path: Path) -> None:
    require_hand_fit_clean(source)
    require_hand_fit_plan(source / "plan.h5")
    track_path = source / TRACK_RELATIVE_PATH
    with np.load(track_path, allow_pickle=False) as tracks:
        object_ids = [int(identifier) for identifier in tracks["object_ids"]]
        frame_count = len(tracks["translation_world"])
    scene_status = status["episodes"][source.name]
    scene_status["frames"] = frame_count
    scene_status["objects"] = scene_status.get("objects", {})
    plans = {}
    for object_id in object_ids:
        identifier = str(object_id)
        target = destination / f"object_{object_id}"
        target.mkdir(parents=True, exist_ok=True)
        if not (target / "cln.npz").exists():
            os.link(source / "cln.npz", target / "cln.npz")
        if hashlib.sha256((target / "cln.npz").read_bytes()).digest() != hashlib.sha256((source / "cln.npz").read_bytes()).digest():
            raise ValueError(f"{target}: cln.npz differs from source episode")
        if not (target / "meta.json").exists():
            shutil.copy2(source / "meta.json", target / "meta.json")
        scene_status["objects"][identifier] = {"status": "running"}
        _write_status(status_path, status)
        print(f"{robot} {source.name} object {object_id}: retarget {frame_count} frames", flush=True)
        started = time.monotonic()
        try:
            if not (target / "plan.h5").is_file():
                options = _options(source / "plan.h5", track_path, object_id)
                options.update({"robot": robot, "object_grasp": True, "pose_source": "hand_fit",
                                "target_position_anchor": "wrist"})
                retarget_episode(Episode(target), options=options)
            result = _plan_result(target / "plan.h5", robot, object_id, frame_count)
            result["elapsed_s"] = time.monotonic() - started
            plans[object_id] = target / "plan.h5"
        except Exception as exc:
            result = {"status": "failed", "error": f"{type(exc).__name__}: {exc}",
                      "elapsed_s": time.monotonic() - started}
        scene_status["objects"][identifier] = result
        _write_status(status_path, status)
        print(f"{robot} {source.name} object {object_id}: {result['status']}", flush=True)
    if len(plans) != len(object_ids):
        scene_status["simulation"] = {
            "status": "not_run", "reason": "one or more object-conditioned plans failed",
        }
        scene_status["status"] = "incomplete"
        _write_status(status_path, status)
        return
    ordered = sorted(object_ids, key=lambda identifier: (
        scene_status["objects"][str(identifier)]["object_grasp"]["prealign_start"],
        identifier,
    ))
    windows = tuple((identifier, 0, frame_count) for identifier in ordered)
    scene_status["simulation"] = {"status": "running"}
    _write_status(status_path, status)
    print(f"{robot} {source.name}: MuJoCo {frame_count} full-scene frames", flush=True)
    started = time.monotonic()
    try:
        result = replay(
            source, destination / "full_replay.json",
            destination / "full_trajectory.npz",
            plan_paths=plans, windows=windows, attach_on_any_pad=False,
        )
        scene_status["simulation"] = {
            "status": "complete", "result": str(destination / "full_replay.json"),
            "frames_reached": result["frames_reached"],
            "elapsed_s": time.monotonic() - started,
        }
        scene_status["status"] = "complete"
    except Exception as exc:
        scene_status["simulation"] = {
            "status": "failed", "error": f"{type(exc).__name__}: {exc}",
            "elapsed_s": time.monotonic() - started,
        }
        scene_status["status"] = "incomplete"
    _write_status(status_path, status)
    print(f"{robot} {source.name} simulation: {scene_status['simulation']['status']}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", type=Path, default=Path("data/datasets/hocap_stable_fused_v4"))
    parser.add_argument("--output-root", type=Path, required=True)
    parser.add_argument("--robot", choices=("ur10", "iiwa14"), required=True)
    arguments = parser.parse_args()
    sources = sorted(path for path in arguments.dataset.iterdir() if path.is_dir())
    arguments.output_root.mkdir(parents=True, exist_ok=True)
    status_path = arguments.output_root / "batch_status.json"
    status = {
        "protocol": "full-length object-aware hand-fit IK; official-mesh MuJoCo contact attachment; strict pad contact",
        "robot": arguments.robot,
        "dataset": str(arguments.dataset),
        "episodes": {source.name: {"status": "queued"} for source in sources},
    }
    _write_status(status_path, status)
    for source in sources:
        destination = arguments.output_root / source.name
        destination.mkdir(parents=True, exist_ok=True)
        status["episodes"][source.name]["status"] = "running"
        _write_status(status_path, status)
        try:
            _run_scene(source, destination, arguments.robot, status, status_path)
        except Exception as exc:
            status["episodes"][source.name]["status"] = "failed"
            status["episodes"][source.name]["error"] = f"{type(exc).__name__}: {exc}"
            _write_status(status_path, status)
            print(f"{arguments.robot} {source.name}: failed: {exc}", flush=True)
    if any(row["status"] != "complete" for row in status["episodes"].values()):
        raise SystemExit(1)


if __name__ == "__main__":
    main()
