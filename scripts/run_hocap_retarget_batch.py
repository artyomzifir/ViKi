"""Resume the 28-scene HO-Cap hand-only retarget batch, one process per scene."""

from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import json
import os
import subprocess

import numpy as np
from pathlib import Path

from viki.retarget.archive import load_archive


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _plan_metrics(path: Path) -> dict:
    with load_archive(path) as plan:
        if plan["coordinate_frame"] != "hocap_tag_1":
            raise ValueError(f"unexpected plan frame in {path}")
        config = json.loads(plan["config_json"])
        if (
            config["robot"] != "ur10"
            or config["gripper"] != "robotiq_2f85"
            or config["base_position"] != [0.0, -1.0, 0.0]
            or config["table_plane_z_m"] != 0.0
            or config["approach_sec"] != 0.0
            or config["object_grasp"]
        ):
            raise ValueError(f"unexpected plan configuration in {path}")
        joint_positions = np.asarray(plan["q"])
        if not np.isfinite(joint_positions).all():
            raise ValueError(f"non-finite joint trajectory in {path}")
        return {
            "solver_status": plan["solver_status"],
            "coordinate_frame": plan["coordinate_frame"],
            "frames": len(plan["timestamps"]),
            "metrics": json.loads(plan["metrics_json"]),
            "config": config,
        }


def _process_episode(dataset: Path, episode: Path) -> dict:
    if not (episode / "cln.npz").is_file():
        return {"status": "missing_cln"}
    plan_path = episode / "plan.h5"
    if plan_path.is_file():
        try:
            return {"status": "complete", **_plan_metrics(plan_path)}
        except (OSError, KeyError, ValueError) as exc:
            return {"status": "invalid_existing_plan", "error": str(exc)}
    command = [
        "viki", "retarget", str(episode), "--robot", "ur10",
        "--gripper", "robotiq_2f85", "--base-position", "0", "-1", "0",
        "--table-plane-z-m", "0", "--approach-sec", "0", "--no-object-grasp",
    ]
    environment = os.environ.copy()
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        environment[variable] = "1"
    log_path = episode / "retarget_run.log"
    with log_path.open("w") as log:
        completed = subprocess.run(
            command, stdout=log, stderr=subprocess.STDOUT,
            env=environment, check=False,
        )
    if completed.returncode != 0:
        return {
            "status": "failed", "exit_code": completed.returncode,
            "log": str(log_path.relative_to(dataset)),
        }
    try:
        return {"status": "complete", **_plan_metrics(plan_path)}
    except (OSError, KeyError, ValueError) as exc:
        return {"status": "invalid_new_plan", "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--require-smoke", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        raise ValueError("workers must be between 1 and 4")
    episodes = sorted(path for path in args.dataset.iterdir() if path.is_dir())
    if len(episodes) != 28:
        raise ValueError(f"expected 28 HO-Cap episodes, found {len(episodes)}")
    if args.require_smoke:
        smoke = _plan_metrics(episodes[0] / "plan.h5")
        scores = smoke["metrics"]
        if (
            smoke["frames"] != 676
            or scores["position_rmse_mm"] > 30.0
            or scores["position_p95_mm"] > 50.0
            or scores["orientation_rmse_deg"] > 45.0
            or scores["velocity_repair_max_rad"] > 1e-3
            or scores["min_velocity_limit_margin_rad_s"] < -1e-7
            or scores["min_floor_margin_mm"] < -0.1
            or (
                scores["min_collision_margin"] is not None
                and scores["min_collision_margin"] < -0.0001
            )
        ):
            raise ValueError(f"smoke did not pass numerical/geometry gate: {scores}")
    status_path = args.dataset / "retarget_batch_status.json"
    status = {
        "protocol": "HO-Cap hand-only UR10/Robotiq, base [0,-1,0] in hocap_tag_1, table z=0, no home approach",
        "workers": args.workers,
        "episodes": {episode.name: {"status": "queued"} for episode in episodes},
    }
    if status_path.is_file():
        old = json.loads(status_path.read_text())
        status["episodes"].update(old.get("episodes", {}))
    _write_json(status_path, status)
    with ThreadPoolExecutor(max_workers=args.workers) as pool:
        futures = {
            pool.submit(_process_episode, args.dataset, episode): episode
            for episode in episodes
        }
        for future in as_completed(futures):
            episode = futures[future]
            try:
                result = future.result()
            except Exception as exc:
                result = {"status": "failed", "error": repr(exc)}
            status["episodes"][episode.name] = result
            _write_json(status_path, status)
            print(episode.name, result["status"], flush=True)
    failed = [name for name, row in status["episodes"].items() if row["status"] != "complete"]
    if failed:
        raise SystemExit(f"retarget incomplete: {len(failed)} episodes: {failed}")


if __name__ == "__main__":
    main()
