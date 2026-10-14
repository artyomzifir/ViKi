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


HAND_TO_EE_RPY_DEG = {
    "left": [-175.675, 15.45, -47.922],
    "right": [4.325, 15.45, -47.922],
}


def _hand_to_ee_rpy_deg(episode: Path) -> list[float]:
    metadata = json.loads((episode / "meta.json").read_text())
    if metadata.get("source") != "hocap" or metadata.get("hocap_world_frame") != "tag_1":
        raise ValueError(f"unexpected source/frame in {episode}")
    sides = metadata.get("hocap_mano_sides_meta")
    if not isinstance(sides, list) or len(sides) != 1 or sides[0] not in HAND_TO_EE_RPY_DEG:
        raise ValueError(f"expected exactly one known HO-Cap hand side in {episode}: {sides}")
    return HAND_TO_EE_RPY_DEG[sides[0]]


def _write_json(path: Path, payload: dict) -> None:
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n")
    os.replace(temporary, path)


def _plan_metrics(path: Path) -> dict:
    expected_rpy = _hand_to_ee_rpy_deg(path.parent)
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
            or config["reference_policy"] != "zero"
            or config["object_grasp"]
            or not config["solver"]["collision_enabled"]
            or config["collision_pairs"] < 1
            or config.get("velocity_limit_scale", 1.0) != 1.0
            or not np.allclose(config["hand_to_ee_rpy_deg"], expected_rpy, atol=1e-6, rtol=0)
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
        "--reference-policy", "zero",
        "--hand-to-ee-rpy-deg", *(str(value) for value in _hand_to_ee_rpy_deg(episode)),
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


def _validate_smoke(smoke: dict) -> None:
    scores = smoke["metrics"]
    thresholds = {
        "position_rmse_mm": (None, 30.0),
        "position_p95_mm": (None, 50.0),
        "orientation_rmse_deg": (None, 45.0),
        "velocity_repair_max_rad": (0.0, 1e-3),
        "min_velocity_limit_margin_rad_s": (-1e-7, None),
        "min_urdf_velocity_limit_margin_rad_s": (-1e-7, None),
        "min_floor_margin_mm": (-0.1, None),
        "min_collision_margin": (-0.0001, None),
    }
    failures = []
    if smoke["frames"] != 676 or smoke["solver_status"] != "converged":
        failures.append("frame count or convergence")
    for name, (lower, upper) in thresholds.items():
        value = scores.get(name)
        if (
            value is None or not np.isfinite(value)
            or (lower is not None and value < lower)
            or (upper is not None and value > upper)
        ):
            failures.append(name)
    if failures:
        raise ValueError(f"smoke did not pass numerical/geometry gate: {failures}; {scores}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("dataset", type=Path)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--require-smoke", action="store_true")
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    if args.workers < 1 or args.workers > 4:
        raise ValueError("workers must be between 1 and 4")
    episodes = sorted(path for path in args.dataset.iterdir() if path.is_dir())
    if len(episodes) != 28:
        raise ValueError(f"expected 28 HO-Cap episodes, found {len(episodes)}")
    for episode in episodes:
        _hand_to_ee_rpy_deg(episode)
        if not (episode / "cln.npz").is_file():
            raise ValueError(f"missing cln.npz in {episode}")
        if (episode / "plan.h5").is_file():
            _plan_metrics(episode / "plan.h5")
    if args.require_smoke:
        smoke = _plan_metrics(episodes[0] / "plan.h5")
        _validate_smoke(smoke)
    if args.preflight_only:
        print(f"preflight passed for {len(episodes)} episodes", flush=True)
        return
    status_path = args.dataset / "retarget_batch_status.json"
    status = {
        "protocol": "HO-Cap hand-only UR10/Robotiq, base [0,-1,0] in hocap_tag_1, table z=0, zero IK reference, no home approach",
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
