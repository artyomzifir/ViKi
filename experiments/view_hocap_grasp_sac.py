"""Replay a saved HO-Cap SAC evaluation with verified training inputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

from stable_baselines3 import SAC

from experiments.hocap_grasp_env import HocapGraspEnv


def verified_inputs(report: dict) -> dict[str, Path]:
    names = ("plan", "track", "mesh", "gt_poses")
    inputs = report.get("input_files")
    if not isinstance(inputs, dict) or set(inputs) != set(names):
        raise ValueError("report lacks the complete training-input manifest")
    paths = {}
    for name in names:
        item = inputs[name]
        path = Path(item["path"])
        digest = hashlib.sha256(path.read_bytes()).hexdigest()
        if digest != item["sha256"]:
            raise ValueError(f"{name} differs from the training input")
        paths[name] = path
    return paths


def view(report_path: Path, evaluation_index: int, verify_only: bool = False) -> dict:
    report = json.loads(report_path.read_text())
    paths = verified_inputs(report)
    evaluation = report["evaluation"][evaluation_index]
    checkpoint = Path(report["checkpoint"])
    if not checkpoint.is_file():
        raise FileNotFoundError(checkpoint)
    if verify_only:
        return {"verified": True, "checkpoint": str(checkpoint)}
    environment = HocapGraspEnv(
        paths["plan"], paths["track"], Path(report["model_root"]),
        paths["mesh"], paths["gt_poses"], report["object_name"],
        report["object_id"], report["gt_anchor_frame"],
        report["reset_frame"], max_steps=report.get("max_steps"),
        reward_mode=report.get("reward_mode", "pickup"), render_mode="human",
    )
    try:
        policy = SAC.load(checkpoint, device="cpu")
        state, _ = environment.reset(seed=evaluation["seed"])
        result = {}
        for step_index in range(environment.max_steps):
            action, _ = policy.predict(state, deterministic=True)
            state, _, terminated, truncated, result = environment.step(action)
            time.sleep(0.04)
            if terminated or truncated:
                break
        return {"steps": step_index + 1, "result": result}
    finally:
        environment.close()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--report", type=Path, required=True)
    parser.add_argument("--evaluation-index", type=int, default=0)
    parser.add_argument("--verify-only", action="store_true")
    arguments = parser.parse_args()
    print(json.dumps(view(
        arguments.report, arguments.evaluation_index, arguments.verify_only,
    ), indent=2))
