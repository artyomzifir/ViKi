"""Check that the simulated UR10/gripper hold a task pose under gravity."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from experiments.scene import ARM_JOINTS, load_scene


def verify(episode: Path, model_root: Path) -> dict:
    plan_path = episode / "plan.h5"
    object_model_path = (episode / "intermediates/segmentation/"
                         "sam2.1_hiera_small/object_models.npz")
    model = load_scene(plan_path, object_model_path, model_root)
    data = mujoco.MjData(model)
    with h5py.File(plan_path) as plan:
        arm_target = np.asarray(plan["q"][0], dtype=float)
    arm_qpos = [int(model.jnt_qposadr[model.joint(name).id]) for name in ARM_JOINTS]
    finger_qpos = int(model.jnt_qposadr[model.joint("finger_joint").id])
    data.qpos[:] = model.qpos0
    data.qpos[arm_qpos] = arm_target
    data.ctrl[:6] = arm_target
    data.ctrl[6] = 0.0
    mujoco.mj_forward(model, data)
    for _ in range(int(2.0 / model.opt.timestep)):
        mujoco.mj_step(model, data)
    arm_error = float(np.max(np.abs(data.qpos[arm_qpos] - arm_target)))
    finger_error = float(abs(data.qpos[finger_qpos]))
    result = {
        "arm_hold_error_rad": arm_error,
        "finger_hold_error_rad": finger_error,
        "accepted": arm_error <= 0.01 and finger_error <= 0.01,
    }
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("episode", type=Path)
    parser.add_argument("--models", type=Path, default=Path("models"))
    arguments = parser.parse_args()
    report = verify(arguments.episode, arguments.models)
    print(json.dumps(report, indent=2))
    if not report["accepted"]:
        raise SystemExit(1)
