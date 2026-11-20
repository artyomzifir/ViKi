"""Check Robotiq pad-center geometry against ViKi's saved plan openings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import mujoco
import numpy as np

from experiments.assemble_urdf import load_combined_model


MIMIC_MULTIPLIERS = {
    "finger_joint": 1.0,
    "left_proximal_phalanx_joint": 1.0,
    "left_distal_phalanx_joint": -1.0,
    "right_proximal_phalanx_joint": 1.0,
    "right_finger_joint": 1.0,
    "right_distal_phalanx_joint": -1.0,
}
PAD_OFFSET_M = np.array([0.0, -0.0220203446692936, 0.03242])


def verify(plan_path: Path, model_root: Path) -> dict:
    model = load_combined_model(model_root)
    data = mujoco.MjData(model)
    with h5py.File(plan_path) as plan:
        joints = np.asarray(plan["q"], dtype=float)
        drive = np.asarray(plan["gripper_joint_position"], dtype=float)
        expected_gap = np.asarray(plan["gripper_opening_m"], dtype=float)
    indices = np.linspace(0, len(joints) - 1, min(15, len(joints)), dtype=int)
    arm_names = (
        "shoulder_pan_joint", "shoulder_lift_joint", "elbow_joint",
        "wrist_1_joint", "wrist_2_joint", "wrist_3_joint",
    )
    errors = []
    for frame in indices:
        for index, name in enumerate(arm_names):
            joint_id = model.joint(name).id
            data.qpos[model.jnt_qposadr[joint_id]] = joints[frame, index]
        for name, multiplier in MIMIC_MULTIPLIERS.items():
            joint_id = model.joint(name).id
            data.qpos[model.jnt_qposadr[joint_id]] = drive[frame] * multiplier
        mujoco.mj_forward(model, data)
        points = []
        for name in ("left_distal_phalanx", "right_distal_phalanx"):
            body_id = model.body(name).id
            points.append(data.xpos[body_id]
                          + data.xmat[body_id].reshape(3, 3) @ PAD_OFFSET_M)
        gap = float(np.linalg.norm(points[1] - points[0]))
        errors.append(abs(gap - expected_gap[frame]))
    result = {"frames_checked": len(indices), "max_pad_gap_error_m": max(errors)}
    result["accepted"] = bool(result["max_pad_gap_error_m"] <= 0.002)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("plan", type=Path)
    parser.add_argument("--models", type=Path, default=Path("models"))
    arguments = parser.parse_args()
    report = verify(arguments.plan, arguments.models)
    print(json.dumps(report, indent=2))
    if not report["accepted"]:
        raise SystemExit(1)
